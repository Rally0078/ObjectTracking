import importlib
from pathlib import Path
import os
import json
from utils.logger import setup_logger
from utils.coconames import coconames

import cv2
import torch
from PIL import Image
import pandas as pd

from torchvision import models, transforms
from deep_sort_realtime.deep_sort.track import Track
from deep_sort_realtime.deepsort_tracker import DeepSort

class ObjectTracker:
    def __init__(self, input_file: Path, output_file: Path,
                  model_name: str, verbose: bool, display: bool, device_name=None):
        self.logger = setup_logger('ObjectTracker', verbose=verbose)
        self.coco_classes = coconames
        
        self.model_name = model_name
        self.device = self._get_device(device_name)
        self.display = display
        if self.model_name == 'coco':
            self.model = models.detection.fasterrcnn_resnet50_fpn_v2(
                weights=models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT).to(self.device)
        elif self.model_name == 'fudan':
            self.model = torch.load('model_weights/model_state_dict.pt', weights_only=False).to(self.device)
        self.input_path = input_file
        self.output_path = output_file

        with open(os.path.abspath('./config/deepsort.json')) as f:
            self.deepsort_cfg = json.load(f)
        
        self.max_age = self.deepsort_cfg['max_age']
        self.max_iou_distance = self.deepsort_cfg['max_iou_distance']
        self.n_init = self.deepsort_cfg['n_init']

        self.logger.debug(f"Input path: {str(self.input_path)}")
        self.logger.debug(f"Output path: {str(self.output_path)}")
        self.logger.debug(f"Model name: {self.model_name}")
        self.logger.debug(f"Device: {self.device}")
        self.logger.debug(f"Deepsort config: max_age={self.max_age}, max_iou_distance={self.max_iou_distance}, n_init={self.n_init}")
    
    def run(self):
        self.model = self.model.eval()
        self.deepsort = DeepSort(max_age=self.max_age, max_iou_distance=self.max_iou_distance, n_init=self.n_init)
        transform = transforms.Compose([transforms.ToTensor()])

        cap = cv2.VideoCapture(str(self.input_path))
        self.logger.debug("Video input opened")
        outfile = self._create_video_writer(cap, str(self.output_path))
        self.logger.debug("Video output writer opened")
        frame_count = 0
        self.object_times = {}
        self.score_threshold = 0.9

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            # Preprocess the frame
            pil_img = Image.fromarray(frame)
            img_tensor = transform(pil_img).unsqueeze(0).to(self.device)
            
            # Do the inference with the model
            with torch.no_grad():
                detections = self.model(img_tensor)
            boxes = detections[0]['boxes'].cpu().numpy()
            labels = detections[0]['labels'].cpu().numpy()
            scores = detections[0]['scores'].cpu().numpy()
            
            # Filter the detections by a score threshold
            valid_boxes = boxes[scores > self.score_threshold]
            valid_scores = scores[scores > self.score_threshold]
            valid_cls_ids = labels[scores > self.score_threshold]

            valid_boxes_tensor = torch.tensor(valid_boxes, dtype=torch.float32)
            valid_scores_tensor = torch.tensor(valid_scores, dtype=torch.float32)
            valid_cls_ids_tensor = torch.tensor(valid_cls_ids, dtype=torch.int64)

            # Prepare detections for DeepSORT (box format: [[x1, y1, width, height], score, label])
            detections_deepsort = []
            for box, score, label in zip(valid_boxes_tensor, valid_scores_tensor, valid_cls_ids_tensor):
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
                detections_deepsort.append([[x1, y1, width, height], score, label])
            
            # Update DeepSORT tracker with the current frame's detections
            trackers: list[Track] = self.deepsort.update_tracks(detections_deepsort, frame=frame)

            # Draw bounding boxes, trajectory lines, and bounding box information
            for track in trackers:
                if not track.is_confirmed():
                    continue
                det_cls = track.det_class
                track_id = track.track_id
                if track_id not in self.object_times:
                    self.object_times[track_id] = {"cls": det_cls.cpu().numpy(),
                                            "cls_name": self.coco_classes(det_cls.cpu().numpy()), 
                                            "trajectory": [], 
                                            "entry_time": cap.get(cv2.CAP_PROP_POS_MSEC), 
                                            "exit_time": None}
                    
                self.object_times[track_id]["exit_time"] = cap.get(cv2.CAP_PROP_POS_MSEC)  # Update exit time every frame the object is tracked

                x1, y1, x2, y2 = track.to_tlbr()  # Returns bounding box in the (x1, y1, x2, y2) format
                x_center, y_center = (x1+x2)//2, (y1+y2)//2   #Get center of box

                #Draw trajectory of the bounding box
                self.object_times[track_id]["trajectory"].append((x_center,y_center))
                traj = self.object_times[track_id]["trajectory"]
                for i in range(1, len(traj)):
                    cv2.line(frame, (int(traj[i-1][0]), int(traj[i-1][1])), 
                            (int(traj[i][0]), int(traj[i][1])), (0, 0, 255), 2)
                #Draw circle at centroid
                cv2.circle(frame, (int(x_center), int(y_center)), 4, (255, 255, 0), -1)
                #Draw bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {track_id} Cls = {det_cls}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            if self.display:
                # Display the frame
                cv2.imshow("Object Tracking", frame)
            # Break loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            #Write the output video
            outfile.write(frame)  
        cap.release()
        self.logger.debug("Video input released")
        outfile.release()
        self.logger.debug("Video output writer released")
        cv2.destroyAllWindows()
        torch.cuda.empty_cache()
        self.logger.info(f"Number of unique object IDs = {len(self.object_times.keys())}")
        self._output_tracks()

    def _get_device(self, device_name: str) -> torch.DeviceObjType:
        # ### Choosing device to load the model and frame in
        
        #CUDA on Nvidia
        if device_name is None:
            if torch.cuda.is_available():
                device = torch.device('cuda')
                self.logger.debug(f"Automatically chosen device: {device}")
            #Apple
            elif torch.backends.mps.is_available():
                device = torch.device('mps)')
                self.logger.debug(f"Automatically chosen device: {device}")
            #DirectML (Windows only, on DX12 supported cards)
            elif importlib.util.find_spec("torch_directml") is not None:
                import torch_directml
                device = torch_directml.device()
                self.logger.debug(f"Automatically chosen device: {torch_directml.device_name(0)}")
            #Fallback to CPU
            else:
                device = torch.device('cpu')
                self.logger.debug(f"Automatically chosen device: {device}")
        else:
            device = torch.device(device_name)
        return device
    
    def _create_video_writer(self, video_cap: cv2.VideoCapture, output_filename: str):
        # grab the width, height, and fps of the frames in the video stream.
        frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(video_cap.get(cv2.CAP_PROP_FPS))

        # initialize the FourCC and a video writer object
        fourcc = cv2.VideoWriter_fourcc(*'h264')
        writer = cv2.VideoWriter(output_filename, fourcc, fps,
                                (frame_width, frame_height))
        return writer
    
    def _output_tracks(self):
        df = pd.DataFrame.from_dict(self.object_times, orient='index', columns=['cls', 'cls_name', 'trajectory', 'entry_time', 'exit_time'])
        df.drop(labels='trajectory',axis=1, inplace=True)
        df.index.name ="track_id"
        df.to_csv('output/objects.csv')
        self.logger.info("Object track details saved in output/objects.csv")
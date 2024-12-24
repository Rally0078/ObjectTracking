
# # 0. Imports


import importlib
import cv2
import torch
from PIL import Image
from pathlib import Path
from torchvision import models, transforms
from torchvision.ops import nms
from deep_sort_realtime.deep_sort.track import Track
from deep_sort_realtime.deepsort_tracker import DeepSort


# ### COCO dataset labels


coco_classes_90 = ["background", "person", "bicycle", "car", "motorcycle",
            "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
            "unknown", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
            "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "unknown", "backpack",
            "umbrella", "unknown", "unknown", "handbag", "tie", "suitcase", "frisbee", "skis",
            "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
            "surfboard", "tennis racket", "bottle", "unknown", "wine glass", "cup", "fork", "knife",
            "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
            "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "unknown", "dining table",
            "unknown", "unknown", "toilet", "unknown", "tv", "laptop", "mouse", "remote", "keyboard",
            "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "unknown",
            "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush" ] 


# ### Choosing device to load the model and frame in


#CUDA on Nvidia
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(device)
#Apple
elif torch.backends.mps.is_available():
    device = torch.device('mps)')
    print(device)
#DirectML (Windows only, on DX12 supported cards)
elif importlib.util.find_spec("torch_directml") is not None:
    import torch_directml
    device = torch_directml.device()
    print(torch_directml.device_name(0))
#Fallback to CPU
else:
    device = torch.device('cpu')
    print(device)


# ### Video writer wrapped around for convenience


def create_video_writer(video_cap, output_filename):
    # grab the width, height, and fps of the frames in the video stream.
    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))

    # initialize the FourCC and a video writer object
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    writer = cv2.VideoWriter(output_filename, fourcc, fps,
                             (frame_width, frame_height))
    return writer


# # 1. Loading the Object Detection Model and DeepSORT model
# Two choices for models:
# 1. COCO trained FasterRCNN with ResNet50 backbone
# 2. COCO trained FasterRCNN with ResNet50 backbone that was fine-tuned with Fudan Pedestrian dataset, and classifies only pedestrians(COCO person class).


torch.cuda.empty_cache()
#Default PyTorch weights
model = models.detection.fasterrcnn_resnet50_fpn_v2(weights=models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT).to(device)
#Model with default weights fine-tuned to detect only pedestrians
#model = torch.load('model_weights/model_state_dict.pt', weights_only=False).to(device)
model = model.eval()
deepsort = DeepSort(max_age=60, max_iou_distance=0.5, n_init=4)


# ### Load video paths
# Both input and output paths are loaded


transform = transforms.Compose([transforms.ToTensor()])
video_path = Path.joinpath(Path.cwd(), 'video/macv-obj-tracking-video.mp4')
output_path = Path.joinpath(Path.cwd(), 'video/output.mp4')
print(f"Video is in {video_path}")


# # 2. Do the inference and perform DeepSORT
# Non max supression is applied by this implementation of DeepSORT


cap = cv2.VideoCapture(str(video_path))
outfile = create_video_writer(cap, str(output_path))
frame_count = 0
object_times = {}
score_threshold = 0.9
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    # Preprocess the frame
    pil_img = Image.fromarray(frame)
    img_tensor = transform(pil_img).unsqueeze(0).to(device)
    
    # Do the inference with the model
    with torch.no_grad():
        detections = model(img_tensor)
    boxes = detections[0]['boxes'].cpu().numpy()
    labels = detections[0]['labels'].cpu().numpy()
    scores = detections[0]['scores'].cpu().numpy()
    
    # Filter the detections by a score threshold
    valid_boxes = boxes[scores > score_threshold]
    valid_scores = scores[scores > score_threshold]
    valid_cls_ids = labels[scores > score_threshold]

    valid_boxes_tensor = torch.tensor(valid_boxes, dtype=torch.float32)
    valid_scores_tensor = torch.tensor(valid_scores, dtype=torch.float32)
    valid_cls_ids_tensor = torch.tensor(valid_cls_ids, dtype=torch.int64)

    # Prepare detections for DeepSORT (box format: x1, y1, width, height, score)
    detections_deepsort = []
    for box, score, label in zip(valid_boxes_tensor, valid_scores_tensor, valid_cls_ids_tensor):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        detections_deepsort.append([[x1, y1, width, height], score, label])
    
    # Update DeepSORT tracker with the current frame's detections
    trackers: list[Track] = deepsort.update_tracks(detections_deepsort, frame=frame)

    # Draw bounding boxes, trajectory lines, and bounding box information
    for track in trackers:
        if not track.is_confirmed():
            continue
        det_cls = track.det_class
        track_id = track.track_id
        if track_id not in object_times:
            object_times[track_id] = {"cls": det_cls,
                                      "cls_name": coco_classes_90[det_cls], 
                                      "trajectory": [], 
                                      "entry_time": cap.get(cv2.CAP_PROP_POS_MSEC), 
                                      "exit_time": None}
            
        object_times[track_id]["exit_time"] = cap.get(cv2.CAP_PROP_POS_MSEC)  # Update exit time every frame the object is tracked

        x1, y1, x2, y2 = track.to_tlbr()  # Returns bounding box in the (x1, y1, x2, y2) format
        x_center, y_center = (x1+x2)//2, (y1+y2)//2   #Get center of box

        #Draw trajectory of the bounding box
        object_times[track_id]["trajectory"].append((x_center,y_center))
        traj = object_times[track_id]["trajectory"]
        for i in range(1, len(traj)):
            cv2.line(frame, (int(traj[i-1][0]), int(traj[i-1][1])), 
                     (int(traj[i][0]), int(traj[i][1])), (0, 0, 255), 2)
        #Draw circle at centroid
        cv2.circle(frame, (int(x_center), int(y_center)), 4, (255, 255, 0), -1)
        #Draw bounding box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id} Cls = {det_cls}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Object Tracking", frame)
    #Write the output video
    outfile.write(frame)

    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
outfile.release()
cv2.destroyAllWindows()
torch.cuda.empty_cache()


# Print the duration for each object (frame range)
for obj_id, times in object_times.items():
    entry_time = times["entry_time"]
    exit_time = times["exit_time"]
    duration = exit_time - entry_time
    print(f"ID: {obj_id} {times['cls_name']} appeared from time {entry_time:.3f} ms to time {exit_time:.3f} ms for a of {duration:.3f} ms.")
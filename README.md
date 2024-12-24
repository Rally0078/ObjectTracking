# Object Tracking Project

Uses a FasterRCNN with ResNet 50 backbone for object detection, and DeepSORT for object tracking.

COCO trained weights as well as COCO trained Fudan pedestrian dataset fine-tuned weights can be used.

### Usage:

Explicitly define model and device: 

```bash
    python main.py -i "./video/input.mp4" -o "./output/output.mp4" --model coco --display --verbose --device cuda
```
                               
Model: coco, Device: automatic

```bash
    python main.py -i "./video/input.mp4" -o "./output/output.mp4" --display --verbose
```
                               
Model: coco finetuned with Fudan pedestrian dataset(tracks only pedestrians), Device: automatic

```bash
    python main.py -i "./video/input.mp4" -o "./output/output.mp4" --model fudan 
```
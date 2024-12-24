import argparse
import textwrap

def check_model(model_name: str) -> str:
    if model_name not in ['coco', 'fudan']:
        raise argparse.ArgumentTypeError('Invalid model! Use the model "coco" or "fudan".')
    else:
        return model_name

def check_device(dev_name: str) -> str:
    if dev_name not in ['cpu', 'cuda', 'mpu']:
        raise argparse.ArgumentTypeError('Invalid device! Use the devices cpu, cuda, mpu')
    else:
        return dev_name

def setup_arghandler() -> argparse.ArgumentParser:
    argparser = argparse.ArgumentParser(
        prog="Object Tracking Program",
        description="Tracks COCO dataset objects and draws a bounding box around them with a trail\
              showing the trajectory of the object.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent('''\
            Usage:
            Explicitly define model and device: 
            python main.py -i "./video/input.mp4" -o "./output/output.mp4" --model coco --display --verbose --device cuda
                               
            Model: coco, Device: automatic
            python main.py -i "./video/input.mp4" -o "./output/output.mp4" --display --verbose
                               
            Model: coco finetuned with Fudan pedestrian dataset(tracks only pedestrians), Device: automatic
            python main.py -i "./video/input.mp4" -o "./output/output.mp4" --model fudan 
                               
                               ''')
    )
    
    argparser.add_argument('-i', '--input',
                           type=str, action='store',
                           help='Input video file path',
                           required=True)
    argparser.add_argument('-o', '--output',
                          type=str, action='store',
                          help='Output video file path with all the bounding boxes.',
                          required=True)
    argparser.add_argument('-m', '--model',
                          type=check_model, action='store',
                          metavar='coco|fudan',
                          help='The model to use (coco or fudan). By default it is coco.',
                          default='coco')
    argparser.add_argument('-v','--verbose',
                          action='store_true',
                          help='Show more info on the console.',
                          default=False)
    argparser.add_argument('-d', '--display',
                          action='store_true',
                          help='Show the video with inference in a new window realtime.',
                          default=False)
    argparser.add_argument('-c', '--device',
                           action='store',type=check_device,
                           help='Selects compute device for inference.',
                           default=None)
    return argparser
import logging
import os
from pathlib import Path
import textwrap
import argparse

def check_model(model_name: str) -> str:
    if model_name not in ['coco', 'fudan']:
        raise argparse.ArgumentTypeError('Invalid model! Use the models "coco" or "fudan".')
    else:
        return model_name

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        prog="Object Tracking Program",
        description="Tracks COCO dataset objects and draws a bounding box around them with a trail\
              showing the trajectory of the object.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent('''\
            Usage:
                               ''')
    )
    cwd = os.getcwd()
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
    args = argparser.parse_args()

    input_file = args.input
    input_file = Path(os.path.abspath(input_file))
    
    output_file = args.output
    output_file = Path(os.path.abspath(output_file))

    print(f"Arguments are:")
    print(f"Input path: {str(input_file)}")
    print(f"Output path: {str(output_file)}")
    print(f"Model: {args.model}")
    print(f"Verbose: {args.verbose}")
    print(f"Display OpenCV window: {args.display}")
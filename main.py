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
    argparse.add_argument('-o', '--output',
                          type=str, action='store',
                          help='Output video file path with all the bounding boxes.',
                          required=True)
    argparse.add_argument('-m', '--model',
                          type=str, action='store',
                          metavar='coco|fudan',
                          help='The model to use (coco or fudan). By default it is coco.',
                          type=check_model,
                          default='coco')
    argparse.add_argument('-v','--verbose',
                          type=bool, action='store_true',
                          help='Show more info on the console.',
                          default=False)
    argparse.add_argument('-d', '--display',
                          type=bool, action='store_true',
                          help='Show the video with inference in a new window realtime.',
                          default=False)
    args = argparser.parse_args()

    input_file = args.input.strip(r'\/')
    input_file = input_file.strip(r'"')
    input_file = Path(os.path.normpath(input_file))

    output_file = args.output.strip(r'\/')
    output_file = input_file.strip(r'"')
    output_file = Path(os.path.normpath(input_file))
    output_file = args.output.strip(r'\/')

    print(f"Arguments are:")
    print(f"Input path: {str(input_file)}")
    print(f"Output path: {str(output_file)}")
    print(f"Model: {args.model}")
    print(f"Verbose: {args.verbose}")
    print(f"Display OpenCV window: {args.display}")
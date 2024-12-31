import os
import sys
#If Windows errors out by trying to load multiple OpenMP DLLs(some dependency issue)
#os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
from pathlib import Path
from utils.logger import setup_logger
from utils.arghandling import setup_arghandler
from tracking import ObjectTracker

if __name__ == '__main__':
    argparser = setup_arghandler()
    args = argparser.parse_args()

    logger = setup_logger('Main', args.verbose)
    logger.info(f"Logging started")
    try:
        input_file = args.input
        input_file = Path(os.path.abspath(input_file))
        if not os.path.exists(input_file):
            raise FileNotFoundError
    except FileNotFoundError:
        logger.info(f"An error occurred! Input file at {input_file} is not found!")
        sys.exit(1)
    
    try:
        output_file = args.output
        output_file = Path(os.path.abspath(output_file))
        if not os.path.exists(output_file.parent.parent):
            raise FileNotFoundError
    except FileNotFoundError:
        logger.info(f"An error occurred! The directory {output_file.parent.parent} does not exist!")
        sys.exit(1)
    
    output_file_name = output_file.name
    output_folder = output_file.parent
    output_file.parent.mkdir(exist_ok=True, parents=True)
    
    logger.info("Arguments are:")
    logger.info(f"Input path: {str(input_file)}")
    logger.info(f"Output path: {str(output_file)}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Verbose: {args.verbose}")
    logger.info(f"Display OpenCV window: {args.display}")
    logger.debug(f"DeepSort config: {os.path.abspath('./config/deepsort.json')}")
    tracker = ObjectTracker(input_file, output_file, args.model, args.verbose, args.display, args.device)
    logger.debug(f"Object Tracker created")
    tracker.run()
    logger.debug(f"Object Tracker finished its job")
    #Output HTML
    with open(output_folder / 'videoplayback.html', 'w+') as f:
        f.write(f"""<!DOCTYPE html>
    <html>
        <body>
        <p>Video playback</p>
        <video width="960" height="540" controls>
        <source src="{output_file_name}" type="video/mp4">
        Your browser does not support the video tag.
    </video>

    </body>
</html>""")
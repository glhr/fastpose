from datetime import time
import cv2
from src.system.interface import AnnotatorInterface
from src.utils.drawer import Drawer
import time
from sys import argv

import argparse
from vision_utils.timing import CodeTimer
import glob

parser = argparse.ArgumentParser(description='Directory of PNG images to use for inference.')
parser.add_argument('--input_dir',
                    default="/home/slave/Pictures/pose/pose test input",
                    help='directory of PNG images to run fastpose on')

args = parser.parse_args()
times = []


def start(folder_path, max_persons):

    annotator = AnnotatorInterface.build(max_persons=max_persons)

    for test_image in glob.glob(f"{args.input_dir}/*.png"):
        img_name = test_image.split("/")[-1]

        frame = cv2.imread(test_image)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        tmpTime = time.time()
        persons = annotator.update(frame)
        fps = int(1/(time.time()-tmpTime))

        poses = [p['pose_2d'] for p in persons]

        ids = [p['id'] for p in persons]
        frame = Drawer.draw_scene(frame, poses, ids, fps, 0)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # cv2.imshow('frame', frame)

        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

        # if cv2.waitKey(33) == ord(' '):
            # curr_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            # cap.set(cv2.CAP_PROP_POS_FRAMES, int(curr_frame + 30))

        cv2.imwrite(img_name,frame)

    annotator.terminate()
    # cap.release()
    # cv2.destroyAllWindows()




if __name__ == "__main__":

    print("starting inference on image folder")

    max_persons = 2

    start(f"{args.input_dir}/*.png", max_persons)

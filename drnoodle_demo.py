#!/usr/bin/env python3
from datetime import time
import cv2
from src.system.interface import AnnotatorInterface
from src.utils.drawer import Drawer
import time
from sys import argv

import argparse
from vision_utils.timing import CodeTimer
from vision_utils.logger import get_logger
logger = get_logger()
import glob
import numpy as np
import json

parser = argparse.ArgumentParser(description='Directory of PNG images to use for inference.')
parser.add_argument('--input_dir',
                    default="/home/slave/Pictures/pose/pose test input",
                    help='directory of PNG images to run fastpose on')

args = parser.parse_args()
times = []


def start(folder_path, max_persons, scale=1):

    annotator = AnnotatorInterface.build(max_persons=max_persons)

    for test_image in glob.glob(f"{args.input_dir}/*.png"):
        img_name = f'{test_image.split("/")[-1].split(".")[-2]}-{scale}.{test_image.split(".")[-1]}' if scale<1 else test_image.split("/")[-1]
        logger.info(img_name)

        frame = cv2.imread(test_image)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        dim = (int(frame.shape[1] * scale), int(frame.shape[0] * scale))
        frame = cv2.resize(frame, dim)

        with CodeTimer() as timer:
            persons = annotator.update(frame)

            poses = [p['pose_2d'] for p in persons]

            ids = [p['id'] for p in persons]
            frame = Drawer.draw_scene(frame, poses, ids, fps=None, curr_frame=None)

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # cv2.imshow('frame', frame)

            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

            # if cv2.waitKey(33) == ord(' '):
                # curr_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                # cap.set(cv2.CAP_PROP_POS_FRAMES, int(curr_frame + 30))

        times.append(timer.took)

        cv2.imwrite(img_name,frame)

        json_out = []
        for pose_2d in poses:
            joints = pose_2d.get_joints()
            joints[:,0] = (joints[:,0] * frame.shape[1])
            joints[:,1] = (joints[:,1] * frame.shape[0])
            joints = joints.astype(int)
            keypoints = []
            for i in range(0,joints.shape[0]):
                keypoints.extend([int(joints[i,0]), int(joints[i,1]),1])
            logger.debug(keypoints)
            json_out.append({'keypoints':keypoints})
        json_out_name = '../eval/fastpose-drnoodle/' + img_name + '.predictions.json'
        with open(json_out_name, 'w') as f:
            json.dump(json_out, f)
        logger.info(json_out_name)

    annotator.terminate()
    # cap.release()
    # cv2.destroyAllWindows()
    print(f"Inference took {np.mean(times)}ms per image (avg)" )




if __name__ == "__main__":

    print("starting inference on image folder")

    max_persons = 2

    start(f"{args.input_dir}/*.png", max_persons, scale=1)

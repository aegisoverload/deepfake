import os
import cv2
import argparse
import mediapipe as mp
import numpy as np
import math
from typing import Tuple, Union
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

'''
turn video into frames
'''
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--vdir', type=str, required=True)
    parser.add_argument('--fdir', type=str, required=True)
    parser.add_argument('--skip', type=int, default=50)

    args = parser.parse_args()

    vdir = args.vdir
    fdir = args.fdir
    skip = args.skip
    
    for filename in os.listdir(vdir):
        if filename.endswith(('.mp4')):
            path = os.path.join(vdir, filename)
            name = os.path.splitext(filename)[0]
                
            cap = cv2.VideoCapture(path)
            frame_count = 0
            index = 0

            while True:
                ret, frame = cap.read()

                if not ret:
                    break
                
                frame_count += 1
                    
                if frame_count % skip == 0:
                    index += 1
                    frame_filename = os.path.join(fdir, f"{name}_{index:03d}.jpg") 
                    cv2.imwrite(frame_filename, frame) 

            cap.release()

main()
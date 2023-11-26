#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cv2
import os
import argparse
import numpy as np
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

'''
extract face from images, only one-face image will be extracted
'''
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--fdir', type=str, required=True)
    parser.add_argument('--pdir', type=str, required=True)
    parser.add_argument('--size', type=int, default=192)
    parser.add_argument('--real', type=int, default=1)

    args = parser.parse_args()

    fdir = args.fdir
    pdir = args.pdir
    rf = args.real
    

    # face detector
    base_options = python.BaseOptions(model_asset_path='detector.tflite')
    options = vision.FaceDetectorOptions(base_options=base_options)
    detector = vision.FaceDetector.create_from_options(options)
    
    index = 0
    for filename in os.listdir(fdir):
        if filename.endswith(('.jpg')):
            path = os.path.join(fdir, filename)
            name = os.path.splitext(filename)[0]

            image = mp.Image.create_from_file(path)

            detection_result = detector.detect(image)

            image_copy = np.copy(image.numpy_view())
            rgb_annotated_image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)

            faces = []
            for detection in detection_result.detections:
                bbox = detection.bounding_box
                x1, y1 = int(bbox.origin_x), int(bbox.origin_y)
                x2, y2 = int(bbox.origin_x + bbox.width), int(bbox.origin_y + bbox.height)
                    
                face_region = rgb_annotated_image[y1:y2, x1:x2]
                faces.append(face_region) 


            size = (args.size, args.size)  
            resized_faces = [cv2.resize(face, size) for face in faces]

            if len(resized_faces) == 1:
                processed_filename = os.path.join(pdir, f"{rf}_frame_{index}.jpg")
                index += 1
                cv2.imwrite(processed_filename, resized_faces[0])

                
main()
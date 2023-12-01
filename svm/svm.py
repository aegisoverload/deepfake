import cv2
import numpy as np
import os
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import argparse

FAKE = 'fake'
REAL = 'REAL'


def load_images(dir, label=FAKE):
    images = []
    labels = []

    for filename in os.listdir(dir):
        img = cv2.imread(os.path.join(dir, filename), cv2.IMREAD_GRAYSCALE)
        
        if img is not None:
            images.append(img.flatten())
            
            if label == FAKE:
                labels.append(0)
            else:
                labels.append(1)
    return images, labels

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--fake_dir', type=str, required=True)
    parser.add_argument('--real_dir', type=str, required=True)
    
    args = parser.parse_args()
    
    fake_dir = args.fake_dir
    real_dir = args.real_dir
    
    fakes_images, fake_label = load_images(fake_dir, FAKE)
    real_images, real_label = load_images(real_dir, REAL)

    x = np.array(fakes_images + real_images)
    y = np.array(fake_label + real_label)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # kernel, degree, gamma
    model = svm.SVC()
    model.fit(x_train, y_train)

    # model = joblib.load('svm_model')
    print(f"Accuracy: {accuracy_score(y_test, model.predict(x_test))}")

    joblib.dump(model, 'svm_model')

main()
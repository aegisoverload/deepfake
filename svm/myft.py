import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import argparse

def do_ft(image_path):
    img = cv2.imread(image_path, 0)  
    f = np.fft.fft2(img)  
    fshift = np.fft.fftshift(f)  
    magnitude_spectrum = 20 * np.log(np.abs(fshift))  
    
    return magnitude_spectrum

input_directory = "faces"
output_directory = "dft"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--ft_dir', type=str, required=True)

    args = parser.parse_args()

    input_dir = args.input_dir
    ft_dir = args.ft_dir

    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg"):  
            file_path = os.path.join(input_dir, filename)
            ft_images = do_ft(file_path)

            output = os.path.join(ft_dir, f"{filename}")
            plt.imsave(output, ft_images, cmap='gray')

main()
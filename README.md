deepfake detection project (group 17)
=
## About Our Work
### step 1: Preprocess our video dataset
a sample video


https://github.com/aegisoverload/deepfake/assets/125433863/cbe33a21-3d71-4e0e-8245-e87d811d3d51

extract some frames from the video as training dataset

![image](https://github.com/aegisoverload/deepfake/assets/125433863/e18b588a-29f5-4a51-810a-6a44cd71c038)



try to get attention on our target, extract face from the frame and focus on the details of them.


![image](https://github.com/aegisoverload/deepfake/assets/125433863/abfffc6a-34ee-46f9-8245-9327ae729f59)


### step 2: Model -- CNN
![image](https://github.com/aegisoverload/deepfake/assets/125433863/166d56bb-9eed-437e-828b-54a9988bb0e2)
### step 3: About our model
![image](https://github.com/aegisoverload/deepfake/assets/125433863/db97745b-2b8e-4d93-a8ea-2c54191d1f06)
#### Hyper parameters
![image](https://github.com/aegisoverload/deepfake/assets/125433863/add66a06-a8b7-471f-92b7-271c7b177cf2)
### step 4: Testing and result
![image](https://github.com/aegisoverload/deepfake/assets/125433863/c7dc0462-d3e5-46a7-9110-adbfae6080bf)
![image](https://github.com/aegisoverload/deepfake/assets/125433863/ba0ceeea-ac71-4873-9954-cdb4d8084c23)

### Data distribution

> ![image](https://github.com/aegisoverload/deepfake/assets/125433863/54ca19d6-7c6c-446a-b653-5afdded3e6d6)

## How to train it your self?

Downloading the file in ~/model/deep_fake_CNN.ipynb, and simply use run all on google colab.

## How to use the model

A save of the best model (.ckpt file) will be saved after training.

In ~/detect/detect_deep_fake.py, change the file path and model path at the top of the code. It takes an image and prints the prediction of the model.

### If you are not running on colab, you might need to install the following packages

```bash
pip install numpy
pip install pillow
pip install opencv-python
pip install mediapipe
pip install tqdm
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -U matplotlib
```
Note that the first two code blocks in ~/model/deep_fake_CNN.ipynb are for google colab, you need to delte them if you'r not running on colab.

### References
https://github.com/EndlessSora/DeeperForensics-1.0







# Semantic Segmentation
### Introduction
In this project, you'll label the pixels of a road in images using a Fully Convolutional Network (FCN).

### Setup
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Start

##### Flags on ```main.py```
 - ```save```: [DON'T CHANGE IT] This flag is for avoid errors when testing the functions, due to when training the network we need to save the   models, and this sends an error when we are testing the function.
 - ```testing_functions```: This flag is for launch the tests. If it is true, then the tests are launched. By default is False.
 - ```training```: This flag is for training the network. Therefore, if it is True, then the FCN will be trained. By default is False. 
##### Run
Run the following command to run the project:
```
python main.py
```

### Output
The output image will be a segmented one in which the FCN has labeled each pixel as road (green) or not road (initial value).

# Star Pattern Recognition using CNN 🌠

## Introduction 🚪
Satellite needs attitude determination to locate where its facing. Star sensor is one of many sensor that satellite has to determine its attitude using pattern of stars.

## Table of Contents 📚
- spr_cnn
  - training
  - testing

## Methodology 🛠️
This works consists of two part, training and testing. 

### Training 🔩
The training is a typical neural network workflow using Tensorflow.  
```cd training```

**start.sh**  
The first thing we must do is unzip the data and installing dependencies.  
```sh start.sh```  
This will generate 2 folders, image folder and metadata (csv) folder.  

**checking.ipynb**  
if you want to perform id grouping, go to this notebook. This notebook will show you range of star image then you can examine each manually which one is similiar and group into single group (pick the smallest star ID as group ID).

**augmentation_creation.ipynb**  
This notebook explains the method of transformation. Currently there are two methods, translation and crop. The ```translate()``` or ```crop()``` function perform single operation of transformation. The ```id_translate()``` or ```id_crop()``` function perform multiple operation of transformation within range. We use ```pool``` as multiprocessing tool, for every core (process) perform a single loop operation of transformation. If you do mistake , you can reset the data augmentation process from initial condition on ```restarting``` section.

**model_creation.ipynb**  
This notebook explains how we generate the model file (h5 or tflite). This notebooks consits of several sections.
  
  - DATA GATHERING    
    This section explains how we collect the image data and turn them into numpy array within its metadata. In the end we will get the image array as the         input (x variable) and [ID, rotation, and coordinate] as the feature we want to predict (y variable). We then split those x and y data into 3 parts,           training, validation, testing with portion 8:1:1 respectively.
    
  - MODEL CREATION  
    This section will make us an empty model with tensorflow functional method (branched neural network).
    
  - TRAINING  
    This section will perform training of the empty model with training and validation data that we previously have. Once the training done, it will generate     the h5 file. Then you can evaluae the performance of the model through epoch on the graph contains of loss, ID (label) accuracy, rotation accuracy, and       coordinate accuracy.
    
  - TRAINING FROM H5  
    If you want to increase the epoch from compiled model you can run this section. Then you will get same performance methode as before but the epoch will be     reset. 
    
  - TESTING  
    At this section we perform comparison of prediction and the true data. We evaluate accuracy on the testing set.
    
  - SINGLE TESTING  
    At this section we can check the prediction by single image and then compare it to the real image.
    
  - CONVERTING TO TFLITE  
    from h5 we get, we can convert it to tflite model so that it decrease in size. We can also use the tflite model for prediction. This will be explained in     Testing part.

### Testing 📰

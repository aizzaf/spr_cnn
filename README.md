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
This notebook explains the method of transformation. Currently there are two methods, translation and crop. The ```translate()``` or ```crop()``` function perform single operation of transformation. The ```id_translate()``` or ```id_crop()``` function perform multiple operation of transformation within range. We use ```pool``` as multiprocessing tool, for every core (process) perform a single loop operation of transformation.

### Testing 📰

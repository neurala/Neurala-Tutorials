# Multi-GPU training using Brain Builder and TensorFlow
This repository contains code for multi-GPU training using Brain Builder and Tensorflow. Data annotation is done using Brain Builder and the data loader can read the Brain Builder exported file directly into the model. We use TensorFlow Estimators and the Dataset API to write a UNet to do semantic segmentation. 
## Requirements
1. [Brain Builder](https://info.neurala.com/brain-builder)
2. [TensorFlow 1.11](https://www.tensorflow.org/) or above
3. Python 3.5 or above
4. [Numpy](http://www.numpy.org/) 
5. [Python Image Library (PIL)](https://pillow.readthedocs.io/en/5.3.x/)

## How to run the code
The model is written in the file `utils/UNet.py`. Please note that we've modified the UNet from its original implementation to speed up training and convergence and enable quicker experiments for the purpose of this tutorial.
The dataset loader that can ingest Brain Builder's exported dataset is in the file `utils/TF_Dataset_Loader.py`. It makes use of TensorFlow's Dataset API to read images directly from the Brain Builder's exported file directly into the model(s). 
The training and evaluation code is present in the Jupyter Notebook `Training/multi_gpu_segmentation.ipynb`. You will only need to change the path to the data directory (and may be batch size based on your GPU memory) in order to be able to run this code. 

Please note that we do not provide the dataset we used to write this tutorial since its proprietary to Neurala. However, you can easily tag a small dataset using images or videos using Brain Builder. 

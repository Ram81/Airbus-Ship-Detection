### Aisbus Ship Detection Challenge

A deep learning model that detects all ships in satellite images as quickly as possible.. The solution is based on [U-Net](https://arxiv.org/abs/1505.04597) model. The work in this repository is a step by step implementation from loading the data in pandas data frame to training a deep learning model for the [Kaggle challenge](https://www.kaggle.com/c/airbus-ship-detection).

The ipython notebook `unet-model.ipynb` has the python code for solution. For designing the solution and choosing the model for the solution first we analyze the dataset we have. The dataset has satellite images of ships in ocean or on docks as th input images. The expected output is given in `train_segmentations.csv` in the form of ImageId -> RLE Encoded Vector. The output of the images in data set are encoded using [Run Length Encoding (RLE)](https://en.wikipedia.org/wiki/Run-length_encoding), the expected output for the problem is a RLE mask of ships with background as a two color image.

The goal of choosing U-Net model for the solution is based on the data set we have. After mining the dataset and observing the positive samples (i.e. those samples which have at least one ship in the input image) in the dataset U-Net is one choice for solution based on its application. U-Net is a segmentation model which uses a strong data augmentation to use the available annotated samples more efficiently. Its architecture consists of a contracting path to capture context and a symmetric expanding path that enables precise localization. 

### Results

By training the model for 5 epochs on a batch size of 200 I achieved a baseline accuracy of 50.12%. And yet the model has to be trained on full training data for a larger time to see the progress.

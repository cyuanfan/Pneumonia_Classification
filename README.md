# Pneumonia_Classification
This project train a classifier to predict whether an X-Ray of a patient shows signs of pneumonia or not.

The figure below shows 9 X-Ray sample images, and the image titled '1' indicates that this patient has pneumonia.

![image](https://github.com/cyuanfan/Pneumonia_Classification/blob/master/sample_image/Chest%20X-Ray.png)

# X-Ray Data

This project uses data from the RSNA Pneumonia Detection Challenge.

[X-Ray Data](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data)

# Model Weights

The classifier uses the ResNet-18 architecture, with only the first convolutional layer and the fully connected layer replaced to match the input channel size and the output classes.

[Weights](https://drive.google.com/drive/folders/1FHdNSTZTLcBEAVFpqpE6twyqWiuTvJGo?usp=drive_link)

# Results

Validation Accuracy: 0.817

Validation Precision: 0.570

Validation Recall: 0.775

![image](https://github.com/cyuanfan/Pneumonia_Classification/blob/master/Confusion%20Matrix.png)

# Packages Install
In order to avoid package version conflicts, I suggest using Anaconda to create a virtual environment for this project.

You can enter the following command in the Anaconda Prompt to create an environment and install all the required packages.

```bash
conda env create -f environment.yml
```

# Spot Nuclei, Speed Cures

## Desription
In this project, we aim to spot nuclei in medical images using deep learning techniques. The project is based on the Kaggle competition "Data Science Bowl 2018." The goal is to develop an algorithm that can accurately segment nuclei in images of human cells. By automating this process, medical researchers can save time and effort, ultimately speeding up the discovery of cures for diseases.

## Dataset
The dataset for this project can be downloaded from the following link: [Dataset](https://shrdc-my.sharepoint.com/personal/kong_kah_chun_shrdc_org_my/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fkong_kah_chun_shrdc_org_my%2FDocuments%2FDeep%20Learning%20TTT%2FImage%20Segmentation%2FData%2Fdata-science-bowl-2018%2Ezip&parent=%2Fpersonal%2Fkong_kah_chun_shrdc_org_my%2FDocuments%2FDeep%20Learning%20TTT%2FImage%20Segmentation%2FData&ga=1
). It contains a set of images of human cells and their corresponding masks, which indicate the location of nuclei in the images.

## Installation

To use this code, please follow the steps below:

1. Clone the repository to your local machine using the following command

```
git clone [repository URL]
```

2. Navigate to the project directory:

```
cd [project directory]
```

3. Install the required dependencies by running the following command:

```
pip install pandas numpy tensorflow matplotlib
```
4. Download the dataset from the provided URL.

5. Extract the dataset files into the appropriate folders:

Train File:
- Training images should be placed in the train/inputs folder.
- Training masks should be placed in the train/masks folder.

Test File:
- Test images should be placed in the test/inputs folder.
- Test masks should be placed in the test/masks folder.

## Usage
The code provided in this repository performs the following tasks:

1. Import the required packages and libraries.
2. Load the training and test images.
3. Preprocess the data for training and testing.
4. Create TensorFlow datasets for training and testing.
5. Define hyperparameters for the model.
6. Implement data augmentation using a custom class.
7. Build the dataset pipeline.
8. Inspect some sample data.
9. Develop the U-Net model for nuclei segmentation.
10. Compile the model.
11. Create functions to visualize predictions.
12. Train the model.
13. Deploy the model and visualize predictions on the test dataset.
14. Save the trained model.

## Outputs

- model architecture
![model](https://github.com/FIT003/YPAI03_SpotNuclei_SpeedCures/assets/97938451/ccfd0fed-f500-420a-a5d3-5b8bf23c461c)

- model training
![model_training](https://github.com/FIT003/YPAI03_SpotNuclei_SpeedCures/assets/97938451/17863791-e889-49dc-8943-4c4842219359)

- loss
![epoch_loss](https://github.com/FIT003/YPAI03_SpotNuclei_SpeedCures/assets/97938451/04f0d505-a20c-42ef-9848-11373c051b53)

- accuracy
![epoch_accuracy](https://github.com/FIT003/YPAI03_SpotNuclei_SpeedCures/assets/97938451/696ea786-7a12-439f-89dc-86f5b9e7441a)

## Credits
URL: https://www.kaggle.com/competitions/data-science-bowl-2018/overview






























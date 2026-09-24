# Forest Fire Detection using Deep Learning

A CNN that looks at an image and says whether it shows a forest fire or not. I built it during my Edunet Foundation AI/ML internship, working in Google Colab / Kaggle.

## Problem

Forest fires spread fast, and watching for them by hand is slow. The idea here is a model that can classify camera or drone images as `fire` or `nofire`, which could later feed an early-warning system.

## Dataset

[The Wildfire Dataset](https://www.kaggle.com/datasets/elmadafri/the-wildfire-dataset) from Kaggle, split into two classes (`fire`, `nofire`):

| Split | Images |
|---|---|
| Train | 1,887 |
| Validation | 402 |
| Test | 410 |

## What's in this repo

- `Forest_Fire_Detection_using_DL.ipynb` : the baseline CNN, trained from scratch
- `Forest_Fire_Detection_Improved.ipynb` : the same baseline plus a transfer learning version (see below)
- `week_3_Project_PPT_Template.pdf` : the project slides

## Approach

**1. Baseline CNN**

- Images resized to 150x150 and scaled to 0-1
- 3 x (Conv2D + MaxPooling) blocks with 32, 64 and 128 filters
- Flatten, Dense(512), Dropout(0.5), then a sigmoid output
- Adam optimizer, binary cross-entropy, 12 epochs, batch size 32

**2. Improved version**

- Data augmentation on the training set (rotation, zoom, shifts, horizontal flip)
- MobileNetV2 pretrained on ImageNet as a frozen feature extractor, with a small Dense head on top
- 224x224 input
- EarlyStopping and ModelCheckpoint on validation loss
- Confusion matrix and classification report on the test set

## Results

The baseline CNN reached about **79.7% accuracy on the test set** (validation accuracy ended around 79%). Training and validation curves stay fairly close, so it isn't badly overfitting, but there is clear room to improve.

The improved notebook is set up to be run and compared against this baseline. Results for it will be added here once I've run it.

## Tools

Python, TensorFlow / Keras, OpenCV, NumPy, Matplotlib, Seaborn, scikit-learn, Google Colab / Jupyter

## How to run

1. Open either notebook in Colab or Kaggle (a GPU makes training much quicker)
2. The first cell downloads the dataset with `kagglehub`
3. Run the cells top to bottom
4. To try a single image, use the `predict_fire('path/to/image.jpg')` function in the baseline notebook

## Limitations

- The dataset is small, so the model may struggle with unusual scenes such as heavy smoke, sunsets or fog
- It only classifies whole images, it doesn't locate the fire
- It hasn't been tested on live video or real camera feeds

## Future work

- Compare the MobileNetV2 model against the baseline properly
- Try object detection (for example YOLO) to mark where the fire is
- Test on video frames

## Author

Shaik Asiya Thasleem  
[LinkedIn](https://linkedin.com/in/asiya-thasleem) | [GitHub](https://github.com/Asiya-hub)

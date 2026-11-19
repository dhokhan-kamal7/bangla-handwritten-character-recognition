# Bangla Handwritten Character Recognition

## Project Overview

This project develops a deep learning-based system for recognizing handwritten Bangla characters using a Convolutional Neural Network (CNN). It includes a Streamlit interface where users can draw Bangla handwriting and receive model predictions.

The project explores image preprocessing, model training, evaluation, experiment tracking, and interactive handwriting recognition.

## Objectives

- Develop a CNN-based Bangla handwritten character classification model.
- Preprocess handwritten character images for model training.
- Train and evaluate the model using the BanglaLekha-Isolated dataset.
- Track training experiments using MLflow.
- Build an interactive handwriting interface using Streamlit.
- Provide a reproducible project structure with Docker support.

## Dataset

The project uses the **BanglaLekha-Isolated** dataset.

- Approximately 166,105 handwritten character images.
- 84 character classes.
- Dataset source: [BanglaLekha-Isolated – Mendeley Data](https://data.mendeley.com/datasets/hf6sf8zrkc/2)

The dataset contains isolated characters rather than complete handwritten words.

The dataset is not included in this repository. It must be downloaded separately and placed in the appropriate project directory.

## Methodology

1. **Data Preparation:** Load handwritten character images from the dataset and prepare them for model training.
2. **Preprocessing:** Resize images and normalize pixel values.
3. **Model Development:** Build a CNN for handwritten Bangla character classification.
4. **Model Training:** Train the CNN using TensorFlow and Keras with two hyperparameter configurations.
5. **Experiment Tracking:** Record training parameters, metrics, and model artifacts using MLflow.
6. **Application Development:** Use Streamlit to provide an interactive drawing interface and display predictions.

## Technologies Used

- Python
- TensorFlow and Keras
- Convolutional Neural Network (CNN)
- Streamlit
- MLflow
- NumPy
- Pillow
- Docker

## Main Project Files

| File | Description |
|---|---|
| `train.py` | Model training and MLflow experiment tracking |
| `app.py` | Streamlit handwriting recognition interface |
| `labels.json` | Character label mapping |
| `requirements.txt` | Python dependencies |
| `Dockerfile` | Docker configuration |
| `README.md` | Project documentation |
| `github_link.txt` | Public GitHub repository link |

## Running the Project

Install the dependencies listed in `requirements.txt`.

Download and extract the BanglaLekha-Isolated dataset into the expected dataset directory.

Train the model using the training script and the dataset image directory. The training process records experiments in MLflow and saves the trained model.

Launch the MLflow interface to inspect the training experiments and their metrics.

Run the Streamlit application to draw handwritten Bangla input and view the model's prediction.

## Key Features

- CNN-based handwritten Bangla character classification.
- Interactive handwriting input through Streamlit.
- Model training and evaluation.
- MLflow experiment tracking.
- Two recorded training configurations.
- Docker support.

## Limitations

- The training dataset contains isolated characters, not complete handwritten words.
- The current application uses heuristic segmentation and character prediction to attempt whole-word recognition.
- Whole-word predictions may be inaccurate or incomplete.
- Recognition performance can vary depending on handwriting style, drawing quality, and character spacing.
- Reliable end-to-end handwritten word recognition requires word-level training data and a suitable sequence recognition model.

## Future Improvements

- Improve character classification accuracy.
- Use word-level Bangla handwriting datasets.
- Explore sequence-based recognition architectures.
- Improve character segmentation and word reconstruction.
- Evaluate performance on more diverse handwriting samples.

## Author

**MD. Dhokhan Kamal**

Student ID: **1000129292**


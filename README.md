# Bangla Handwritten Word Recognition System

**Student name:** MD. Dhokhan Kamal  
**Student ID:** 1000129292

A Streamlit prototype for recognizing a handwritten Bangla word by segmenting the drawing into character-like units, classifying each unit with a CNN, and concatenating the predictions into one output string.

## Important dataset/approach note

The official [BanglaLekha-Isolated v2 dataset](https://data.mendeley.com/datasets/hf6sf8zrkc/2) contains 166,105 **isolated character images** in 84 classes (50 basic characters, 10 numerals, and 24 selected compound characters). It is **not a word-image dataset**. Therefore this project trains a character classifier, not an end-to-end word recognizer.

The Streamlit page accepts a complete drawing and displays one combined Bangla output string. It uses vertical-projection segmentation followed by character classification. This provides a word-shaped output, but it cannot guarantee correct recognition of connected handwriting. The model does not learn word-level context or all Bangla vowel signs. Do not report word-level accuracy as if it were measured on a word-labelled test set.

## Approach

1. Load the numeric class folders (1–84) from BanglaLekha-Isolated.
2. Convert each image to grayscale, resize to 64×64, and normalize to [0, 1].
3. Train a three-block CNN with batch normalization, dropout, augmentation, and Adam.
4. Run two experiments with different learning rates/dropout settings. MLflow logs parameters, validation metrics, and the saved model for each run.
5. In the app, threshold the canvas image, segment it with a simple vertical-projection heuristic, classify each segment, and concatenate predicted labels into one Bangla string.

`labels.json` maps the zero-based model output index to the corresponding Bangla character. Folder 1 maps to index 0, folder 2 to index 1, and so on.

## Project structure

```text
bangla-ocr-assignment/
├── train.py
├── app.py
├── requirements.txt
├── Dockerfile
├── README.md
├── labels.json
├── github_link.txt
├── data/                 # dataset goes here; not submitted
├── models/               # generated model; not submitted
├── artifacts/mlflow/     # generated MLflow runs; not submitted
└── screenshots/          # add the required screenshots after running
```

## Setup

Python 3.10 is recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## Dataset

1. Download `BanglaLekha-Isolated.zip` from [Mendeley Data v2](https://data.mendeley.com/datasets/hf6sf8zrkc/2).
2. Extract it so the 84 numeric class folders are below `data/`, commonly:

```text
data/
└── BanglaLekha-Isolated/
    └── Images/
        ├── 1/
        ├── 2/
        └── ...
        └── 84/
```

The training script searches recursively for the numeric class folders.

## Train and track experiments

```bash
python train.py
```

Each invocation creates two MLflow runs:
- `baseline_lr_1e-3`
- `regularized_lr_3e-4`

For a quick smoke test, use a smaller subset (not suitable for final reported results):

```bash
python train.py --max-per-class 100 --epochs 2 --batch-size 32
```

Start MLflow:

```bash
mlflow ui --backend-store-uri file:./artifacts/mlflow --host 0.0.0.0 --port 5000
```

Open `http://localhost:5000`. Confirm there are at least two completed runs and capture the experiment page as `screenshots/mlflow_experiment.png`.

## Run Streamlit locally

```bash
streamlit run app.py
```

Open `http://localhost:8501`, draw one word, and click **Recognize whole word**. Capture the result page as `screenshots/streamlit_app.png`.

## Docker

Build:

```bash
docker build -t bangla-ocr-app:0.1 .
```

Run:

```bash
docker run --rm -p 8501:8501 bangla-ocr-app:0.1
```

Open `http://localhost:8501`. The trained model must exist at `models/model.keras` before running the container. Train locally first; generated model files are intentionally excluded from the source ZIP.

## Limitations

- The source dataset has isolated characters, not handwritten words; no true word-level training or word-level benchmark is available from it.
- The vertical-projection segmentation is heuristic and can fail when Bangla matra/headlines connect multiple letters.
- The dataset's 84 classes do not represent every Bangla conjunct or all vowel signs/কার in their attached forms.
- Concatenating character predictions does not use a language model or dictionary context.
- Mouse/touch canvas strokes may differ from scanned handwriting.
- A production-quality whole-word recognizer should be trained on labelled word/line images with a sequence architecture such as CNN–BiLSTM–CTC or a Transformer OCR model.

## Submission checklist

- [ ] Train both MLflow runs and verify them in the MLflow UI.
- [ ] Save `screenshots/mlflow_experiment.png`.
- [ ] Draw a word in Streamlit and save `screenshots/streamlit_app.png`.
- [ ] Push this project to a **public** GitHub repository.
- [ ] Replace the placeholder in `github_link.txt` with the public repository URL.
- [ ] Keep the dataset, trained model, MLflow run store, virtual environment, and caches out of the source ZIP.

## Dataset citation

Biswas, M., Islam, R., Shom, G. K., Shopon, M., Mohammed, N., Momen, S., & Abedin, A. (2017). *BanglaLekha-Isolated: A multi-purpose comprehensive dataset of Handwritten Bangla Isolated characters*. Data in Brief, 12, 103–107. https://doi.org/10.1016/j.dib.2017.03.035

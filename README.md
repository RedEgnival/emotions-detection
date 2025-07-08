# Emotion Detection from Text

This project uses machine learning to classify emotions (happy, sad, anger, etc.) from short English text messages. It utilizes TF-IDF features and a Logistic Regression classifier for emotion detection.

## Dataset

The dataset is from Kaggle: [Emotion Dataset](https://www.kaggle.com/datasets/djonesdev/emotion-dataset), containing three files:
- `train.txt`
- `val.txt`
- `test.txt`

Each file has two columns: `text` and `emotion`, separated by a semicolon (`;`).

## Project Structure

- `sentiment-analysis.ipynb`: Main Jupyter notebook containing all code for preprocessing, training, and evaluation.
- `README.md`: Project overview and usage instructions.
- `requirements.txt`: Python dependencies.

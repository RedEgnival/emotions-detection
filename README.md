# Emotion Detection App 🎭

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://emotions-detection-nhinjjaenxd4iziqfznwxx.streamlit.app/)

A real-time emotion detection system that classifies text into 6 emotional categories using machine learning.

![App Screenshot](./assets/screenshot.png) *(Add screenshot later)*

## Features ✨

- **Six Emotion Detection**: Anger, Fear, Joy, Love, Sadness, Surprise
- **Probability Visualization**: See confidence levels for all emotions
- **Emoji Integration**: Visual feedback with expressive emojis
- **Responsive Design**: Works on desktop and mobile devices

## Tech Stack 🛠️

- **Frontend**: Streamlit
- **Backend**: Python
- **Machine Learning**:
  - Scikit-learn (Logistic Regression)
  - TF-IDF Vectorization
- **Utilities**: Joblib, Pandas, NumPy

## How It Works 🔍

1. User inputs text via text area
2. Model processes text through pipeline:
   ```mermaid
   graph LR
   A[Raw Text] --> B(TF-IDF Vectorization)
   B --> C(Logistic Regression)
   C --> D[Emotion Prediction]



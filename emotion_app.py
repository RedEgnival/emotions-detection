import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Set page config
st.set_page_config(
    page_title="Emotion Detection App",
    page_icon="😊",
    layout="wide"
)

# Emotion emojis
emotion_emojis = {
    'anger': '😠',
    'fear': '😨',
    'joy': '😊',
    'love': '❤️',
    'sadness': '😢',
    'surprise': '😲'
}

# Load model components
@st.cache_resource
def load_components():
    try:
        # Load the complete pipeline (should include vectorizer)
        model = joblib.load('emotion_model.pkl')
        # Load label encoder
        le = joblib.load('label_encoder.pkl')
        return model, le
    except Exception as e:
        st.error(f"Error loading model components: {str(e)}")
        st.stop()

model, le = load_components()

# Verify model is a pipeline
if not hasattr(model, 'steps'):
    st.error("""
    ❌ Model is not a complete pipeline. 
    Please ensure your 'emotion_model.pkl' includes both vectorizer and classifier.
    """)
    st.stop()

# App layout
st.title("Emotion Detection App")
st.write("""
This app predicts the emotion behind text using a machine learning model.
""")

# Text input
user_input = st.text_area(
    "Enter your text here:", 
    "I'm feeling excited about this project!",
    height=150
)

# Prediction function
def predict_emotion(text):
    try:
        # Predict using the full pipeline (automatically handles vectorization)
        prediction = model.predict([text])[0]  # Returns numpy array
        probabilities = model.predict_proba([text])[0]
        return prediction, probabilities
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        return None, None

# Prediction section
if st.button("Predict Emotion", type="primary"):
    if not user_input.strip():
        st.warning("Please enter some text to analyze!")
    else:
        with st.spinner("Analyzing emotion..."):
            # Get prediction
            prediction_idx, probabilities = predict_emotion(user_input)
            
            if prediction_idx is not None:
                # Convert prediction to label
                emotion = le.inverse_transform([prediction_idx])[0]
                
                # Create probability dictionary
                emotion_probs = {
                    le.classes_[i]: prob 
                    for i, prob in enumerate(probabilities)
                }
                
                # Display results
                st.subheader("Results")
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.metric(
                        "Predicted Emotion", 
                        f"{emotion.capitalize()} {emotion_emojis.get(emotion, '')}",
                        delta=f"{emotion_probs[emotion]:.1%} confidence"
                    )
                
                with col2:
                    st.write("**Emotion Probabilities:**")
                    prob_df = pd.DataFrame.from_dict(
                        emotion_probs, 
                        orient='index', 
                        columns=['Probability']
                    ).sort_values('Probability', ascending=False)
                    
                    for emo, prob in prob_df.itertuples():
                        st.progress(
                            float(prob), 
                            text=f"{emo.capitalize()} {emotion_emojis.get(emo, '')}: {prob:.1%}"
                        )

# Sidebar with info
with st.sidebar:
    st.header("About")
    st.write("""
    This model recognizes six emotions:
    """)
    for emotion in le.classes_:
        st.write(f"- {emotion.capitalize()} {emotion_emojis.get(emotion, '')}")
    
    st.markdown("---")
    st.write("""
    **Model Details:**
    - Algorithm: Logistic Regression
    - Vectorizer: TF-IDF
    - Accuracy: ~86%
    """)
    
    st.markdown("---")
    st.write("Created with ❤️ using Streamlit")

# Footer
st.markdown("---")
st.caption("Tip: Try phrases like 'I'm so angry!' or 'This makes me happy'")
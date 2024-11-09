import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from sentence_transformers import SentenceTransformer
import time

# Show a loading spinner until the model is loaded
with st.spinner('Loading model and resources...'):
    # Load the trained Keras model
    model = load_model('travel_neural_model.h5')

    # Load the LabelEncoder
    label_encoder = joblib.load('label_encoder.pkl')

    # Initialize the SentenceTransformer model for embedding sentences
    embedder = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    # Simulate a small delay to mimic model loading (optional, if you want to slow it down slightly for user experience)
    time.sleep(1)

# Function to generate the embedding for a sentence
def embed_sentence(sentence):
    return embedder.encode([sentence])[0]  # Return the first element of the batch (embedding for the sentence)

# Streamlit app UI setup
st.title("Text Classification with Travel Model")
st.write("Enter a sentence below, and the model will predict the class for it.")

# Text input for new sentence
user_input = st.text_input("Enter a sentence:")

# Button to trigger prediction
if st.button("Predict"):
    if user_input:
        # Show a loading spinner while predicting
        with st.spinner('Making prediction...'):
            # Generate embedding for the input sentence
            new_embedding = embed_sentence(user_input)
            new_embedding = new_embedding.reshape(1, -1)  # Reshape to (1, embedding_dim)
            
            # Make the prediction using the trained model
            y_pred_new = model.predict(new_embedding)
            
            # Get the predicted class index
            predicted_class_index = np.argmax(y_pred_new, axis=1)[0]
            
            # Convert the numerical label back to class name
            predicted_class = label_encoder.inverse_transform([predicted_class_index])
            
            # Show the prediction result
            st.write(f"The predicted class for the sentence is: {predicted_class[0]}")
    else:
        st.warning("Please enter a sentence before clicking 'Predict'.")

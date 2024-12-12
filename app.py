import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Ensure required libraries are installed
try:
    import matplotlib
    import seaborn
    import sklearn
    import wordcloud
except ImportError:
    raise ImportError("Make sure to install required packages: pandas, numpy, matplotlib, seaborn, scikit-learn, wordcloud")

# Streamlit app title
st.title("Depression Detection App")

# File uploader for dataset
uploaded_file = st.file_uploader("cleaned_depression_data.csv", type="csv")

if uploaded_file:
    # Load the dataset
    try:
        df = pd.read_csv(uploaded_file, encoding='unicode_escape')
        st.write("Dataset Preview:")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Error loading the file: {e}")

    # Split the dataset into features (text) and target (label)
    X = df['text']
    y = df['label']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a TfidfVectorizer object
    vectorizer = TfidfVectorizer()

    # Fit the vectorizer to the training data and transform both the training and testing data
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    # Train an SVM model on the training data
    model = svm.SVC()
    model.fit(X_train_vectorized, y_train)

    # Make predictions on the test data
    y_pred = model.predict(X_test_vectorized)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Model Accuracy: {accuracy:.2f}")

    # Instructions for the user
    st.write("This app uses a machine learning model to predict if a given text indicates depression. Please enter a sentence or text in the input box below.")

    # User input for text
    user_input = st.text_area("Enter your sentence or words:", height=150)

    # Prediction button
    if st.button("Predict"):
        if user_input.strip():
            try:
                # Transform the input using the loaded vectorizer
                input_vectorized = vectorizer.transform([user_input]).toarray()
                
                # Make prediction
                prediction = model.predict(input_vectorized)
                
                # Display the result
                if prediction[0] == 1:
                    st.error("The model predicts: You may be experiencing depression. Please consider reaching out to a mental health professional for support.")
                else:
                    st.success("The model predicts: You are not experiencing depression. Stay positive and take care of your mental health!")
            except Exception as e:
                st.error(f"Error during prediction: {e}")
        else:
            st.warning("Please enter some text for prediction.")

# Footer
st.write("\n---\n")
st.caption("Disclaimer: This app is for informational purposes only and is not a substitute for professional mental health advice. If you are feeling distressed, please seek help from a licensed professional.")
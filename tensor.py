import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

st.title("Iris Species Classification using Neural Networks")

# Upload CSV file
uploaded_file = st.file_uploader("Upload CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Raw Dataset")
    st.dataframe(df.head())

    # Preprocessing
    le = LabelEncoder()
    df['species'] = le.fit_transform(df['species'])
    X = df.drop('species', axis=1)
    y = to_categorical(df['species'])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Define model architecture
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(X.shape[1],)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    # Compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    if st.button("Train Model"):
        history = model.fit(X_train, y_train, epochs=50, validation_split=0.2, verbose=0)

        # Evaluate model
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        st.success(f"Test Accuracy: {accuracy:.4f}")
        st.info(f"Test Loss: {loss:.4f}")

        # Plot loss
        st.subheader("Training & Validation Loss")
        fig, ax = plt.subplots()
        ax.plot(history.history['loss'], label='Training Loss')
        ax.plot(history.history['val_loss'], label='Validation Loss')
        ax.set_title("Loss Over Epochs")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

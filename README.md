# Iris Classification using Neural Networks (Streamlit App)

This project builds a **simple feed-forward neural network** using **TensorFlow/Keras** to classify iris flower species. It features a **Streamlit interface** for uploading data, training the model, and visualizing the training process.

---

## Features

- Upload any CSV file formatted like the Iris dataset.
- Automatic data preprocessing (label encoding, scaling).
- Neural network architecture with:
  - 1 input layer
  - 2 hidden layers (ReLU activation)
  - 1 output layer (softmax activation for multi-class classification)
- Live training on button click.
- Visualization of:
  - **Training vs Validation Loss**
  - **Model Accuracy**

---

## Model Architecture

```python
Sequential([
    Dense(64, activation='relu', input_shape=(4,)),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')
])
```
## Demo 

https://github.com/user-attachments/assets/9966a4c2-c83f-4d8a-8a20-1d85decab602

---

## Installation

Clone this repository or download the code.

Install dependencies:

```bash
pip install -r requirements.txt
```
Or manually:
```bash
pip install streamlit pandas matplotlib scikit-learn tensorflow
```
Run the app:
```bash
streamlit run tensor.py
```

# Character Prediction using RNN and LSTM

This project is a Streamlit-based web application that trains a Recurrent Neural Network (RNN) or Long Short-Term Memory (LSTM) model to predict the next character in a sequence based on user-provided text.

The application allows users to enter custom training text, train a model, and predict the next character for a given 4-letter sequence.

---

## Project Overview

The application demonstrates sequence modeling using deep learning techniques. It builds a character-level language model using either:

- Vanilla RNN
- LSTM (Long Short-Term Memory)

The model learns character patterns from input text and predicts the most probable next character.

---

## Features

- User-provided training text
- Choice between RNN and LSTM models
- One-hot encoding for character representation
- Real-time model training
- Interactive next-character prediction
- Streamlit web interface

---

## Technologies Used

- Python
- Streamlit
- PyTorch
- NumPy

---

## How It Works

1. The user enters training text.
2. A character vocabulary is created from the unique characters in the text.
3. Input sequences of fixed length (4 characters) are generated.
4. The model is trained using CrossEntropyLoss and Adam optimizer.
5. After training, the user provides a 4-character sequence.
6. The model predicts the next character based on learned patterns.

---

## Model Architecture

### RNN Model
- Input Layer (One-hot encoded characters)
- RNN layer
- Fully connected output layer

### LSTM Model
- Input Layer (One-hot encoded characters)
- LSTM layer
- Fully connected output layer

Hidden layer size: 128 units  
Sequence length: 4  

---

## Project Structure

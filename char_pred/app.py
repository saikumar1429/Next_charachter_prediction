
import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

st.title("🔤 Character Prediction using RNN / LSTM")
text = st.text_input("Enter training text:", value="hello world")
seq_length = 4
model_type = st.selectbox("Choose Model Type", ["LSTM", "RNN"])
train_button = st.button("Train Model")

# Build vocab
chars = sorted(set(text))
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}
num_chars = len(chars)

# Data prep
def prepare_data(text):
    X, y = [], []
    for i in range(len(text) - seq_length):
        input_seq = text[i:i+seq_length]
        output_char = text[i+seq_length]
        X.append([char_to_idx[c] for c in input_seq])
        y.append(char_to_idx[output_char])
    X = torch.tensor(X)
    y = torch.tensor(y)
    X = F.one_hot(X, num_classes=num_chars).float()
    return X, y

# Models
class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])

class CharLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# Training
def train_model(model, X, y):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(200):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
    return model

# TRAIN button
if train_button:
    if len(text) < seq_length + 1:
        st.warning("Text must be at least 5 characters long.")
    else:
        X, y = prepare_data(text)
        if model_type == "LSTM":
            model = CharLSTM(num_chars, 128, num_chars)
        else:
            model = CharRNN(num_chars, 128, num_chars)
        model = train_model(model, X, y)
        st.session_state.model = model
        st.session_state.char_to_idx = char_to_idx
        st.session_state.idx_to_char = idx_to_char
        st.session_state.num_chars = num_chars
        st.success(f"{model_type} model trained successfully!")

# Prediction block
if "model" in st.session_state:
    test_input = st.text_input("Enter 4-letter input for prediction (e.g. 'hell')")
    if st.button("Predict Next Character"):
        if len(test_input) != 4 or any(c not in st.session_state.char_to_idx for c in test_input):
            st.warning("Enter a valid 4-letter sequence using training characters.")
        else:
            input_seq = torch.tensor([[st.session_state.char_to_idx[c] for c in test_input]])
            input_seq = F.one_hot(input_seq, num_classes=st.session_state.num_chars).float()
            model = st.session_state.model
            output = model(input_seq)
            pred_idx = torch.argmax(output).item()
            pred_char = st.session_state.idx_to_char[pred_idx]
            st.success(f"'{test_input}' → **'{pred_char}'**")

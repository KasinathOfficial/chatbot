import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import nltk
import streamlit as st
from nltk.stem import WordNetLemmatizer


# Download NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Load training data
with open("train.json", "r") as file:
    data = json.load(file)

# Text processing
lemmatizer = WordNetLemmatizer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence.lower())

def stem(word):
    return lemmatizer.lemmatize(word.lower())

def bag_of_words(tokenized_sentence, words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag

# Prepare training data
all_words = []
tags = []
x_train = []
y_train = []
for intent in data["intents"]:
    tags.append(intent["tag"])
    for pattern in intent["patterns"]:
        words = tokenize(pattern)
        all_words.extend(words)
        x_train.append(words)
        y_train.append(intent["tag"])
all_words = sorted(set([stem(w) for w in all_words if w not in ["?", "!", "."]]))
tags = sorted(set(tags))

x_train_encoded = [bag_of_words(sentence, all_words) for sentence in x_train]
y_train_encoded = [tags.index(tag) for tag in y_train]
x_train_encoded = np.array(x_train_encoded)
y_train_encoded = np.array(y_train_encoded)

# Define Neural Network
class ChatBotModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ChatBotModel, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        return self.layer3(x)

# Train the model
input_size = len(all_words)
hidden_size = 8
output_size = len(tags)
model = ChatBotModel(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

dataset = [(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)) for x, y in zip(x_train_encoded, y_train_encoded)]
for epoch in range(1000):
    total_loss = 0
    for x_batch, y_batch in dataset:
        optimizer.zero_grad()
        output = model(x_batch)
        loss = criterion(output.unsqueeze(0), y_batch.unsqueeze(0))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

# Chatbot response function
def chatbot_response(user_input):
    tokenized_sentence = tokenize(user_input)
    bow = bag_of_words(tokenized_sentence, all_words)
    bow_tensor = torch.tensor(bow, dtype=torch.float32)
    output = model(bow_tensor)
    predicted_tag = tags[torch.argmax(output).item()]
    for intent in data["intents"]:
        if intent["tag"] == predicted_tag:
            return random.choice(intent["responses"])

# Streamlit UI
st.title("🤖 AI Chatbot")
st.write("Ask me anything!")

# Input field
user_input = st.text_input("You:", "")

if st.button("Send"):
    if user_input:
        response = chatbot_response(user_input)
        st.text_area("Bot:", value=response, height=100, max_chars=None, key="response")

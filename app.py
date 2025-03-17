import json
import random
import numpy as np
import nltk
# ✅ FIX: Download required NLTK models
nltk.download('punkt')  
nltk.download('wordnet')  
import streamlit as st
import tensorflow as tf
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Load chatbot intents
intents = {
    "intents": [
        {
            "tag": "greeting",
            "patterns": ["Hello", "Hi", "Hey", "How are you?", "Is anyone there?"],
            "responses": ["Hello!", "Hi there!", "Hey!", "How can I help you?"]
        },
        {
            "tag": "goodbye",
            "patterns": ["Bye", "See you later", "Goodbye"],
            "responses": ["Goodbye!", "See you soon!", "Take care!"]
        },
        {
            "tag": "thanks",
            "patterns": ["Thanks", "Thank you", "That's helpful"],
            "responses": ["You're welcome!", "Anytime!", "Glad I could help."]
        }
    ]
}

# Initialize NLP tools
lemmatizer = WordNetLemmatizer()

# Prepare training data
words = []
classes = []
documents = []
ignore_words = ["?", "!"]

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        word_list = nltk.word_tokenize(pattern)  # This will work now
        words.extend(word_list)
        documents.append((word_list, intent["tag"]))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

# Lemmatize and sort words
words = sorted(set([lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]))
classes = sorted(set(classes))

# Encode training data
X_train = []
y_train = []
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(classes)

for pattern, tag in documents:
    bag = [1 if w in [lemmatizer.lemmatize(word.lower()) for word in pattern] else 0 for w in words]
    X_train.append(bag)
    y_train.append(labels_encoded[classes.index(tag)])

X_train = np.array(X_train)
y_train = np.array(y_train)

# Build Neural Network Model
model = Sequential([
    Dense(128, activation="relu", input_shape=(len(X_train[0]),)),
    Dropout(0.5),
    Dense(64, activation="relu"),
    Dropout(0.5),
    Dense(len(classes), activation="softmax")
])

# Compile and train the model
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=200, batch_size=8, verbose=0)

# Define chatbot functions
def clean_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w.lower()) for w in tokens]
    return tokens

def bag_of_words(sentence):
    sentence_words = clean_text(sentence)
    bag = [1 if w in sentence_words else 0 for w in words]
    return np.array(bag)

def predict_intent(sentence):
    bow = bag_of_words(sentence)
    prediction = model.predict(np.array([bow]))[0]
    predicted_class_index = np.argmax(prediction)
    return classes[predicted_class_index]

def get_response(intent):
    for i in intents["intents"]:
        if i["tag"] == intent:
            return random.choice(i["responses"])

# Streamlit UI
st.title("🤖 AI Chatbot")
st.write("Ask me anything!")

user_input = st.text_input("You: ", "")

if user_input:
    intent = predict_intent(user_input)
    response = get_response(intent)
    st.text_area("Chatbot:", value=response, height=100, max_chars=None)

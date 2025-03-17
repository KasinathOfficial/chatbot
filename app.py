import json
import random
import numpy as np
import nltk
# ✅ FIX: Download required NLTK models
nltk.download('punkt')  
nltk.download('wordnet')  
nltk.download('punkt_tab')
import streamlit as st
import tensorflow as tf
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Load chatbot intents
with open("train.json") as file:
    intents = json.load(file)

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
#label_encoder = LabelEncoder()
#y_train = to_categorical(y_train, num_classes=len(classes))


label_encoder = LabelEncoder()
label_encoder.fit(classes)
labels_encoded = label_encoder.fit_transform(classes)
#y_train.append(label_encoder.transform([tag])[0])  # Fix indexing issue


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
model.fit(X_train, y_train, epochs=50, batch_size=4, verbose=1)

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
st.title("🤖SVCE AI Chatbot")
st.write("Ask me anything!")

user_input = st.text_input("You: ", "")

if user_input:
    intent = predict_intent(user_input)
    response = get_response(intent)
    st.text_area("Chatbot:", value=response, height=100, max_chars=None)

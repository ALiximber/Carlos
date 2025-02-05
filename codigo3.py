import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import spacy
from database import collection

nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc if not token.is_stop and token.is_alpha])

def load_training_data():
    data = list(collection.find({}, {"_id": 0}))
    texts = [preprocess_text(doc["text"]) for doc in data]
    labels = np.random.randint(0, 3, len(texts))  # Etiquetas aleatorias por ahora
    return texts, labels

texts, labels = load_training_data()

tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
max_length = 100
X = pad_sequences(sequences, maxlen=max_length, padding="post")
y = tf.keras.utils.to_categorical(labels, num_classes=3)

model = Sequential([
    Embedding(input_dim=10000, output_dim=256, input_length=max_length),
    Dense(1000, activation="relu"),
    Dense(1000, activation="relu"),
    Dense(1000, activation="relu"),
    Dense(1000, activation="relu"),
    Dense(1000, activation="relu"),
    Dense(1000, activation="relu"),
    Dense(1000, activation="relu"),
    Dense(1000, activation="relu"),
    Dense(1000, activation="relu"),
    Dense(3, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(X, y, epochs=50, batch_size=32, verbose=1)

model.save("ai_talent_matcher.h5")
print("Modelo entrenado y guardado ")

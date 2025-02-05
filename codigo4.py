from fastapi import FastAPI
import tensorflow as tf
import numpy as np
from modelo_ia import preprocess_text, tokenizer, pad_sequences, max_length

app = FastAPI()
model = tf.keras.models.load_model("ai_talent_matcher.h5")

@app.post("/predict")
def predict(data: dict):
    text = preprocess_text(data["cv"])
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding="post")
    
    prediction = model.predict(padded_sequence)
    category = np.argmax(prediction)
    
    categories = {0: "TI", 1: "Marketing", 2: "Ingeniería"}
    return {"categoria_predicha": categories[category]}


from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["recruitment_db"]
collection = db["candidates"]

def save_candidate(profile_text, category="Desconocido"):
    document = {"text": profile_text, "category": category}
    collection.insert_one(document)

from extractor import cv_texts

for text in cv_texts:
    save_candidate(text)

print("CVs guardados en la base de datos")

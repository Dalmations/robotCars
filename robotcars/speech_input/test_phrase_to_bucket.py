import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from shapes import SHAPES

EMBEDDING_FILE = "robotcars/speech_input/shape_embeddings.pkl"
THRESHOLD = 0.6

model = SentenceTransformer("all-MiniLM-L6-v2")

def embed(texts):
    return model.encode(texts, normalize_embeddings=True)

def load_or_build_phrase_embeddings():
    if os.path.exists(EMBEDDING_FILE):
        with open(EMBEDDING_FILE, "rb") as f:
            phrase_to_vec = pickle.load(f)
    else:
        phrase_to_vec = {}

    updated = False

    for shape, phrases in SHAPES.items():
        for phrase in phrases:
            if phrase not in phrase_to_vec:
                phrase_to_vec[phrase] = embed([phrase])[0]
                updated = True

    if updated:
        with open(EMBEDDING_FILE, "wb") as f:
            pickle.dump(phrase_to_vec, f)

    return phrase_to_vec

def build_centroids(phrase_to_vec):
    shape_centroids = {}

    for shape, phrases in SHAPES.items():
        vectors = np.vstack([phrase_to_vec[p] for p in phrases])
        centroid = np.mean(vectors, axis=0)

        centroid = centroid / np.linalg.norm(centroid)

        shape_centroids[shape] = centroid

    return shape_centroids

phrase_to_vec = load_or_build_phrase_embeddings()
shape_centroids = build_centroids(phrase_to_vec)

CENTROID_MATRIX = np.vstack(list(shape_centroids.values()))
CENTROID_LABELS = np.array(list(shape_centroids.keys()))

def classify(phrase):
    phrase_vec = embed([phrase])[0]

    scores = CENTROID_MATRIX @ phrase_vec
    best_idx = np.argmax(scores)
    best_score = scores[best_idx]
    print(best_score)

    if best_score >= THRESHOLD:
        return CENTROID_LABELS[best_idx]
    return "No match"

while True:
    phrase = input('input phrase\n')
    print(classify(phrase),'\n')
import numpy as np
from sentence_transformers import SentenceTransformer

SHAPES = {
    "circle": [
        "a round shape",
        "a shape with no corners",
        "perfectly circular object",
        "all points equally distant from the center",
        "no edges or vertices",
        'circle'
    ],
    "triangle": [
        "a shape with three sides",
        "three cornered polygon",
        "polygon with three angles",
        "three straight edges meeting at vertices",
        "a three sided figure",
        'triangle'
    ],
    "rectangle": [
        "a shape with four right angles",
        "a box-like shape with equal opposite sides",
        "four sided shape with ninety degree corners",
        "opposite sides are parallel and equal",
        "a rectangular figure",
        'rectangle'
    ],
    "parallelogram": [
        "a slanted rectangle",
        "four sided shape with opposite sides parallel",
        "a quadrilateral with parallel opposite edges",
        "a skewed four sided polygon",
        "a sloping four sided shape",
        'parallelogram'
    ],
    "pentagon": [
        "a shape with five sides",
        "five sided polygon",
        "a polygon with five angles",
        "five straight edges forming a closed shape",
        "a five cornered figure",
        'pentagon'
    ]
}
THRESHOLD = 0.60

model = SentenceTransformer("all-MiniLM-L6-v2")

def embed(texts):
    return model.encode(
        texts,
        normalize_embeddings=True
    )

shape_matrices = {}
shape_labels = []

all_vectors = []
all_shapes = []

for shape, phrases in SHAPES.items():
    vecs = embed(phrases)
    shape_matrices[shape] = vecs
    all_vectors.append(vecs)
    all_shapes.extend([shape] * vecs.shape[0])

ALL_VECTORS = np.vstack(all_vectors)
ALL_SHAPES = np.array(all_shapes)

def classify(phrase):
    phrase_vec = embed(phrase) 
    phrase_vec = phrase_vec.reshape(-1) 

    scores = ALL_VECTORS @ phrase_vec

    best_idx = np.argmax(scores)
    best_score = scores[best_idx]
    best_shape = ALL_SHAPES[best_idx]

    return best_shape if best_score >= THRESHOLD else 'No match'

# while True:
#     phrase = input('input phrase\n')
#     print(classify(phrase),'\n')
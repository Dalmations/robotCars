import re
from shapes import SHAPES

pattern = re.compile(r"\b(" + "|".join(SHAPES.keys()) + r")s?\b", re.IGNORECASE)

def classify(phrase):
    match = pattern.search(phrase)
    return match.group(1).lower() if match else "No match"

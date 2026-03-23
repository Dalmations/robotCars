import re
from speech_input.classification import classify, classify_MVP

def get_robot_names(phrase):
    robot_pattern = re.compile(r"\b(" + "|".join(['strawberry', 'blueberry', 'raspberry']) + r")s?\b", re.IGNORECASE)
    return robot_pattern.findall(phrase)

def handle_input(phrase):
    robots = get_robot_names(phrase)
    for robot in robots:
        phrase = phrase.replace(robot, '')
    shape = classify(phrase)
    return shape, robots
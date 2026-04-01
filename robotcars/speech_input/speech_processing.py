from robotcars.car_tools.motor_controller import MotorController, MotorConfig
from robotcars.car_tools.picarx_path_follower import PurePursuitFollower, FollowerConfig
from robotcars.car_tools.movement import MovementPlanner
from robotcars.car_tools.shape_path import drive_in_shape

try:
    from picarx.stt import Vosk
except ImportError:
    Vosk = None
    print("Vosk speech recognition not available.")

if Vosk:
    vosk = Vosk(language="en-us")
    print(vosk.available_languages)
else:
    vosk = None

def processCommand(result):
    if result and "circle" in result:
        return "circle"
    if result and "square" in result:
        return "square"
    return None

def run_circle_demo():
    motor = MotorController(MotorConfig(speed=35))
    planner = MovementPlanner(world_size=(25, 25))
    circle_path = planner.plan_formation('circle')
    follower = PurePursuitFollower(motor, FollowerConfig())
    print("Running circle demo...")
    follower.follow(circle_path)
    print("Circle demo complete.")

if vosk:
    while True:
        print("Listening for commands")
        result = vosk.listen(stream=False)
        command = processCommand(result)
        if command == "circle":
            run_circle_demo()
        elif command == "triangle":
            drive_in_shape(3)
        elif command == "square":
            drive_in_shape(4)
        elif command == "pentagon":
            drive_in_shape(5)
        print(result)
else:
    print("Speech recognition is not available.")

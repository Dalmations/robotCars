from robotcars.car_tools.motor_controller import MotorController, MotorConfig
import time

def drive_in_shape(sides):
	if sides < 3:
		print("A shape must have at least 3 sides")
		return

	motor = MotorController(MotorConfig(speed=50))

	for i in range(sides):
		motor.set_steering(0)
		motor.forward_for(1)
		motor.set_steering(-90)
		motor.px.set_motor_speed(2, -20)
		time.sleep(1.7)
		motor.stop()
		motor.set_steering(0)

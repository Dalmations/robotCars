from car_tools.motor_controller import MotorController, MotorConfig
import time

def drive_in_shape(sides):
	motor = MotorController(MotorConfig(speed=50))
	calibrate_sec = 1.6
	# circle
	if sides == 0:
		motor.set_steering(30)
		motor.forward_for(10)
		motor.stop()
		motor.set_steering(0)
		return
	if sides < 3:
		print("A shape must have at least 3 sides")
		return

	t = (1/sides)*6.8

	for i in range(sides):
		motor.set_steering(0)
		motor.forward_for(4)
		motor.set_steering(-90)
		motor.backward_for(calibrate_sec)
		# motor.px.set_motor_speed(2, -20)
		# time.sleep(t)
		motor.stop()
		motor.set_steering(0)

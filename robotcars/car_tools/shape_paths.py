import time
from car_tools.params import PARAMS
from car_tools.motor_controller import MotorController
import os

def drive_in_shape(shape, motor:MotorController):
	params = PARAMS[os.uname().nodename][shape]
	motor.cfg.speed = PARAMS[os.uname().nodename][shape]['speed']
	match shape:
		case 'circle':
			motor.set_steering(params['steering'])
			motor.forward_for(params['forward_for'])
			motor.stop()
			motor.set_steering(0)
		case 'square':
			for i in range(4):
				motor.set_steering(0)
				motor.forward_for(params['forward_for'])
				motor.stop()
				time.sleep(0.25)
				motor.set_steering(-90)
				motor.backward_for(params['backward_for'])
				motor.stop()
				motor.set_steering(0)
		case 'hexagon':
			for i in range(6):
				motor.set_steering(0)
				motor.forward_for(params['forward_for'])
				motor.stop()
				time.sleep(0.05)
				motor.px.set_motor_speed(1, 80)
				motor.px.set_motor_speed(2, 80)
				time.sleep(params['sleep'])
				motor.stop()
				motor.set_steering(0)
		case _:
			return

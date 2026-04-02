import time
from car_tools.params import PARAMS
import os

def drive_in_shape(shape, motor):
	params = PARAMS[os.uname().nodename][shape]
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
				motor.set_steering(-90)
				motor.backward_for(params['backward_for'])
				motor.stop()
				motor.set_steering(0)
		case 'hexagon':
			for i in range(6):
				motor.set_steering(0)
				motor.forward_for(params['forward_for'])
				motor.set_steering(-90)
				motor.px.set_motor_speed(2, params['motor_speed'])
				time.sleep(params['sleep'])
				motor.stop()
				motor.set_steering(0)
		case _:
			return

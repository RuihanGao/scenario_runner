import scipy.misc

import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from srunner.challenge.autoagents.autonomous_agent import AutonomousAgent, Track

try:
		import numpy as np
except ImportError:
		raise RuntimeError('cannot import numpy, make sure numpy package is installed')

# for recording dataset
import csv
# for use nn_controller
from NN_controller import *
from NN_controller import MLP

class NNAgent(AutonomousAgent):
		def setup(self, path_to_conf_file, nn_model_path='models/NN_model_epo50.pth'):
				# similar to __init__
				self.track = Track.ALL_SENSORS_HDMAP_WAYPOINTS
				# RH
				self.nn_model = torch.load(nn_model_path)
				self.nn_model.eval()
				self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

		def sensors(self):
				"""
				Define the sensor suite required by the agent

				:return: a list containing the required sensors in the following format:

				[
						{'type': 'sensor.camera.rgb', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
											'width': 300, 'height': 200, 'fov': 100, 'id': 'Left'},

						{'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
											'width': 300, 'height': 200, 'fov': 100, 'id': 'Right'},

						{'type': 'sensor.lidar.ray_cast', 'x': 0.7, 'y': 0.0, 'z': 1.60, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0,
						 'id': 'LIDAR'}


				"""
				sensors = [{'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 0.0, 'z': 1.60, 'roll':0.0, 'pitch':0.0, 'yaw': 0.0,
										'width': 800, 'height': 600, 'fov':100, 'id': 'Center'},
									 {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0,
										'yaw': -45.0, 'width': 800, 'height': 600, 'fov': 100, 'id': 'Left'},
									 {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 45.0,
										'width': 800, 'height': 600, 'fov': 100, 'id': 'Right'},
									 {'type': 'sensor.lidar.ray_cast', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0,
										'yaw': -45.0, 'id': 'LIDAR'},
									 {'type': 'sensor.other.gnss', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'id': 'GPS'},
									 {'type': 'sensor.can_bus', 'reading_frequency': 25, 'id': 'can_bus'},
									 {'type': 'sensor.hd_map', 'reading_frequency': 1, 'id': 'hdmap'},
									]

				return sensors

		def run_step(self, input_data, timestamp):
				# print input_data as a dictionary
				# print("=====================>")
				# for key, val in input_data.items():
				# 		if hasattr(val[1], 'shape'):
				# 				shape = val[1].shape
				# 				print("[{} -- {:06d}] with shape {}".format(key, val[0], shape))
				# 		else:
				# 				print("[{} -- {:06d}] ".format(key, val[0]))
				# print("<=====================")

				# # an example of the input
				# [LIDAR -- 006493] with shape (21843, 3)
				# [Center -- 006494] with shape (600, 800, 4)
				# [hdmap -- 000010] 
				# [GPS -- 006494] with shape (3,)
				# [Right -- 006494] with shape (600, 800, 4)
				# [Left -- 006494] with shape (600, 800, 4)
				# [can_bus -- 000196] 

				# 1. parse the input_data
				# get localization data from GPS sensor
				print(input_data['GPS'][1]) # e.g. (23093, array([48.99839538,  7.9999472 ,  1.8426466 ]))
				location = carla.Location(x=input_data['GPS'][1][0], y=input_data['GPS'][1][1], z=input_data['GPS'][1][2])
				# get waypoint data from hdmap sensor
				# print("location", location)
				map = CarlaDataProvider.get_map()
				wp = map.get_waypoint(location) # seem to provide only one wp ahead 
				if wp is None:
						raise ValueError("No waypoint can be obtained from the current location")
						# set/global planner
				control = self.get_nn_control(location, wp)
				print("control in run_step", control)
				return control


		def get_nn_control(self, location, waypoint):
				tf = waypoint.transform
				states = np.hstack((location.x, location.y, location.z, waypoint.lane_width, tf.location.x, tf.location.y, tf.location.z, \
						tf.rotation.pitch, tf.rotation.yaw, tf.rotation.roll))				
				# expand vector to 2D array
				states = np.expand_dims(states, axis=0)
				states = torch.from_numpy(states).double() #.to(world.device)
				print("states", states.size(), states)
				nn_control = self.nn_model(states)
				# print("nn_control", nn_control)
				nn_control = nn_control.data.cpu().numpy()[0]
				print("control from nn_controller", nn_control)
				# make a whole set of control
				control = carla.VehicleControl()
				control.steer = nn_control[1]
				control.throttle = nn_control[0]
				control.brake = 0.0
				control.hand_brake = False
				return control

if __name__ == '__main__':
	conf = '${ROOT_SCENARIO_RUNNER}/srunner/challenge/autoagents/HumanAgent.py'
	agent = NNAgent(conf)
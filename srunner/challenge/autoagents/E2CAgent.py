import os, sys
import glob
import numpy as np
import scipy
import scipy.misc

import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from srunner.challenge.autoagents.autonomous_agent import AutonomousAgent, Track

from configs import g_conf
from logger import coil_logger
from e2c_controller import E2C, CarlaData
# TODO create a file to parse conf file
import checkpoint_parse_configuration_file

class E2CAgent(AutonomousAgent):

	def setup(self, path_to_conf_file, dim_in = 200*88*3, dim_z = 100, dim_u = 2, checkpoint_number=None):
		# TODO: adjust dim_z, dim_u = 2 or 3 (brake?)
		self.img_height = 88
		self.img_width = 200
		dim_in = self.image_width * self.img_height * 3
		self.track = Track.ALL_SENSORS_HDMAP_WAYPOINTS
		# config, checkpoint_number = checkpoint_parse_conf_file(path_to_conf_file)
		
		# TODO: check the ckp path
		checkpoint = torch.load(os.path.join('/', os.path.join(*os.path.realpath(__file__).split('/')[:-2]),
									  '_logs',
									 yaml_conf.split('/')[-2], yaml_conf.split('/')[-1].split('.')[-2]
									 , 'checkpoints', str(checkpoint_number) + '.pth'))
		self.checkpoint = checkpoint
		self._model = E2C(dim_in, dim_z, dim_u, config)
		self._model.load_state_dict(checkpoint['state_dict'])
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		self.nn_model.eval()


	def sensor(self):
		sensors = [{'type': 'sensor.camera.rgb',
					'x': 2.0, 'y': 0.0, 'z': 1.40, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
					'width': 800, 'height': 600, 'fov': 100, 'id': 'rgb'},
				   {'type': 'sensor.can_bus', 'reading_frequency': 25, 'id': 'can_bus' },
				   {'type': 'sensor.other.gnss', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'id': 'GPS'},
				   {'type': 'sensor.hd_map', 'reading_frequency': 1, 'id': 'hdmap'}
				   ]
		# 	 TODO	For three cameras
		# {'type': 'sensor.camera.rgb', 
		# 'x': 0.7, 'y': 0.0, 'z': 1.60, 'roll':0.0, 'pitch':0.0, 'yaw': 0.0,
		# 'width': 800, 'height': 600, 'fov':100, 'id': 'Center'},
		# 		 {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0,
		# 			'yaw': -45.0, 'width': 800, 'height': 600, 'fov': 100, 'id': 'Left'},
		# 		 {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 45.0,
		# 			'width': 800, 'height': 600, 'fov': 100, 'id': 'Right'},

	def run_step(self, input_data, timestamp):
		# see  CoILBaseline for parsing camera data
		# obtain speed from can_bus sensor (may not need speed info for e2c)
		# # obtain location from GPS sensor		
		# location = carla.Location(x=input_data['GPS'][1][0], y=input_data['GPS'][1][1], z=input_data['GPS'][1][2])
		# # for e2c, wp is not needed
		# # get waypoint data from hdmap sensor
		# map = CarlaDataProvider.get_map()
		# wp = map.get_waypoint(location) # seem to provide only one wp ahead 
		# if wp is None:
		# 		raise ValueError("No waypoint can be obtained from the current location")
		# 		# set/global planner

		# obtain images from camera sensor

		# process image from real-time sensor, with id of 'rgb'
		img = _process_img(input_data['rgb'][1])
		latent_z, control = self.get_e2c_control(img)
		print("control in run_step", control)
		return control

	def _process_img(self, sensor):
		sensor = sensor[:, :, 0:3]  # BGRA->BRG drop alpha channel
		sensor = sensor[:, :, ::-1]  # BGR->RGB
		# sensor = sensor[g_conf.IMAGE_CUT[0]:g_conf.IMAGE_CUT[1], :, :]  # crop
		sensor = scipy.misc.imresize(sensor, (self.img_height, self.img_width))
		self.latest_image = sensor

		sensor = np.swapaxes(sensor, 0, 1)
		sensor = np.transpose(sensor, (2, 1, 0))
		sensor = torch.from_numpy(sensor / 255.0).type(torch.FloatTensor).cuda()
		image_input = sensor.unsqueeze(0)
		self.latest_image_tensor = image_input

		return image_input


	def get_e2c_control(self, img)


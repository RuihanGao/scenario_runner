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
		self._model.eval()


	def sensor(self):
		sensors = [{'type': 'sensor.camera.rgb',
					'x': 2.0, 'y': 0.0, 'z': 1.40, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
					'width': 800, 'height': 600, 'fov': 100, 'id': 'rgb'},
				   {'type': 'sensor.can_bus', 'reading_frequency': 25, 'id': 'can_bus' },
				   {'type': 'sensor.other.gnss', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'id': 'GPS'},
				   {'type': 'sensor.hd_map', 'reading_frequency': 1, 'id': 'hdmap'}
				   ]

	def run_step(self, input_data, timestamp):
		# process image from real-time camera sensor, with id of 'rgb'
		img = _process_img(input_data['rgb'][1])
		latent_z, control = self.get_e2c_control(img)
		print("control in run_step", control)
		return control

	def _process_img(self, sensor):
		# copy from CoILBaseline
		# TODO: check with _process_img in e2c_controller
		sensor = sensor[:, :, 0:3]  # BGRA->BRG drop alpha channel
		sensor = sensor[:, :, ::-1]  # BGR->RGB
		# sensor = sensor[g_conf.IMAGE_CUT[0]:g_conf.IMAGE_CUT[1], :, :]  # crop
		sensor = scipy.misc.imresize(sensor, (self.img_height, self.img_width))
		self.latest_image = sensor
		print("latest_image", self.latest_image.shape)

		sensor = np.swapaxes(sensor, 0, 1)
		sensor = np.transpose(sensor, (2, 1, 0))
		sensor = torch.from_numpy(sensor / 255.0).type(torch.FloatTensor).cuda()
		img = sensor.unsqueeze(0)
		self.latest_image_tensor = img
		print("img", img.size())
		return img


	def get_e2c_control(self, img):
		# TODO check how to extract latent z
		e2c_control = self._model(img)
		e2c_control = e2c_control.data.cpu().numpy()[0]
		print("control from e2c controller", e2c_control)
		# make a whole set of control
		control = carla.VehicleControl()
		control.steer = e2c_control[1]
		control.throttle = e2c_control[0]
		# control.brake = 0.0
		control.brake = e2c_control[2]
		control.hand_brake = False
		return control


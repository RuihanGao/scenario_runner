#!/usr/bin/env python

# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Allows controlling a vehicle with a keyboard.

"""
Welcome to CARLA manual control.

Use ARROWS or WASD keys for control.

	W            : throttle
	S            : brake
	AD           : steer
	Q            : toggle reverse
	Space        : hand-brake
	P            : toggle autopilot
	# RH
	N            : toggle NN controller

	TAB          : change sensor position
	`            : next sensor
	[1-9]        : change to sensor [1-9]
	C            : change weather (Shift+C reverse)
	Backspace    : change vehicle

	R            : toggle recording images to disk

	F1           : toggle HUD
	H/?          : toggle help
	ESC          : quit
"""

from __future__ import print_function

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================


import carla

from carla import ColorConverter as cc

import argparse
import collections
import datetime
import logging
import math
import re
import time
import weakref
import random

try:
	import pygame
	from pygame.locals import KMOD_CTRL
	from pygame.locals import KMOD_SHIFT
	from pygame.locals import K_0
	from pygame.locals import K_9
	from pygame.locals import K_BACKQUOTE
	from pygame.locals import K_BACKSPACE
	from pygame.locals import K_DOWN
	from pygame.locals import K_ESCAPE
	from pygame.locals import K_F1
	from pygame.locals import K_LEFT
	from pygame.locals import K_RIGHT
	from pygame.locals import K_SLASH
	from pygame.locals import K_SPACE
	from pygame.locals import K_TAB
	from pygame.locals import K_UP
	from pygame.locals import K_a
	from pygame.locals import K_c
	from pygame.locals import K_d
	from pygame.locals import K_h
	from pygame.locals import K_p
	from pygame.locals import K_q
	from pygame.locals import K_r
	from pygame.locals import K_s
	from pygame.locals import K_w
	from pygame.locals import K_n
	from pygame.locals import K_e
except ImportError:
	raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
	import numpy as np
except ImportError:
	raise RuntimeError('cannot import numpy, make sure numpy package is installed')

# for recording dataset
import csv
# for use nn_controller
from NN_controller import *


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor

from cat_configs import E2C_cat
from e2c_NN import MLP_e2c, FC_coil



# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================


def find_weather_presets(): # taken care of in challenge_evaluator_
	rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
	name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
	presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
	return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
	name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
	return (name[:truncate-1] + u'\u2026') if len(name) > truncate else name


class World(object):
	# 'models/NN_model_relative_epo50.pth'
	def __init__(self, carla_world, hud, \
				 nn_model_path= 'models/MLP/MLP_model_long_states_50.pth', \
				 e2c_model_path='models/E2C/E2C_model_ctv_logdepth_norm_5.pth'):
		# 'models/MLP/MLP_model_ctv_logdepth_norm_catwp_50_5_WSE_Adam_monly_1000.pth', \
		self.world = carla_world
		self.mapname = carla_world.get_map().name
		self.hud = hud
		self.world.on_tick(hud.on_world_tick)
		self.world.wait_for_tick(10.0)
		self.vehicle = None
		while self.vehicle is None:
			print("Scenario not yet ready")
			time.sleep(1)
			possible_vehicles = self.world.get_actors().filter('vehicle.*')
			for vehicle in possible_vehicles:
				if vehicle.attributes['role_name'] == "hero":
					self.vehicle = vehicle
		self.vehicle_name = self.vehicle.type_id
		self.collision_sensor = CollisionSensor(self.vehicle, self.hud)
		self.lane_invasion_sensor = LaneInvasionSensor(self.vehicle, self.hud)
		self.camera_manager = CameraManager(self.vehicle, self.hud)
		self.camera_manager.set_sensor(3, notify=False)
		self.controller = None
		self._weather_presets = find_weather_presets()
		self._weather_index = 0
		# RH
		self.nn_model = torch.load(nn_model_path)
		self.nn_model.eval()
		print("world.nn_model", nn_model_path)
		print(self.nn_model)
		self.e2c_model = torch.load(e2c_model_path)
		self.e2c_model.eval()
		# print("world.e2c_model")
		# print(self.e2c_model)
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		# print("init None tensor")
		self.m_tensor = None
		self.img_tensor = None

	def restart(self):
		cam_index = self.camera_manager._index
		cam_pos_index = self.camera_manager._transform_index
		start_pose = self.vehicle.get_transform()
		start_pose.location.z += 2.0
		start_pose.rotation.roll = 0.0
		start_pose.rotation.pitch = 0.0
		blueprint = self._get_random_blueprint()
		self.destroy()
		self.vehicle = self.world.spawn_actor(blueprint, start_pose)
		self.collision_sensor = CollisionSensor(self.vehicle, self.hud)
		self.lane_invasion_sensor = LaneInvasionSensor(self.vehicle, self.hud)
		self.camera_manager = CameraManager(self.vehicle, self.hud)
		self.camera_manager._transform_index = cam_pos_index
		self.camera_manager.set_sensor(cam_index, notify=False)
		actor_type = get_actor_display_name(self.vehicle)
		self.hud.notification(actor_type)

	def next_weather(self, reverse=False):
		self._weather_index += -1 if reverse else 1
		self._weather_index %= len(self._weather_presets)
		preset = self._weather_presets[self._weather_index]
		self.hud.notification('Weather: %s' % preset[1])
		self.vehicle.get_world().set_weather(preset[0])

	def tick(self, clock):
		if len(self.world.get_actors().filter(self.vehicle_name)) < 1:
			print("Scenario ended -- Terminating")
			return False

		self.hud.tick(self, self.mapname, clock)
		return True

	def render(self, display):
		self.camera_manager.render(display)
		self.hud.render(display)

	def destroy(self):
		actors = [
			self.camera_manager.sensor,
			self.collision_sensor.sensor,
			self.lane_invasion_sensor.sensor,
			self.vehicle]
		for actor in actors:
			if actor is not None:
				actor.destroy()


# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================


class KeyboardControl(object):
	def __init__(self, world, start_in_autopilot, start_in_nn_controller=False, start_in_e2c_controller=False):
		self._autopilot_enabled = start_in_autopilot
		self._nn_controller_enabled = start_in_nn_controller
		self._e2c_controller_enabled = start_in_e2c_controller
		self._control = carla.VehicleControl()
		self._steer_cache = 0.0
		world.vehicle.set_autopilot(self._autopilot_enabled)
		world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

	def parse_events(self, world, clock):
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				return True
			elif event.type == pygame.KEYUP:
				if self._is_quit_shortcut(event.key):
					return True
				elif event.key == K_BACKSPACE:
					world.restart()
				elif event.key == K_F1:
					world.hud.toggle_info()
				elif event.key == K_h or (event.key == K_SLASH and pygame.key.get_mods() & KMOD_SHIFT):
					world.hud.help.toggle()
				elif event.key == K_TAB:
					world.camera_manager.toggle_camera()
				elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
					world.next_weather(reverse=True)
				elif event.key == K_c:
					world.next_weather()
				elif event.key == K_BACKQUOTE:
					world.camera_manager.next_sensor()
				elif event.key > K_0 and event.key <= K_9:
					world.camera_manager.set_sensor(event.key - 1 - K_0)
				elif event.key == K_r:
					world.camera_manager.toggle_recording()
				elif event.key == K_q:
					self._control.reverse = not self._control.reverse
				elif event.key == K_p:
					self._autopilot_enabled = not self._autopilot_enabled
					world.vehicle.set_autopilot(self._autopilot_enabled)
					world.hud.notification('Autopilot %s' % ('On' if self._autopilot_enabled else 'Off'))
				elif event.key == K_n:
					# toggle NN controller
					self._nn_controller_enabled = not self._nn_controller_enabled
				elif event.key == K_e:
					world.camera_manager.toggle_return()
					self._e2c_controller_enabled = world.camera_manager._return

		if not self._autopilot_enabled:
			if self._nn_controller_enabled:
				# self.get_nn_controller(world)
				# self._parse_keys_other(pygame.key.get_pressed(), clock.get_time())
				self.get_nn_controller_wp(world)
			elif self._e2c_controller_enabled:
				t, s, b = self.get_e2c_controller(world)
				# print("t, s, b")
				# print(type(t), t, type(s), s, type(b), b)
				self._control.hand_brake = False
				# t = 1 if t>0.5 else 0
				self._control.throttle = float(t) #+0.00000000001 # float(t) # u[0].item()
				self._control.steer = float(s) # float(s) # u[1].item()*2-1 # remap from [0,1] to [-1, 1]
				self._control.brake = 0 # float(b) 
			else:
				self._parse_keys(pygame.key.get_pressed(), clock.get_time())
			
			
			# print("before constant control", self._control)


			# print("{} apply control".format(world.hud.frame_number), self._control)
			world.vehicle.apply_control(self._control)
			# location = world.vehicle.get_transform().location
			# print("{} location".format(world.hud.frame_number), world.vehicle.get_transform())
		# else:
			# print autopilot control
			# print("{} apply control".format(world.hud.frame_number), world.vehicle.get_control())
				
		# record_dataset(world)

	def _parse_keys(self, keys, milliseconds):
		self._control.throttle = 1.0 if keys[K_UP] or keys[K_w] else 0.0
		steer_increment = 5e-4 * milliseconds
		if keys[K_LEFT] or keys[K_a]:
			self._steer_cache -= steer_increment
		elif keys[K_RIGHT] or keys[K_d]:
			self._steer_cache += steer_increment
		else:
			self._steer_cache = 0.0
		self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
		self._control.steer = round(self._steer_cache, 1)
		self._control.brake = 1.0 if keys[K_DOWN] or keys[K_s] else 0.0
		self._control.hand_brake = keys[K_SPACE]

	def _parse_keys_other(self, keys, milliseconds):
		# supplementary to get_nn_controller
		self._control.hand_brake = keys[K_SPACE]

	def get_nn_controller(self, world):
		model = world.nn_model
		location = world.vehicle.get_transform().location
		waypoint = world.world.get_map().get_waypoint(location)
		tf = waypoint.transform
		states = np.hstack((location.x, location.y, location.z, waypoint.lane_width, tf.location.x, tf.location.y, tf.location.z, \
			tf.rotation.pitch, tf.rotation.yaw, tf.rotation.roll))
		# expand vector to 2D array
		states = np.expand_dims(states, axis=0)
		states = torch.from_numpy(states).double() #.to(world.device)
		print("states", states.size(), states)
		control = model(states) #RuntimeError: size mismatch, m1: [10 x 1], m2: [10 x 100] at /pytorch/aten/src/TH/generic/THTensorMath.cpp:961
		control = control.data.cpu().numpy()[0]
		print("control from nn_controller", control)
		self._control.throttle = control[0]
		self._control.steer = control[1]


	def get_nn_controller_wp(self, world, horizon=50, sampling_radius=2.0):
		model = world.nn_model
		cur_loc = world.vehicle.get_transform().location
		map = world.world.get_map()
		# concatenate future_wps within horizon
		cur_wp = map.get_waypoint(cur_loc)
		future_wps = []
		future_wps.append(cur_wp)

		for j in range(horizon):
			future_wps.append(random.choice(future_wps[-1].next(sampling_radius)))

		future_wps_np = []
		for future_wp in future_wps:
			future_wps_np.append(np.array([future_wp.transform.location.x, future_wp.transform.location.y]))
		future_wps_np = np.array(future_wps_np)
		future_wps_np = future_wps_np - np.array([cur_wp.transform.location.x, cur_wp.transform.location.y])

		state = np.hstack((np.array([cur_loc.x, cur_loc.y])- np.array([cur_wp.transform.location.x, cur_wp.transform.location.y]), \
					  future_wps_np.flatten()))
		
		state =  torch.from_numpy(state.astype(np.float32))
		state = state.view(1, -1)

		control = model(state)
		control = control.data.cpu().numpy()[0]
		print("control from nn_controller", control)
		self._control.throttle = control[0].item()
		self._control.steer = control[1].item()
		self._control.brake = 1 if control[2].item()>0.5 else 0


	def get_e2c_controller(self, world):
		# print("get_e2c_controller")
		# print("world.vehicle", world.vehicle)
		# my_world = world.vehicle.get_world()
		my_world = world.camera_manager
		# print("my_world", my_world)
		if my_world.img_tensor is None or my_world.m_tensor is None:
			# no available data, no-op
			print("no op")
			self._control.throttle = 0
			self._control.steer = 0
			self._control.brake = 0
			return 0, 0, 0
		else:
			# print("load e2c")
			e2c = world.e2c_model
			# z = e2c.latent_embeddings(my_world.img_tensor, my_world.m_tensor)
			z = my_world.m_tensor

			# print("z", z.size()) # torch.Size([1, 106]) 
			# print(z)
			u = world.nn_model(z)
			# convert tensor to numpy array
			u = u.data.cpu().numpy()[0]
			# print("u ", u)

			# print("type u", type(u))
			# print(type(u[0])) # <class 'numpy.float32'>
			# print(type(u[0].item())) # <class 'float'>
			
			# self._control.hand_brake = False
			# self._control.throttle = 0.99 * float(u[0]) # .item()
			# self._control.steer = 0.99 * (-1+2*float(u[1])) # remap from [0,1] to [-1, 1]
			# self._control.brake = 0.99 * float(u[2])# .item() # max(min(u[2].item(), 1.0), 0)

		return 0.99 * float(u[0]), 0.99 * (-1+2*float(u[1])), 0.99 * float(u[2])
	
	@staticmethod
	def _is_quit_shortcut(key):
		return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


# ==============================================================================
# -- Try to record the localization data and control output of autopilot -------
# ==============================================================================
def record_dataset(world):
	map = world.world.get_map()
	transform = world.vehicle.get_transform()
	location = transform.location
	waypoint = map.get_waypoint(transform.location)
	# print("location, waypoint")
	# print(location, waypoint)
	# print(location.x)
	# print(waypoint.transform.location, waypoint.transform.rotation)
	# print("lane_width", waypoint.lane_width) # lane_width = 3.5 constant
	control = world.vehicle.get_control()
	write_in_csv(location, waypoint.transform, waypoint.lane_width, control)

def write_in_csv(location, waypoint_tf, lane_width, control, ds='localization_ds.csv'):
	# example of waypoint_tf Location(x=394.307587, y=-294.747772, z=0.000000) Rotation(pitch=360.000000, yaw=246.589417, roll=0.000000)
	#   location.z, rotation.pitch, rotation.roll may be helpful with ControlLoss scenario, where the chasis changes
	# example of vehicle control: VehicleControl(throttle=1.000000, steer=-0.001398, brake=0.000000, hand_brake=False, reverse=False, manual_gear_shift=False, gear=3)
	# TODO: check whether all control params are needed 
	row = [location.x, location.y, location.z, lane_width, waypoint_tf.location.x, waypoint_tf.location.y, waypoint_tf.location.z, \
		waypoint_tf.rotation.pitch, waypoint_tf.rotation.yaw, waypoint_tf.rotation.roll, control.throttle, control.steer]

	# append the current data to csv file
	with open(ds, 'a+') as csvFile:
		writer = csv.writer(csvFile)
		writer.writerow(row)
		csvFile.close()

# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================

class HUD(object):
	def __init__(self, width, height):
		self.dim = (width, height)
		font = pygame.font.Font(pygame.font.get_default_font(), 20)
		fonts = [x for x in pygame.font.get_fonts() if 'mono' in x]
		default_font = 'ubuntumono'
		mono = default_font if default_font in fonts else fonts[0]
		mono = pygame.font.match_font(mono)
		self._font_mono = pygame.font.Font(mono, 14)
		self._notifications = FadingText(font, (width, 40), (0, height - 40))
		self.help = HelpText(pygame.font.Font(mono, 24), width, height)
		self.server_fps = 0
		self.frame_number = 0
		self.simulation_time = 0
		self._show_info = True
		self._info_text = []
		self._server_clock = pygame.time.Clock()

	def on_world_tick(self, timestamp):
		self._server_clock.tick()
		self.server_fps = self._server_clock.get_fps()
		self.frame_number = timestamp.frame
		self.simulation_time = timestamp.elapsed_seconds

	def tick(self, world, mapname, clock):
		if not self._show_info:
			return
		t = world.vehicle.get_transform()
		v = world.vehicle.get_velocity()
		c = world.vehicle.get_control()
		heading = 'N' if abs(t.rotation.yaw) < 89.5 else ''
		heading += 'S' if abs(t.rotation.yaw) > 90.5 else ''
		heading += 'E' if 179.5 > t.rotation.yaw > 0.5 else ''
		heading += 'W' if -0.5 > t.rotation.yaw > -179.5 else ''
		colhist = world.collision_sensor.get_collision_history()
		collision = [colhist[x + self.frame_number - 200] for x in range(0, 200)]
		max_col = max(1.0, max(collision))
		collision = [x / max_col for x in collision]
		vehicles = world.world.get_actors().filter('vehicle.*')
		self._info_text = [
			'Server:  % 16d FPS' % self.server_fps,
			'Client:  % 16d FPS' % clock.get_fps(),
			'',
			'Vehicle: % 20s' % get_actor_display_name(world.vehicle, truncate=20),
			'Map:     % 20s' % mapname,
			'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
			'',
			'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)),
			u'Heading:% 16.0f\N{DEGREE SIGN} % 2s' % (t.rotation.yaw, heading),
			'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (t.location.x, t.location.y)),
			'Height:  % 18.0f m' % t.location.z,
			'',
			('Throttle:', c.throttle, 0.0, 1.0),
			('Steer:', c.steer, -1.0, 1.0),
			('Brake:', c.brake, 0.0, 1.0),
			('Reverse:', c.reverse),
			('Hand brake:', c.hand_brake),
			'',
			'Collision:',
			collision,
			'',
			'Number of vehicles: % 8d' % len(vehicles)
		]
		if len(vehicles) > 1:
			self._info_text += ['Nearby vehicles:']
			distance = lambda l: math.sqrt((l.x - t.location.x)**2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2)
			vehicles = [(distance(x.get_location()), x) for x in vehicles if x.id != world.vehicle.id]
			for d, vehicle in sorted(vehicles):
				if d > 200.0:
					break
				vehicle_type = get_actor_display_name(vehicle, truncate=22)
				self._info_text.append('% 4dm %s' % (d, vehicle_type))
		self._notifications.tick(world, clock)

	def toggle_info(self):
		self._show_info = not self._show_info

	def notification(self, text, seconds=2.0):
		self._notifications.set_text(text, seconds=seconds)

	def error(self, text):
		self._notifications.set_text('Error: %s' % text, (255, 0, 0))

	def render(self, display):
		if self._show_info:
			info_surface = pygame.Surface((220, self.dim[1]))
			info_surface.set_alpha(100)
			display.blit(info_surface, (0, 0))
			v_offset = 4
			bar_h_offset = 100
			bar_width = 106
			for item in self._info_text:
				if v_offset + 18 > self.dim[1]:
					break
				if isinstance(item, list):
					if len(item) > 1:
						points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
						pygame.draw.lines(display, (255, 136, 0), False, points, 2)
					item = None
					v_offset += 18
				elif isinstance(item, tuple):
					if isinstance(item[1], bool):
						rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
						pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
					else:
						rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
						pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
						f = (item[1] - item[2]) / (item[3] - item[2])
						if item[2] < 0.0:
							rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
						else:
							rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
						pygame.draw.rect(display, (255, 255, 255), rect)
					item = item[0]
				if item: # At this point has to be a str.
					surface = self._font_mono.render(item, True, (255, 255, 255))
					display.blit(surface, (8, v_offset))
				v_offset += 18
		self._notifications.render(display)
		self.help.render(display)


# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
	def __init__(self, font, dim, pos):
		self.font = font
		self.dim = dim
		self.pos = pos
		self.seconds_left = 0
		self.surface = pygame.Surface(self.dim)

	def set_text(self, text, color=(255, 255, 255), seconds=2.0):
		text_texture = self.font.render(text, True, color)
		self.surface = pygame.Surface(self.dim)
		self.seconds_left = seconds
		self.surface.fill((0, 0, 0, 0))
		self.surface.blit(text_texture, (10, 11))

	def tick(self, _, clock):
		delta_seconds = 1e-3 * clock.get_time()
		self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
		self.surface.set_alpha(500.0 * self.seconds_left)

	def render(self, display):
		display.blit(self.surface, self.pos)


# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
	def __init__(self, font, width, height):
		lines = __doc__.split('\n')
		self.font = font
		self.dim = (680, len(lines) * 22 + 12)
		self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
		self.seconds_left = 0
		self.surface = pygame.Surface(self.dim)
		self.surface.fill((0, 0, 0, 0))
		for n, line in enumerate(lines):
			text_texture = self.font.render(line, True, (255, 255, 255))
			self.surface.blit(text_texture, (22, n * 22))
			self._render = False
		self.surface.set_alpha(220)

	def toggle(self):
		self._render = not self._render

	def render(self, display):
		if self._render:
			display.blit(self.surface, self.pos)


# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
	def __init__(self, parent_actor, hud):
		self.sensor = None
		self._history = []
		self._parent = parent_actor
		self._hud = hud
		world = self._parent.get_world()
		bp = world.get_blueprint_library().find('sensor.other.collision')
		self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
		# We need to pass the lambda a weak reference to self to avoid circular
		# reference.
		weak_self = weakref.ref(self)
		self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

	def get_collision_history(self):
		history = collections.defaultdict(int)
		for frame, intensity in self._history:
			history[frame] += intensity
		return history

	@staticmethod
	def _on_collision(weak_self, event):
		self = weak_self()
		if not self:
			return
		actor_type = get_actor_display_name(event.other_actor)
		self._hud.notification('Collision with %r' % actor_type)
		impulse = event.normal_impulse
		intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
		self._history.append((event.frame_number, intensity))
		if len(self._history) > 4000:
			self._history.pop(0)


# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================


class LaneInvasionSensor(object):
	def __init__(self, parent_actor, hud):
		self.sensor = None
		self._parent = parent_actor
		self._hud = hud
		world = self._parent.get_world()
		bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
		self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
		# We need to pass the lambda a weak reference to self to avoid circular
		# reference.
		weak_self = weakref.ref(self)
		self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

	@staticmethod
	def _on_invasion(weak_self, event):
		self = weak_self()
		if not self:
			return
		text = ['%r' % str(x).split()[-1] for x in set(event.crossed_lane_markings)]
		self._hud.notification('Crossed line %s' % ' and '.join(text))


# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
	def __init__(self, parent_actor, hud):
		self.sensor = None
		self._surface = None
		self._parent = parent_actor
		self._hud = hud
		self._recording = False
		# RH
		self._return = False # pass image to get latent_embeddings
		self._camera_transforms = [
			carla.Transform(carla.Location(x=1.6, z=1.7)),
			carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15))]
		self._transform_index = 1
		self._sensors = [
			['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
			['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)'],
			['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)'],
			['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)'],
			['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)'],
			['sensor.camera.semantic_segmentation', cc.CityScapesPalette, 'Camera Semantic Segmentation (CityScapes Palette)'],
			['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)']]
		world = self._parent.get_world()
		bp_library = world.get_blueprint_library()
		for item in self._sensors:
			bp = bp_library.find(item[0])
			if item[0].startswith('sensor.camera'):
				bp.set_attribute('image_size_x', str(hud.dim[0]))
				bp.set_attribute('image_size_y', str(hud.dim[1]))
			item.append(bp)
		self._index = None

		self.m_tensor = None
		self.img_tensor = None

	def toggle_return(self):
		self._return = not self._return

	def toggle_camera(self):
		self._transform_index = (self._transform_index + 1) % len(self._camera_transforms)
		self.sensor.set_transform(self._camera_transforms[self._transform_index])

	def set_sensor(self, index, notify=True):
		index = index % len(self._sensors)
		needs_respawn = True if self._index is None \
			else self._sensors[index][0] != self._sensors[self._index][0]
		if needs_respawn:
			if self.sensor is not None:
				self.sensor.destroy()
				self._surface = None
			self.sensor = self._parent.get_world().spawn_actor(
				self._sensors[index][-1],
				self._camera_transforms[self._transform_index],
				attach_to=self._parent)
			# We need to pass the lambda a weak reference to self to avoid
			# circular reference.
			
			# self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
			self.sensor.listen(lambda image: CameraManager._parse_image_and_measurement(self, image))
		if notify:
			self._hud.notification(self._sensors[index][2])
		self._index = index

	def next_sensor(self):
		self.set_sensor(self._index + 1)

	def toggle_recording(self):
		self._recording = not self._recording
		self._hud.notification('Recording %s' % ('On' if self._recording else 'Off'))

	def render(self, display):
		if self._surface is not None:
			display.blit(self._surface, (0, 0))

	@staticmethod
	def _parse_image(weak_self, image, save_dir=None):
		self = weak_self()
		if not self:
			return
		if self._sensors[self._index][0].startswith('sensor.lidar'):
			points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
			points = np.reshape(points, (int(points.shape[0]/3), 3))
			lidar_data = np.array(points[:, :2])
			lidar_data *= min(self._hud.dim) / 100.0
			lidar_data += (0.5 * self._hud.dim[0], 0.5 * self._hud.dim[1])
			lidar_data = np.fabs(lidar_data)
			lidar_data = lidar_data.astype(np.int32)
			lidar_data = np.reshape(lidar_data, (-1, 2))
			lidar_img_size = (self._hud.dim[0], self._hud.dim[1], 3)
			lidar_img = np.zeros(lidar_img_size)
			lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
			self._surface = pygame.surfarray.make_surface(lidar_img)
		else:
			image.convert(self._sensors[self._index][1])
			array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
			array = np.reshape(array, (image.height, image.width, 4))
			array = array[:, :, :3]
			array = array[:, :, ::-1]
			self._surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
		if self._recording:
			# RH: change the path from '_out/%08d' to 'data/%08d', 
			#
			# image.save_to_disk('data_ctv/%08d' % image.frame_number)
			image.save_to_disk(save_dir+'{:08d}'.format(image.frame_number))
		# RH: add return
		if self._return:
			return np.dot(array, [0.2989, 0.5870, 0.1140])

	def _parse_image_and_measurement(self, image, max_loc_diff = 60, max_vel = 90.0):
		if not self._return:
			# print("parse image only")
			weak_self = weakref.ref(self)
			image = self._parse_image(weak_self, image)

		else:
			# use nn and e2c models

			# parse measurement
			# print("parse measurement")
			sampling_radius = 90.0*1/3.6
			transform = self._parent.get_transform()
			velocity = self._parent.get_velocity()
			world = self._parent.get_world()

			# use single waypoint
			# waypoint = random.choice(map.get_waypoint(transform.location).next(sampling_radius))
			# concatenate a series of waypoints
			sub_sampling_radius = 1.0
			num_wps = 50
			w0 = world.get_map().get_waypoint(transform.location) 
			wps = []
			wps.append(w0)
			for i in range(1,num_wps):
				wps.append(wps[i-1].next(sub_sampling_radius)[0])

			# print("concatenate waypoints", len(wps))
			loc_diffs = []
			for wp in wps:
				loc_diffs.append(transform.location - wp.transform.location)

			m = []
			for loc_diff in loc_diffs:
				# wp here is already relative
				m = np.hstack((np.array(m), np.array([norm(loc_diff.x, max_loc_diff), norm(loc_diff.y, max_loc_diff), norm(loc_diff.z, max_loc_diff)])))

			# yaw = math.atan2(velocity.y, velocity.x)
			# speed = np.sqrt(norm(velocity.y, max_vel)**2 + norm(velocity.x, max_vel)**2 )
			# print("yaw {}, speed {}".format(yaw, speed))
			# m = np.hstack((m, np.array([yaw, norm(speed, max_vel)])))

			m = np.hstack((m, np.array([norm(velocity.x, max_vel), norm(velocity.y, max_vel), norm(velocity.z, max_vel)])))

			# m = np.array([norm(loc_diff.x, max_loc_diff), norm(loc_diff.y, max_loc_diff), norm(loc_diff.z, max_loc_diff), \
			# 			  norm(velocity.x, max_vel), norm(velocity.y, max_vel), norm(velocity.z, max_vel)])
			# print("m array", m) # e.g. [0.48041382 0.94163796 0.49981057 0.5        0.5        0.5       ]
			m =  torch.from_numpy(m.astype(np.float32))
			m = m.view(1, -1)

			# print("parse image")
			weak_self = weakref.ref(self)
			image = self._parse_image(weak_self, image)
			# print("return from parse image") # nd.array (88, 200)
			image = torch.from_numpy(image.astype(np.float32))
			image = image.view(1, -1)
			# print("img", image.size())	
			# TODO save image and m for control
			
			# # print("pass m")
			# print("parent of CameraManager", self._parent)
			# print("world in _parse_image_and_measurement", world)
			self.m_tensor = m
			# print("world m", self.m_tensor)
			# print("pass image")
			self.img_tensor = image
			# print("world img", self.img_tensor)

def norm(x, x_max):
	n = x/(x_max*2) + 0.5
	if n>0 and n< 1:
		return n
	else:
		raise ValueError("abnormal norm x {}, x_max {}, norm {}".format(x, x_max, n)) 


# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================


def game_loop(args):
	pygame.init()
	pygame.font.init()
	world = None

	try:
		client = carla.Client(args.host, args.port)
		client.set_timeout(2.0)

		display = pygame.display.set_mode(
			(args.width, args.height),
			pygame.HWSURFACE | pygame.DOUBLEBUF)

		hud = HUD(args.width, args.height)
		world = World(client.get_world(), hud)
		controller = KeyboardControl(world, args.autopilot)

		clock = pygame.time.Clock()
		while True:
			clock.tick_busy_loop(60) # 60
			if controller.parse_events(world, clock):
				return
			if not world.tick(clock):
				return
			world.render(display)
			pygame.display.flip()
			world.world.wait_for_tick()

	finally:

		if world is not None:
			world.destroy()

		pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
	argparser = argparse.ArgumentParser(
		description='CARLA Manual Control Client')
	argparser.add_argument(
		'-v', '--verbose',
		action='store_true',
		dest='debug',
		help='print debug information')
	argparser.add_argument(
		'--host',
		metavar='H',
		default='127.0.0.1',
		help='IP of the host server (default: 127.0.0.1)')
	argparser.add_argument(
		'-p', '--port',
		metavar='P',
		default=2000,
		type=int,
		help='TCP port to listen to (default: 2000)')
	argparser.add_argument(
		'-a', '--autopilot',
		action='store_true',
		help='enable autopilot')
	argparser.add_argument(
		'--res',
		metavar='WIDTHxHEIGHT',
		default='200x88',
		help='window resolution (default: 200x88)') # 1280x720
	args = argparser.parse_args()

	args.width, args.height = [int(x) for x in args.res.split('x')]

	log_level = logging.DEBUG if args.debug else logging.INFO
	logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

	logging.info('listening to server %s:%s', args.host, args.port)

	print(__doc__)

	try:

		game_loop(args)

	except KeyboardInterrupt:
		print('\nCancelled by user. Bye!')
	except Exception as error:
		logging.exception(error)


if __name__ == '__main__':

	main()

#!/usr/bin/env python

#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Object crash without prior vehicle action scenario:
The scenario realizes the user controlled ego vehicle
moving along the road and encountering a cyclist ahead.
"""

import random
import sys

import py_trees
import carla

from ScenarioManager.atomic_scenario_behavior import *
from ScenarioManager.atomic_scenario_criteria import *
from ScenarioManager.scenario_manager import Scenario
from ScenarioManager.timer import TimeOut
from Scenarios.basic_scenario import *


class StationaryObjectCrash(BasicScenario):

    """
    This class holds everything required for a simple object crash
    without prior vehicle action involving a vehicle and a cyclist.
    The ego vehicle is passing through a road and encounters a stationary cyclist.
    """

    timeout = 60

    # ego vehicle parameters
    ego_vehicle_model = 'vehicle.carlamotors.carlacola'
    ego_vehicle_start_x = 100
    ego_vehicle_start = carla.Transform(
        carla.Location(x=ego_vehicle_start_x, y=129, z=1), carla.Rotation(yaw=180))
    ego_vehicle_max_velocity_allowed = 20
    ego_vehicle_distance_to_other = 15

    # other vehicle parameters
    other_vehicle_model = 'vehicle.diamondback.century'
    other_vehicle_start_x = 70
    other_vehicle_start = carla.Transform(
        carla.Location(x=other_vehicle_start_x, y=129, z=0), carla.Rotation(yaw=200))

    def __init__(self, world, debug_mode=False):
        """
        Setup all relevant parameters and create scenario
        and instantiate scenario manager
        """
        self.other_vehicles = [setup_vehicle(world,
                                             self.other_vehicle_model,
                                             self.other_vehicle_start)]
        self.ego_vehicle = setup_vehicle(world,
                                         self.ego_vehicle_model,
                                         self.ego_vehicle_start,
                                         hero=True)

        super(StationaryObjectCrash, self).__init__(name="stationaryobjectcrash",
                                                    debug_mode=debug_mode)

    def create_behavior(self):
        """
        Example of a user defined scenario behavior. This function should be
        adapted by the user for other scenarios.
        """
        redundant = TimeOut(self.timeout)
        return redundant

    def create_test_criteria(self):
        """
        A list of all test criteria will be created
        that is later used in parallel behavior tree.
        """
        criteria = []

        max_velocity_criterion = MaxVelocityTest(
            self.ego_vehicle,
            self.ego_vehicle_max_velocity_allowed)
        collision_criterion = CollisionTest(self.ego_vehicle)
        keep_lane_criterion = KeepLaneTest(self.ego_vehicle)
        driven_distance_criterion = DrivenDistanceTest(
            self.ego_vehicle,
            self.ego_vehicle_distance_to_other)

        criteria.append(max_velocity_criterion)
        criteria.append(collision_criterion)
        criteria.append(keep_lane_criterion)
        criteria.append(driven_distance_criterion)

        return criteria


class DynamicObjectCrash(BasicScenario):
    """
    This class holds everything required for a simple object crash
    without prior vehicle action involving a vehicle and a cyclist,
    The ego vehicle is passing through a road,
    And encounters a cyclist crossing the road.
    """

    timeout = 60

    # ego vehicle parameters
    ego_vehicle_model = 'vehicle.carlamotors.carlacola'
    ego_vehicle_start_x = 90
    ego_vehicle_start = carla.Transform(
        carla.Location(x=ego_vehicle_start_x, y=129, z=1), carla.Rotation(yaw=180))
    ego_vehicle_max_velocity_allowed = 10
    ego_vehicle_distance_driven = 20

    # other vehicle parameters
    other_vehicles = []
    other_vehicle_model = 'vehicle.diamondback.century'
    other_vehicle_start_x = 47.5
    other_vehicle_start = carla.Transform(
        carla.Location(x=other_vehicle_start_x, y=124, z=1), carla.Rotation(yaw=90))
    other_vehicle_target_velocity = 10
    trigger_distance_from_ego_vehicle = 35
    other_vehicle_max_throttle = 1.0
    other_vehicle_max_brake = 1.0

    def __init__(self, world, debug_mode=False):
        """
        Setup all relevant parameters and create scenario
        and instantiate scenario manager
        """
        self.other_vehicles = [setup_vehicle(world,
                                             self.other_vehicle_model,
                                             self.other_vehicle_start)]
        self.ego_vehicle = setup_vehicle(world,
                                         self.ego_vehicle_model,
                                         self.ego_vehicle_start,
                                         hero=True)

        super(
            DynamicObjectCrash, self).__init__(name="dynamicobjectcrash",
                                               debug_mode=debug_mode)

    def create_behavior(self):
        """
        After invoking this scenario, cyclist will wait for the user
        controlled vehicle to enter the in the trigger distance region,
        the cyclist starts crossing the road once the condition meets,
        then after 60 seconds, a timeout stops the scenario

        """
        # leaf nodes
        trigger_dist = InTriggerDistanceToVehicle(
            self.other_vehicles[0],
            self.ego_vehicle,
            self.trigger_distance_from_ego_vehicle)
        start_other_vehicle = KeepVelocity(
            self.other_vehicles[0],
            self.other_vehicle_target_velocity)
        timeout_stop = TimeOut(11)
        stop_other_vehicle = StopVehicle(
            self.other_vehicles[0],
            self.other_vehicle_max_brake)
        timeout_other = TimeOut(20)
        root_timeout = TimeOut(self.timeout)

        # non leaf nodes
        root = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        scenario_sequence = py_trees.composites.Sequence()
        keep_velocity_other_parallel = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

        # building tree
        root.add_child(scenario_sequence)
        root.add_child(root_timeout)
        scenario_sequence.add_child(trigger_dist)
        scenario_sequence.add_child(keep_velocity_other_parallel)
        scenario_sequence.add_child(stop_other_vehicle)
        scenario_sequence.add_child(timeout_other)
        keep_velocity_other_parallel.add_child(start_other_vehicle)
        keep_velocity_other_parallel.add_child(timeout_stop)

        return root

    def create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = []

        max_velocity_criterion = MaxVelocityTest(
            self.ego_vehicle,
            self.ego_vehicle_max_velocity_allowed)
        collision_criterion = CollisionTest(self.ego_vehicle)
        keep_lane_criterion = KeepLaneTest(self.ego_vehicle)
        driven_distance_criterion = DrivenDistanceTest(
            self.ego_vehicle, self.ego_vehicle_distance_driven)

        criteria.append(max_velocity_criterion)
        criteria.append(collision_criterion)
        criteria.append(keep_lane_criterion)
        criteria.append(driven_distance_criterion)

        return criteria

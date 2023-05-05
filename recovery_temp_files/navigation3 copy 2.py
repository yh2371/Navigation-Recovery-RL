"""
A robot that can exert force in cardinal directions. The robot's goal is to
reach the origin and it experiences zero-mean Gaussian Noise. State
representation is (x, y, z). Action representation is (dx, dy, dz).
"""

import os
import pickle

import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
from gym import Env
from gym import utils
from gym.spaces import Box

from env.obstacle import Obstacle, ComplexObstacle3D
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import io
import cv2
import copy

import random
"""
Constants associated with the Navigation3 env.
"""

START_POS = [0.5, 2.5, 5.5]
END_POS = [5, 2.5, 0.5]
GOAL_THRESH = 0.05
START_STATE = START_POS
GOAL_STATE = END_POS

MAX_FORCE = 0.1
HORIZON = 200

NOISE_SCALE = 0.001
AIR_RESIST = 0.2

HARD_MODE = False

OBSTACLE = [[[4.1,4.9], [0.0,5.0], [2.1,6.0]],
            [[0.1,0.9],[0.0,5.0],[0.0,3.9]],
            [[-10,10],[-10,0.0],[-10,10]],
            [[-10,10],[5.0,10],[-10,10]],
            [[-10,10],[-10,10],[-10,0.0]],
            [[-10,10],[-10,10],[6.0,10]],
            [[-10,0],[-10,10],[-10,10]],
            [[6,10],[-10,10],[-10,10]]] #4x3x2

#self.observation_space = Box(np.array([0.0,0.0,0.0]),
# np.array([11.0,5.0,6.0]))

def get_formatted_obs(obs):
    new_obs = []
    for i in obs:
        new_obs.append([i[0][0],i[0][1],i[1][0],i[1][1],i[2][0],i[2][1]])
    return np.array(new_obs)

# {"extents": [3.1, 3.9, 0.0, 5.0, 2.1, 6.0], "color": [1.0, 0.0, 0.0]},
# {"extents": [9.1, 9.9, 0.0, 5.0, 2.1, 6.0], "color": [1.0, 0.0, 0.0]},
# {"extents": [0.1, 0.9, 0.0, 5.0, 0.0, 3.9], "color": [0.0, 0.0, 1.0]},
# {"extents": [6.1, 6.9, 0.0, 5.0, 0.0, 3.9], "color": [0.0, 0.0, 1.0]}

CAUTION_ZONE = [[[4.1-0.25,4.9+0.25], [0.0-0.25,5.0+0.25], [2.1-0.25,6.0+0.25]],
            [[0.1-0.25,0.9+0.25],[0.0-0.25,5.0+0.25],[0.0-0.25,3.9+0.25]],
            [[-10,10],[-10,0.0+0.25],[-10,10]],
            [[-10,10],[5.0-0.25,10],[-10,10]],
            [[-10,10],[-10,10],[-10,0.0+0.25]],
            [[-10,10],[-10,10],[6.0-0.25,10]],
            [[-10,0+0.25],[-10,10],[-10,10]],
            [[6-0.25,10],[-10,10],[-10,10]]]

OBSTACLE_COORD = get_formatted_obs(CAUTION_ZONE)

OBSTACLE = ComplexObstacle3D(OBSTACLE)
CAUTION_ZONE = ComplexObstacle3D(CAUTION_ZONE)


def process_action(a):
    return np.clip(a, -MAX_FORCE, MAX_FORCE)


class Navigation3(Env, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        self.hist = self.cost = self.done = self.time = self.state = None
        self.A = np.eye(3)
        self.B = np.eye(3)
        self.horizon = HORIZON
        self.action_space = Box(-np.ones(3) * MAX_FORCE,
                                np.ones(3) * MAX_FORCE)
        self.observation_space = Box(np.array([0.0,0.0,0.0]),
                                     np.array([11.0,5.0,6.0]))
        #{"extents": [0.0, 11.0, 0.0, 5.0, 0.0, 6.0]}
        self.max_episode_steps = HORIZON
        self.obstacle = OBSTACLE
        self.caution_zone = CAUTION_ZONE
        self.transition_function = get_offline_data
        self.goal = GOAL_STATE

    def step(self, a):
        a = process_action(a)
        old_state = self.state.copy()
        next_state = self._next_state(self.state, a)
        cur_cost = self.step_cost(self.state, a)
        self.cost.append(cur_cost)
        self.state = next_state
        self.time += 1
        self.hist.append(self.state)
        self.done = cur_cost > -4 or self.obstacle(next_state)

        return self.state, cur_cost, self.done, {
            "constraint": self.obstacle(next_state),
            "reward": cur_cost,
            "state": old_state,
            "next_state": next_state,
            "action": a,
            "success": cur_cost>-4
        }

    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)
        
    def reset(self):
        print("Start",START_STATE)
        self.state = copy.deepcopy(START_STATE) + NOISE_SCALE * np.random.randn(3)
        self.time = 0
        self.cost = []
        self.done = False
        self.hist = [self.state]
        return self.state

    def _next_state(self, s, a, override=False):
        if self.obstacle(s):
            print("obs", s, a)
            return s
        return self.A.dot(s) + self.B.dot(a) + NOISE_SCALE * np.random.randn(
            len(s))

    def step_cost(self, s, a):
        if HARD_MODE:
            return int(
                np.linalg.norm(np.subtract(GOAL_STATE, s)) < GOAL_THRESH)
        return -np.linalg.norm(np.subtract(GOAL_STATE,
                                           s)) - self.obstacle(s) * 100

    def values(self):
        return np.cumsum(np.array(self.cost)[::-1])[::-1]

    def sample(self):
        """
        samples a random action from the action space.
        """
        return np.random.random(3) * 2 * MAX_FORCE - MAX_FORCE

    def plot_trajectory(self, states=None):
        if states == None:
            states = self.hist
        states = np.array(states)
        plt.scatter(states[:, 0], states[:, 2])
        plt.show()

    # Returns whether a state is stable or not
    def is_stable(self, s):
        return np.linalg.norm(np.subtract(GOAL_STATE, s)) <= GOAL_THRESH

def get_action_to_closest_obs(point, boxes):
    # point: np array of shape (3,)
    # box: np array of shape (N, 6)
    d = np.zeros((boxes.shape[0],3))
    for j in range(boxes.shape[0]):
        for i in range(3):
            if point[i] < boxes[j, 2*i]:
                d[j,i] = boxes[j,2*i] - point[i]
            elif point[i] > boxes[j,2*i+1]:
                d[j,i] = -(point[i] - boxes[j, 2*i+1])
    dist = np.linalg.norm(d, axis = 1)
    idx = np.argmin(dist)
    action = d[idx] - point
    action = action/np.linalg.norm(action)
    return action*MAX_FORCE

def get_offline_data(num_transitions, task_demos=False, save_rollouts=False):
    env = Navigation3()
    transitions = []
    rollouts = []
    done = False
    for i in range(num_transitions):
        state = np.array(
            [np.random.uniform(0, 6),
             np.random.uniform(0, 5),
             np.random.uniform(0, 6)])
        while env.obstacle(state):
            state = np.array(
                [np.random.uniform(0, 6),
                np.random.uniform(0, 5),
                np.random.uniform(0, 6)])
        for j in range(20):
                if i < 3*num_transitions//2:
                    action = process_action(get_action_to_closest_obs(state, OBSTACLE_COORD) + NOISE_SCALE*np.random.randn(3))
                else:
                    action = process_action(np.random.randn(3))
                next_state = env._next_state(state, action, override=True)
                constraint = env.obstacle(next_state)
                reward = env.step_cost(state, action)
                transitions.append(
                    (state, action, constraint, next_state, not constraint))
            # rollouts[-1].append(
            #     (state, action, constraint, next_state, not constraint))
                state = next_state
                if constraint:
                    break

    if save_rollouts:
        return rollouts
    else:
        return transitions

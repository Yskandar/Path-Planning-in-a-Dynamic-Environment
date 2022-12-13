#!/usr/bin/env python

__author__ = "david kao"

import numpy as np
import random
from utils import move

#obstacle: generic obstacle class for global map to keep track of moving obstacles.
#params: movement
#movement is a sequence of actions that corresponds to the obstacle moves
#should be pre-determined through #defines, for example
#define VERTICAL = [up, down, down, up]
#start: starting coordinates
class Obstacle(object):
    def __init__(self, movement, start):
        self.movement = movement
        if np.all(self.movement == move.RANDOM):
            #print("random obstacle")
            self.next = self.next_random
        else:
            #print("path set")
            self.next = self.next_deterministic
        self.location = np.array(start)

    #next: returns what this obstacle's next movement is.
    #input: global time step from zero - use mod to keep cycling actions
    def next_deterministic(self, time):
        period = len(self.movement)
        return self.movement[time % period]

    def next_random(self, time):
        return random.choice(move.RANDOM)

class Agent(object):
    def __init__(self, start):
        self.location = np.array(start)
        self.range = 1 #observe range
        self.action = move.NONE

    def next(self):
        return self.action #this class decides what its move is, global updates the position
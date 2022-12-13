#!/usr/bin/env python

import numpy as np

NONE    = np.array([0,0])
LEFT   = np.array([-1,0])
RIGHT    = np.array([1,0])
DOWN    = np.array([0,-1])
UP   = np.array([0,1])

NE = np.array([1,1])
NW = np.array([-1,1])
SE = np.array([1,-1])
SW = np.array([-1,-1])

UD      = np.array((UP,DOWN))
LR      = np.array((LEFT, RIGHT))
RANDOM  = np.array((NONE, LEFT, RIGHT, UP, DOWN))
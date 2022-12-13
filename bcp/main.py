import sys

import argparse
import numpy as np
import random
import math
import move
import matplotlib.pyplot as plt
from QuadMDP.QuadTree import Point, Rect
from QuadMDP.QuadMDP import QuadMDP
from utils.Obstacle import Obstacle, Agent
from utils.Global import Global
from utils.agent_graph import *

__author__ = "David Kao", "Hayato Kato", "Yskandar Gas"

def main(args):
	print("Starting...")
	# Load stationary world obstacle map
	filename = args.input
	image = plt.imread('map/'+filename)
	mapWidth, mapHeight = image.shape
	print("Map Width:", mapWidth)
	print("Map height:", mapHeight)
	mapDepth = (int)(math.log(mapWidth,2))
	searchDepth = 2 # Search depth within QuadMDP

	# Global class for the whole environment
	g = Global((mapWidth,mapHeight))
	# QuadMDP class for generating the quad decomposed states
	quad = QuadMDP(Rect(mapWidth/2-0.5,mapHeight/2-0.5,mapWidth,mapHeight),mapDepth)

	# Load all obstacles
	for x in range(mapWidth):
		for y in range(mapHeight):
			if image[y,x] == 0:
				g.createObstacle(move.NONE,(x,y))
				quad.insert(Point(x,y,True))

	# Generate quad decomposed states
	S = quad.findEmptySpace(searchDepth)

	graph = quad.generateGraph(S, searchDepth)

	startPos = (8,8)
	goalPos = (119,119)
	startQuadMDP = quad.findContainedQuadMDP(startPos)
	goalQuadMDP = quad.findContainedQuadMDP(goalPos)

	path = quad.getOptimalPath(S,searchDepth,startQuadMDP.getTuple(),goalQuadMDP.getTuple())
	path.insert(0,startPos)
	path.append(goalPos)

	for i,p in enumerate(path):
		match random.randint(0,3):
			case 0:
				path[i] = (math.floor(p[0]), math.floor(p[1]))
			case 1:
				path[i] = (math.ceil(p[0]), math.floor(p[1]))
			case 2:
				path[i] = (math.floor(p[0]), math.ceil(p[1]))
			case 3:
				path[i] = (math.ceil(p[0]), math.ceil(p[1]))

	print(path)

	agent = g.createAgent(startPos)
	agent.range = 3

	obs = g.observe()
	print(obs)


	# Setup Figure
	fig = plt.figure()
	ax = plt.subplot()

	quad.draw(ax,False)
	startQuadMDP.draw(ax)
	goalQuadMDP.draw(ax)
	ax.scatter(startPos[0],startPos[1],zorder=1)
	ax.scatter(goalPos[0],goalPos[1],zorder=1)
	ax.plot(*zip(*path),'-o')

	ax.set_xlim([-0.5,mapWidth-0.5])
	ax.set_ylim([-0.5,mapHeight-0.5])
	ax.set_aspect('equal')
	ax.invert_yaxis()

	plt.show()

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Quad Decomposition Demo')
	parser.add_argument('-i', '--input', help='Grayscale PNG map file name', default='worldmap.png')
	args = parser.parse_args()
	main(args)
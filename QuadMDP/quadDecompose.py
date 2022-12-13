#!/usr/bin/env python

""" main.py """

__author__ = "Hayato Kato" 

import argparse
import numpy as np
from collections import defaultdict
import random
import time
import math
import matplotlib.pyplot as plt

from QuadMDP.QuadTree import Point, Rect
from QuadMDP.QuadMDP import QuadMDP
from bcp.graph_utils import Graph

def main(args):
	print("Starting...")

	plt.rcParams['text.usetex'] = True

	# Setup Figure
	fig = plt.figure(figsize=(8,8))
	ax = plt.subplot()

	# Load Stationary Obstacle Image Data
	image = plt.imread('map/'+args.input)
	width, height = image.shape

	depth = (int)(math.log(width,2))
	print(depth)

	# Initialize with bounds of Quadtree and depth of tree
	quad = QuadMDP(Rect(width/2-0.5,height/2-0.5,width,height),depth)

	# Iterate through all obstacles loaded from image
	for i in range(width):
		for j in range(height):
			if image[j,i] == 0:
				quad.insert(Point(i,j,True))

	searchDepth = 1

	# Search empty states within quadtree
	# Parameter determines how deep we want to construct our quadtree
	# Smaller number means the deepest, i.e. more states
	# Larger number means more high level, i.e. less states
	S = quad.findEmptySpace(searchDepth)

	# Generate graph usable by graph-search based path planning
	V = [s.getTuple() for s in S]
	E = quad.generateGraph(S, searchDepth)

	S1 = quad.findEmptySpace(1)
	V1 = [s.getTuple() for s in S1]
	E1 = quad.generateGraph(S1, 1)

	print('Vertices(' + str(len(V)) + ')')
	print('Edges(' + str(len(E)) + ')')

	# Draw QuadMDP Structure 
	# (border=False means dont draw the Quadtree decomposition but still draw the obstacles)
	quad.draw(ax, border=True)

	startPos = (0,0)
	goalPos = (15,15)
	startQuadMDP = quad.findContainedQuadMDP(startPos)
	goalQuadMDP = quad.findContainedQuadMDP(goalPos)

	# Plot all vertices and edges
	for v1 in V:
		#if v1 in V1:
	#		continue
		for v2 in E[v1]:
			plt.plot([v1[0],v2[0]],[v1[1],v2[1]],'-',color='b',lw=2,zorder=-1)
		plt.scatter(v1[0],v1[1],20,color='k',edgecolor='k',zorder=3)

	#path = quad.getOptimalPath(S,searchDepth,startQuadMDP.getTuple(),goalQuadMDP.getTuple())
	#path.insert(0,startPos)
	#path.append(goalPos)

	#ax.scatter(startPos[0],startPos[1],50,'lime',zorder=5)
	#ax.scatter(goalPos[0],goalPos[1],50,'purple',zorder=5)

	#ax.plot(*zip(*path),'--',lw=2,zorder=2)

	# Plot Formatting
	ax.set_xlim([-0.5,width-0.5])
	ax.set_ylim([-0.5,height-0.5])
	ax.invert_yaxis()
	ax.set_aspect('equal')
	plt.tight_layout()
	plt.show()
	#plt.savefig('worldMap_graph.png')
		
	print("Ending...")

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Quad Decomposition Demo')
	parser.add_argument('-i', '--input', help='Grayscale PNG map file name', default='worldmap.png')
	args = parser.parse_args()
	main(args)
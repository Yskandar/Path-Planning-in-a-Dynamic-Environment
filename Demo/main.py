#!/usr/bin/env python

""" QuadMDP.py """

import sys
import argparse
import time
from tqdm import tqdm
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from QuadTree import Point, Rect
from QuadMDP import QuadMDP
#from Obstacle import Obstacle, Agent
#from Global import Global
#from agent_graph import *

__author__ = "David Kao", "Hayato Kato", "Yskandar Gas"

def main(args):
	print("Starting...")

	filename = args.input
	image = plt.imread('map/' + filename)
	mapWidth, mapHeight = image.shape
	mapDepth = (int)(math.log(mapWidth,2))
	searchDepth = 0#int(input('Input a search depth:'))

	quad = QuadMDP(Rect(mapWidth/2-0.5,mapHeight/2-0.5,mapWidth,mapHeight),mapDepth)
	quad_original = QuadMDP(Rect(mapWidth/2-0.5,mapHeight/2-0.5,mapWidth,mapHeight),mapDepth)

	# Load all obstacles
	for x in range(mapWidth):
		for y in range(mapHeight):
			#plt.close("all")
			#ax = plt.subplots()
			if image[y,x] == 0:
				quad.insert(Point(x,y,True))
				quad_original.insert(Point(x,y,True))
			else:
				quad_original.insert(Point(x,y,False))

	fig = plt.figure(1,figsize=(9,9))
	ax = plt.subplot()

	quad_original.draw(ax,True)

	ax.set_xlim([-0.5,mapWidth-0.5])
	ax.set_ylim([-0.5,mapHeight-0.5])
	ax.set_aspect('equal')
	ax.invert_yaxis()
	ax.set_title('Step 0: Original State Space')

	plt.tight_layout()
	plt.show()

	fig = plt.figure(2,figsize=(9,9))
	ax = plt.subplot()

	quad.draw(ax,True)

	ax.set_xlim([-0.5,mapWidth-0.5])
	ax.set_ylim([-0.5,mapHeight-0.5])
	ax.set_aspect('equal')
	ax.invert_yaxis()
	ax.set_title('Step 1: Quad Decomposed State Space')

	plt.tight_layout()
	plt.show()

	startPos = (1,1)
	goalPos = (65,65)

	startQuadMDP = quad.findContainedQuadMDP(startPos)
	goalQuadMDP = quad.findContainedQuadMDP(goalPos)
	for depth in reversed(range(mapDepth)):
		S = quad.findEmptyState(depth)
		V = [s.getTuple() for s in S]
		E = quad.generateGraph(S, depth)
		if len(E) == 0:
			continue

		fig = plt.figure(3, figsize=(9,9))
		ax = plt.subplot()

		for v1 in tqdm(V):
			for v2 in E[v1]:
				ax.plot([v1[0],v2[0]],[v1[1],v2[1]],'-',color='b',lw=1,zorder=-1)
			ax.scatter(v1[0],v1[1],2,color='k',edgecolor='k',zorder=3)

		quad.draw(ax,True)
		ax.scatter(startPos[0],startPos[1],edgecolor='k',zorder=3)
		ax.scatter(goalPos[0],goalPos[1],edgecolor='k',zorder=3)

		ax.set_xlim([-0.5,mapWidth-0.5])
		ax.set_ylim([-0.5,mapHeight-0.5])
		ax.set_aspect('equal')
		ax.invert_yaxis()
		ax.set_title('Step 2: Directed Graph used for Graph-Search Based Motion Planning (Depth = ' + str(depth) + ')')

		plt.tight_layout()
		plt.show()

		path = quad.getOptimalPath(S,searchDepth,startQuadMDP.getTuple(),goalQuadMDP.getTuple())
		if path != None:
			searchDepth = depth
			break

	S = quad.findEmptyState(searchDepth)
	V = [s.getTuple() for s in S]
	E = quad.generateGraph(S, searchDepth)
	path = quad.getOptimalPath(S,searchDepth,startQuadMDP.getTuple(),goalQuadMDP.getTuple())
	if path != None:
		path.insert(0,startPos)
		path.append(goalPos)
	else:
		sys.exit('No Connected Path Found')

	fig = plt.figure(4, figsize=(9,9))
	ax = plt.subplot()

	quad.draw(ax,True)

	ax.scatter(startPos[0],startPos[1],edgecolor='k',zorder=3)
	ax.scatter(goalPos[0],goalPos[1],edgecolor='k',zorder=3)

	ax.plot(*zip(*path),'b--',lw=2,zorder=2)

	ax.set_xlim([-0.5,mapWidth-0.5])
	ax.set_ylim([-0.5,mapHeight-0.5])
	ax.set_aspect('equal')
	ax.invert_yaxis()
	ax.set_title('Step 3: Path From Start to Goal Using BFS')

	plt.tight_layout()
	plt.show()


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Quad Decomposition Demo')
	parser.add_argument('-i', '--input', help='Grayscale PNG map file name', default='worldmap.png')
	args = parser.parse_args()
	main(args)


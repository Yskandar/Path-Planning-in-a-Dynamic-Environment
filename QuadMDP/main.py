#!/usr/bin/env python

""" QuadMDP.py """

__author__ = "Hayato Kato" 

import numpy as np
import random
import time
import matplotlib.pyplot as plt

from QuadTree import Point, Rect
from QuadMDP import QuadMDP

def main():
	print("Starting...")

	fig = plt.figure()
	ax = plt.subplot()

	image = plt.imread('worldmap.png')
	width, height = image.shape

	# Initialize with bounds of Quadtree and depth of tree
	quad = QuadMDP(Rect(width/2-0.5,height/2-0.5,width,height),7)

	# Iterate through all static obstacles loaded from image
	for i in range(width):
		for j in range(height):
			if image[j,i] == 0:
				quad.insert(Point(i,j,True))

	# Draw QuadMDP Structure
	quad.draw(ax, border = False)

	# Search empty states within quadtree
	S = quad.findEmptySpace(1)

	# Draw entire state space S
	sx = []
	sy = []
	for i in range(len(S)):
		cState = S[i]
		sx.append(cState.point.x)
		sy.append(cState.point.y)
		N = cState.findNeighbors(1)
		for nState in N:
			plt.plot([cState.point.x,nState.point.x],[cState.point.y,nState.point.y],lw=1,zorder=-1)
	plt.scatter(sx,sy,2,color='k',edgecolor='k',zorder=0)

	print(len(S))

	# Plot Formatting
	ax.set_xlim([-0.5,width-0.5])
	ax.set_ylim([-0.5,height-0.5])
	ax.invert_yaxis()
	ax.set_aspect('equal')
	plt.tight_layout()
	plt.show()

	print("Ending...")

if __name__ == '__main__':
	main()
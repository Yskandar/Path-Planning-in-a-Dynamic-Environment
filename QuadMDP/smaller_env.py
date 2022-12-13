
import numpy as np
import random
import time
import math
import matplotlib.pyplot as plt

from QuadTree import Point, Rect
from QuadMDP import QuadMDP
#from graph_utils import Graph
from PIL import Image

def main():
	print("Starting...")

	# Setup Figure
	fig = plt.figure()
	ax = plt.subplot()
	# Load Stationary Obstacle Image Data
	#image = plt.imread('QuadMDP/worldmap.png')
	img = Image.open('QuadMDP/worldmap.png')
	image = np.asarray(img.resize((np.array(img.size)/2).astype(int)))
	width, height = image.shape

	depth = (int)(math.log(width,2))
	print(depth)

	# Initialize with bounds of Quadtree and depth of tree
	quad = QuadMDP(Rect(width/2-0.5,height/2-0.5,width,height),depth)

	# Iterate through all obstacles loaded from image
	for i in range(width):
		for j in range(height):
			if image[j,i] <= 200:
				quad.insert(Point(i,j,True))

	# Draw QuadMDP Structure 
	# (border=False means dont draw the Quadtree decomposition but still draw the obstacles)
	quad.draw(ax, border=False)

	# Search empty states within quadtree
	# Parameter determines how deep we want to construct our quadtree
	# Smaller number means the deepest, i.e. more states
	# Larger number means more high level, i.e. less states
	S = quad.findEmptySpace(1)


	initial_state = S[0]
	goal_state = S[70]
	
	path = quad.get_optimal_path(1, initial_state, goal_state)
	sx = []
	sy = []

	for cState in S:
		sx.append(cState.point.x)
		sy.append(cState.point.y)
		#vertices.append(cState.getTuple())
		N = cState.findNeighbors(1)
		for nState in N:
			#edges.append((cState.getTuple(),nState.getTuple()))
			plt.plot([cState.point.x,nState.point.x],[cState.point.y,nState.point.y],color='b',lw=1,zorder=-1)
	for i in range(len(path)-1):
		cState, nState = path[i], path[i+1]
		plt.plot([cState.point.x,nState.point.x],[cState.point.y,nState.point.y],color='r',lw=1,zorder=-1)
	plt.scatter(sx,sy,2,color='k',edgecolor='k',zorder=0)
	print(path)

	# Plot Formatting
	ax.set_xlim([-0.5,width-0.5])
	ax.set_ylim([-0.5,height-0.5])
	ax.invert_yaxis()
	ax.set_aspect('equal')
	plt.tight_layout()
	plt.show()

	"""

	# Go through each quadtree state and connect neighboring states with edges
	sx = []
	sy = []
	vertices = []
	edges = []
	for cState in S:
		sx.append(cState.point.x)
		sy.append(cState.point.y)
		vertices.append(cState.getTuple())
		N = cState.findNeighbors(1)
		for nState in N:
			edges.append((cState.getTuple(),nState.getTuple()))
			plt.plot([cState.point.x,nState.point.x],[cState.point.y,nState.point.y],color='b',lw=1,zorder=-1)
	plt.scatter(sx,sy,2,color='k',edgecolor='k',zorder=0)

	print('\nVertices(' + str(len(vertices)) + '):')
	print(vertices)
	print('\nEdges(' + str(len(edges)) + '):')
	print(edges)

	# Plot Formatting
	ax.set_xlim([-0.5,width-0.5])
	ax.set_ylim([-0.5,height-0.5])
	ax.invert_yaxis()
	ax.set_aspect('equal')
	plt.tight_layout()
	plt.show()
	"""
	print("Ending...")

if __name__ == '__main__':
	main()
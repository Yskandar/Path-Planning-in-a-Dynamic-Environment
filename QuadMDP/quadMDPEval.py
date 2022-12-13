#!/usr/bin/env python

""" main.py """

__author__ = "Hayato Kato" 

import argparse
import numpy as np
from collections import defaultdict
import random
import time
import math
import csv
import matplotlib.pyplot as plt

from QuadMDP.QuadTree import Point, Rect
from QuadMDP.QuadMDP import QuadMDP
from bcp.graph_utils import Graph
from RandomMap import RandomMap

def main(args):
	print("Starting...")

	dim = 32
	total = dim**2

	f = open('data.csv','w')
	writer = csv.writer(f)

	header = [
		'Threshold', 
		'Resolution', 
		'Obstacle %',
		'Base V #', 
		'Base E #', 
		#'QuadMDP Time'
	]
	for searchDepth in range((int)(math.log(dim,2))):
		#header.append('D(' + str(searchDepth) + ') Graph Time')
		header.append('D(' + str(searchDepth) + ') V #')
		header.append('D(' + str(searchDepth) + ') E #')
	writer.writerow(header)

	# Setup Figure
	fig = plt.figure()
	ax = plt.subplot()

	for thresh in np.linspace(0.3,0.7,5):
		for iter in range(500):

			#thresh = 0.7
			resolution = random.uniform(5,15)

			thresh = round(thresh,2)

			print(iter, thresh)

			randImage = RandomMap((dim,dim),thresh,resolution) # Smaller = More Detail
			image = randImage.image
			width,height = randImage.xDim, randImage.yDim
			depth = (int)(math.log(width,2))
			quad_original = QuadMDP(Rect(width/2-0.5,height/2-0.5,width,height),depth)
			quad_decomposed = QuadMDP(Rect(width/2-0.5,height/2-0.5,width,height),depth)
			obsCount = 0
			for x in range(width):
				for y in range(height):
					if image[y,x] == 0:
						obsCount += 1
						quad_original.insert(Point(x,y,True))
					else:
						quad_original.insert(Point(x,y,False))
			S_o = quad_original.findEmptySpace(0)
			E_o = quad_original.generateGraph(S_o, 0)
			V_o = [s.getTuple() for s in S_o]
			lenV_o = len(V_o)
			lenE_o = 0
			for key in E_o.keys():
				lenE_o += len(E_o[key])

			time_d = time.time()
			for x in range(width):
				for y in range(height):
					if image[y,x] == 0:
						quad_decomposed.insert(Point(x,y,True))
			S_d = quad_decomposed.findEmptySpace(0)
			time_d = time.time() - time_d

			data = [
				thresh, 
				resolution,
				obsCount/total,
				lenV_o, 
				lenE_o, 
				#time_d
			]

			for searchDepth in range(depth):
				time_g = time.time()
				E_d = quad_decomposed.generateGraph(S_d, searchDepth)
				time_g = time.time() - time_g
				V_d = []
				for s in S_d:
					if s.depth >= searchDepth:
						V_d.append(s.getTuple())
				lenV_d = len(V_d)
				lenE_d = 0
				for key in E_d.keys():
					lenE_d += len(E_d[key])

				#data.append(time_g)
				data.append(lenV_d)
				data.append(lenE_d)


			writer.writerow(data)

		
		# Plot
		#quad_decomposed.draw(ax,False)
		'''
		for v1 in V_o:
			for v2 in E_o[v1]:
				plt.plot([v1[0],v2[0]],[v1[1],v2[1]],color='b',lw=1,zorder=-1)
			plt.scatter(v1[0],v1[1],3,color='k',edgecolor='k',zorder=0)
		'''

		#ax.set_xlim([-0.5,width-0.5])
		#ax.set_ylim([-0.5,height-0.5])
		#ax.invert_yaxis()
		#ax.set_aspect('equal')
		#plt.tight_layout()
		#plt.show()
		#plt.savefig('randomMap_' + str(thresh) + '_' + str(resolution) + '.png')
		
		


	f.close()


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Quad Decomposition Demo')
	parser.add_argument('-i', '--input', help='Grayscale PNG map file name', default='worldmap.png')
	args = parser.parse_args()
	main(args)
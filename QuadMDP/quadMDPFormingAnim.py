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
import matplotlib.animation as animation

from QuadMDP.QuadTree import Point, Rect
from QuadMDP.QuadMDP import QuadMDP
from bcp.graph_utils import Graph
from RandomMap import RandomMap

def main(args):
	print("Starting...")

	global image, quad_decomposed,width,height,dim
	dim = 16
	total = dim**2

	#f = open('collected_data.csv','w')
	#writer = csv.writer(f)

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
	#writer.writerow(header)

	# Setup Figure
	fig = plt.figure(figsize=(6,6))
	ax = plt.subplot()


	for iter in range(1):

		thresh = 0.7
		resolution = 6

		print(iter, thresh)

		
		image = plt.imread('map/smile.png')
		width, height = image.shape
		depth = (int)(math.log(width,2))

		quad_decomposed = QuadMDP(Rect(width/2-0.5,height/2-0.5,width,height),depth)


		#plt.savefig('randomMap_' + str(thresh) + '_' + str(resolution) + '.png')
		anim = animation.FuncAnimation(fig, animate, fargs = (ax,), interval = 1,frames = dim*dim,repeat=False)

		plt.tight_layout()
		anim.save('quadDecomposedSmileAnim.gif', writer='imagemagick',fps=60,extra_args=['-loop','0'])
		#plt.show()
		
def animate(time, ax):
	global image, quad_decomposed,width,height,dim
	x = time // dim
	y = time % dim
	if x >= dim or y >= dim:
		return
	print(x,y)
	plt.cla()
	if image[y,x] == 0:
		quad_decomposed.insert(Point(x,y,True))
	quad_decomposed.draw(ax,True)
	ax.set_xlim([-0.5,width-0.5])
	ax.set_ylim([-0.5,height-0.5])
	ax.invert_yaxis()
	ax.set_aspect('equal')
	return



if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Quad Decomposition Demo')
	parser.add_argument('-i', '--input', help='Grayscale PNG map file name', default='worldmap.png')
	args = parser.parse_args()
	main(args)
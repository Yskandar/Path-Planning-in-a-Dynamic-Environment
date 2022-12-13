#!/usr/bin/env python

""" QuadMDP.py """

__author__ = "Hayato Kato" 

import os
import sys
dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir)

from copy import copy
import numpy as np
import matplotlib.pyplot as plt
import cv2

from QuadTree import Point, Rect, QuadTree

class QuadMDP:
	def __init__(self, worldmap):
		# State Space
		self.X_dim, self.Y_dim = worldmap.transpose().shape
		self.domain = Rect(self.X_dim/2, self.Y_dim/2, self.X_dim, self.Y_dim)
		self.QT = QuadTree(self.domain, 1)
		self.S = []
		self.S_obs = []

		for i in range(self.X_dim):
			for j in range(self.Y_dim):
				self.S.append((i,j))
				if worldmap[j,i] == 0:
					self.QT.insert(Point(i,j))
					self.S_obs.append((i,j))

	def draw(self, ax):
		self.QT.draw(ax)
		for x,y in self.S_obs:
			rx = [x,x+1,x+1,x]
			ry = [y,y,y+1,y+1]
			plt.fill(rx,ry,color='lightgrey',zorder=-2)

def main(): 
	print("Starting...")

	DPI = 72
	image = plt.imread('worldmap.png')
	width, height = image.shape
	mdp = QuadMDP(image)

	
	fig = plt.figure(dpi=DPI)
	ax = plt.subplot()
	#ax.set_xlim(0, width)
	#ax.set_ylim(0, height)

	mdp.draw(ax)

	ax.invert_yaxis()
	ax.set_aspect('equal')

	ax.set_xticks(np.arange(0, mdp.X_dim+1, 16))
	ax.set_yticks(np.arange(0, mdp.Y_dim+1, 16))\

	plt.tight_layout()
	plt.show()
	



if __name__ == '__main__':
	main()

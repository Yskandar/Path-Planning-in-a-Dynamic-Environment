#!/usr/bin/env python

""" QuadMDP.py """

from __future__ import annotations

__author__ = "Hayato Kato" 

import os
import sys
dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir)

from copy import copy
from enum import *

import math
import time
import random
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import copy

from QuadTree import Point, Rect

class QuadMDP:

	class Child(IntEnum):
		NW = 0
		NE = 1
		SW = 2
		SE = 3

	class Direction(Enum):
		#NW = 0
		#NE = 1
		#SW = 2
		#SE = 3
		N = 4
		S = 5
		W = 6
		E = 7

	def __init__(self, boundary, depth, parent = None):
		self.parent = parent
		self.child = []

		self.boundary = boundary
		self.depth = depth
		cx, cy = self.boundary.cx, self.boundary.cy
		self.point = Point(cx,cy,False)
		self.divided = False

	def divide(self):
		cx, cy = self.boundary.cx, self.boundary.cy
		w, h = self.boundary.w / 2, self.boundary.h / 2
		self.child.append(QuadMDP(Rect(cx - w/2, cy - h/2, w, h), self.depth - 1, self))
		self.child.append(QuadMDP(Rect(cx + w/2, cy - h/2, w, h), self.depth - 1, self))
		self.child.append(QuadMDP(Rect(cx - w/2, cy + h/2, w, h), self.depth - 1, self))
		self.child.append(QuadMDP(Rect(cx + w/2, cy + h/2, w, h), self.depth - 1, self))
		self.divided = True

	def insert(self, point):
		if not self.boundary.contains(point):
			return False

		if self.depth == 0:
			self.point = point
			return True

		if not self.divided:
			self.point = None
			self.divide()

		for index in self.Child:
			if self.child[index].insert(point):
				return True
		return False

	def findEmptySpace(self, depth:int) -> list[QuadMDP]:
		space = []
		if self.divided:
			for index in self.Child:
				space.extend(self.child[index].findEmptySpace(depth))
		else:
			if self.point.payload == False and self.depth >= depth:
				cx, cy = self.boundary.cx, self.boundary.cy
				space.append(self)
		return space

	def findLargerEqualNeighbor(self, direction):
		# If root node is reached
		if self.parent is None:
			return None
		'''
		if direction == self.Direction.NW: # 0
			node = None
			if   self.parent.child[self.Child.NW] == self:
				node = self.parent.findLargerEqualNeighbor(self.Direction.NW)
				return node if node is None or node.divided == False else node.child[self.Child.SE]
			elif self.parent.child[self.Child.NE] == self:
				node = self.parent.findLargerEqualNeighbor(self.Direction.N)
				return node if node is None or node.divided == False else node.child[self.Child.SW]
			elif self.parent.child[self.Child.SW] == self:
				node = self.parent.findLargerEqualNeighbor(self.Direction.W)
				return node if node is None or node.divided == False else node.child[self.Child.NE]
			elif self.parent.child[self.Child.SE] == self:
				return self.parent.child[self.Child.NW]

		elif direction == self.Direction.NE: # 1
			node = None
			if   self.parent.child[self.Child.NW] == self:
				node = self.parent.findLargerEqualNeighbor(self.Direction.N)
				return node if node is None or node.divided == False else node.child[self.Child.SE]
			elif self.parent.child[self.Child.NE] == self:
				node = self.parent.findLargerEqualNeighbor(self.Direction.NE)
				return node if node is None or node.divided == False else node.child[self.Child.SW]
			elif self.parent.child[self.Child.SW] == self:
				return self.parent.child[self.Child.NE]
			elif self.parent.child[self.Child.SE] == self:
				node = self.parent.findLargerEqualNeighbor(self.Direction.E)
				return node if node is None or node.divided == False else node.child[self.Child.NW]

		elif direction == self.Direction.SW: # 2
			node = None
			if   self.parent.child[self.Child.NW] == self:
				node = self.parent.findLargerEqualNeighbor(self.Direction.W)
				return node if node is None or node.divided == False else node.child[self.Child.SE]
			elif self.parent.child[self.Child.NE] == self:
				return self.parent.child[self.Child.SW]
			elif self.parent.child[self.Child.SW] == self:
				node = self.parent.findLargerEqualNeighbor(self.Direction.SW)
				return node if node is None or node.divided == False else node.child[self.Child.NE]
			elif self.parent.child[self.Child.SE] == self:
				node = self.parent.findLargerEqualNeighbor(self.Direction.S)
				return node if node is None or node.divided == False else node.child[self.Child.NW]
		
		elif direction == self.Direction.SE: # 3
			node = None
			if   self.parent.child[self.Child.NW] == self:
				return self.parent.child[self.Child.SE]
			elif self.parent.child[self.Child.NE] == self:
				node = self.parent.findLargerEqualNeighbor(self.Direction.E)
				return node if node is None or node.divided == False else node.child[self.Child.SW]
			elif self.parent.child[self.Child.SW] == self:
				node = self.parent.findLargerEqualNeighbor(self.Direction.S)
				return node if node is None or node.divided == False else node.child[self.Child.NE]
			elif self.parent.child[self.Child.SE] == self:
				node = self.parent.findLargerEqualNeighbor(self.Direction.SE)
				return node if node is None or node.divided == False else node.child[self.Child.NW]
		'''
		if direction == self.Direction.N: # 4
			# If neighbor is located within same parent
			if self.parent.child[self.Child.SW] == self:
				return self.parent.child[self.Child.NW]
			if self.parent.child[self.Child.SE] == self:
				return self.parent.child[self.Child.NE]
			# Check parent's neighboring node for a child
			node = self.parent.findLargerEqualNeighbor(self.Direction.N)
			if node is None or node.divided == False:
				return node
			# Return neighboring parent node's child
			if self.parent.child[self.Child.NW] == self:
				return node.child[self.Child.SW]
			else:
				return node.child[self.Child.SE]

		elif direction == self.Direction.S: # 5
			# If neighbor is located within same parent
			if self.parent.child[self.Child.NW] == self:
				return self.parent.child[self.Child.SW]
			if self.parent.child[self.Child.NE] == self:
				return self.parent.child[self.Child.SE]
			# Check parent's neighboring node for a child
			node = self.parent.findLargerEqualNeighbor(self.Direction.S)
			if node is None or node.divided == False:
				return node
			# Return neighboring parent node's child
			if self.parent.child[self.Child.SW] == self:
				return node.child[self.Child.NW]
			else:
				return node.child[self.Child.NE]

		elif direction == self.Direction.W: # 6
			# If neighbor is located within same parent
			if self.parent.child[self.Child.NE] == self:
				return self.parent.child[self.Child.NW]
			if self.parent.child[self.Child.SE] == self:
				return self.parent.child[self.Child.SW]
			# Check parent's neighboring node for a child
			node = self.parent.findLargerEqualNeighbor(self.Direction.W)
			if node is None or node.divided == False:
				return node
			# Return neighboring parent node's child
			if self.parent.child[self.Child.NW] == self:
				return node.child[self.Child.NE]
			else:
				return node.child[self.Child.SE]

		elif direction == self.Direction.E: # 7
			# If neighbor is located within same parent
			if self.parent.child[self.Child.NW] == self:
				return self.parent.child[self.Child.NE]
			if self.parent.child[self.Child.SW] == self:
				return self.parent.child[self.Child.SE]
			# Check parent's neighboring node for a child
			node = self.parent.findLargerEqualNeighbor(self.Direction.E)
			if node is None or node.divided == False:
				return node
			# Return neighboring parent node's child
			if self.parent.child[self.Child.NE] == self:
				return node.child[self.Child.NW]
			else:
				return node.child[self.Child.SW]
		else:
			assert False
			return []

	def findSmallerNeighbor(self, neighbor, direction, depth):
		candidates = [] if neighbor is None else [neighbor]
		neighbors = []

		while len(candidates) > 0:
			if candidates[0].divided == False:
				if candidates[0].depth >= depth and candidates[0].point.payload == False:
					neighbors.append(candidates[0])
			else:
				#if direction == self.Direction.NW or direction == self.Direction.N or direction == self.Direction.W:
				if direction == self.Direction.N or direction == self.Direction.W:
					candidates.append(candidates[0].child[self.Child.SE])
				#if direction == self.Direction.NE or direction == self.Direction.N or direction == self.Direction.E:
				if direction == self.Direction.N or direction == self.Direction.E:
					candidates.append(candidates[0].child[self.Child.SW])
				#if direction == self.Direction.SW or direction == self.Direction.S or direction == self.Direction.W:
				if direction == self.Direction.S or direction == self.Direction.W:
					candidates.append(candidates[0].child[self.Child.NE])
				#if direction == self.Direction.SE or direction == self.Direction.S or direction == self.Direction.E:
				if direction == self.Direction.S or direction == self.Direction.E:
					candidates.append(candidates[0].child[self.Child.NW])
			candidates.remove(candidates[0])
		return neighbors

	# Return a list of QuadMDP that are neighbors to this particular QuadMDP
	# • Neighbors entail that a particular QuadMDP is touching the other QuadMDP by at least a point
	# • Works only for QuadMDP objects that are leaves (does not divide into another QuadMDP)
	def findNeighbors(self, depth:int=0) -> list[QuadMDP]:
		# Find all larger or equal sized nodes
		candidates = []
		directions = []
		for direction in self.Direction:
			neighbor = self.findLargerEqualNeighbor(direction)
			if neighbor != None and neighbor not in candidates:
				candidates.append(neighbor)
				directions.append(direction)
		neighbors = []
		for candidate,direction in zip(candidates,directions):
			neighbors.extend(self.findSmallerNeighbor(candidate,direction,depth))
		return neighbors

	# Returns a QuadMDP that contains the given point within its boundary
	def findContainedQuadMDP(self, point:Point) -> QuadMDP:
		if self.divided:
			for index in self.Child:
				if self.child[index].boundary.contains(point):
					return self.child[index].findContainedQuadMDP(point)
		else:
			return self

	def generateGraph(self, S:list[QuadMDP], depth:int=0):
		edges = defaultdict(list)
		for cState in S:
			N = cState.findNeighbors(depth)
			for nState in N:
				edges[cState.getTuple()].append(nState.getTuple())
		return edges

	# Adapted Optimal Path function that uses hashmaps, but seems like it doesnt change computation time that much
	# The algorithm still searches through a redundant 
	def getOptimalPath(self, S:list[QuadMDP], depth:int, initialState:tuple, endState:tuple):
		pathList = [[initialState]]
		visited = []
		V = [s.getTuple() for s in S]
		E = self.generateGraph(S, depth)
		k = 0
		while pathList and k < 1000000:
			#if k%1000==0:
				#print(k)
			path = pathList.pop(0)
			v1 = path[-1]
			if v1 not in visited:
				for v2 in E[v1]:
					if v2 == endState:
						return path + [v2]
					else:
						pathList.append(path + [v2])
				visited.append(v1)
				k += 1


	def getPathDFS_bcp(self,  S:list[QuadMDP], depth:int, initialState:tuple, endState:tuple):
		pathList = [[initialState]]
		visited = []
		V = [s.getTuple() for s in S]
		E = self.generateGraph(S, depth)
		k = 0
		while pathList and k < 1000000:
			#if k%1000==0:
				#print(k)
			path = pathList.pop(0)
			v1 = path[-1]
			if v1 not in visited:
				for v2 in E[v1]:
					if v2 == endState:
						return path + [v2]
					else:
						new_path = path + [v2]
						pathList = [new_path] + pathList
				visited.append(v1)
				k += 1

		return "no path detected"
		
	def getPathDFSV2(self,S, depth, initialState, endState, path, visited):
		
		new_path = path
		new_visited = visited

		new_path.append(initialState)
		new_visited.add(initialState)

		if initialState == endState:
			return path
		for neighbor in initialState.findNeighbors(depth):
			if neighbor not in visited:
				result = self.getPathDFSV2(S, depth, neighbor, endState, new_path, new_visited)
				if result is not None:
					return result
		new_path.pop()
		return None
	
	def getPathDFS(self, S:list[QuadMDP], depth:int, initialState, endState, path = [], visited = set()):

		new_path = path
		new_visited = visited
		result = self.getPathDFSV2(S, depth, initialState, endState, new_path, new_visited)
		if result == None or result == []:
			return [initialState, initialState]
		else:
			return result



	def getOptimalPath_v2(self, S:list[QuadMDP], depth:int, initialState:tuple, endState:tuple):
		pathList = [[initialState]]
		visited = []
		V = [s.getTuple() for s in S]
		E = self.generateGraph(S, depth)
		if initialState == endState:
			return "Initial State is Goal State"

		while pathList:
			path = pathList.pop(0)
			state = path[-1]
			if state not in visited:
				neighbors = E[state]
				for neighbor in neighbors:
					new_path = list(path)
					new_path.append(neighbor)
					pathList.append(new_path)
					if neighbor == endState:
						return new_path
				visited.append(state)
		
		return "Did not find any path..."


	def get_optimal_path(self, depth, initial_state, end_state):
		# Computes the shortest path to the end state
		path_list = [[initial_state]]
		k = 0
		while k < len(path_list) and k < 10000000:
			path = path_list[k]
			last_state = path[-1]
			neighbors = last_state.findNeighbors(depth)
			for neighbor in neighbors:
				if neighbor == end_state:
					return path + [neighbor]
				else:
					if neighbor not in path and path + [neighbor] not in path_list:
						path_list.append(path + [neighbor])
			k += 1
		
		#print("no path detected")

	def getTuple(self) -> tuple:
		return (self.point.x, self.point.y)

	def projectToState(self,randomPick:bool=True) -> list[tuple] | tuple:
		result = []
		if self.divided == True:
			return []
		if self.getTuple() == (math.floor(self.point.x), math.floor(self.point.y)):
			result.append(self.getTuple())
		else:
			result.append((math.floor(self.point.x), math.floor(self.point.y)))
			result.append((math.ceil(self.point.x), math.floor(self.point.y)))
			result.append((math.floor(self.point.x), math.ceil(self.point.y)))
			result.append((math.ceil(self.point.x), math.ceil(self.point.y)))
		if randomPick:
			return result[random.randint(0,3)]
		else:
			return result

	def draw(self, ax, border:bool=True):
		if self.divided:
			for index in self.Child:
				self.child[index].draw(ax,border)
		else:
			if border:
				self.boundary.draw(ax)
			if self.point.payload is True:
				x = self.point.x
				y = self.point.y
				rx = [x-0.5,x+0.5,x+0.5,x-0.5]
				ry = [y-0.5,y-0.5,y+0.5,y+0.5]
				plt.fill(rx,ry,color='red',zorder=-2)
















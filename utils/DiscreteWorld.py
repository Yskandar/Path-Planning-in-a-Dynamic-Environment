#!/usr/bin/env python

""" DiscreteWorld.py """

__author__ = "Hayato Kato" 

import os
import sys
dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir)

from copy import copy
import numpy as np
import matplotlib.pyplot as plt
from time import time

# Custom Written Parent Class for MDP Systems
from utils import MDP

#todos:
#project global end state to obs end state //yskar
#need to update obs
#need to make global world with moving obstacles (main)
#visualization
#object class

#makes a new discrete world every time
class DiscreteWorld(MDP.MDP):
	def __init__(self, dx, dy, start, end, obs):
		# State Space
		self.S = []
		self.X_dim, self.Y_dim = dx,dy
		for i in range(dx):
			for j in range(dy):
				self.S.append((i,j))
		self.S_start = start #list of tuples
		self.S_end = end #end state
		self.S_obs = obs #obstacle states

		# Action Space (Side + Diagonal + Stay Put)
		self.A = [(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1),(0,1),(0,0)]
		#self.A = [(1,0),(0,-1),(-1,0),(0,1),(0,0)]

		# Error probability
		self.p_e = 0.4

		# Reward for reaching the final destination
		self.R_D = 100

		# Transition Probabilities
		N_S = len(self.S)
		N_A = len(self.A)
		self.P = np.zeros((N_S,N_A,N_S))
		for s in range(N_S):
			for a in range(N_A):
				for s_ in range(N_S):
					self.P[s,a,s_] = self.pr(self.S[s],self.A[a],self.S[s_])

		# Reward Function
		self.R = np.zeros((N_S,N_A,N_S))
		for s in range(N_S):
			if self.S[s] in self.S_end:
				self.R[s,:,:] = self.R_D

		# Horizon
		self.H = -1 # -1 means infinity

		# Discount Gamma
		self.g = 0.9

		super().__init__(self.S,self.A,self.P,self.R,self.H,self.g)

		self.figsize = 4

	# Returns desired state given a certain action
	def desired_state(self, state, action):
		desired_state = (state[0]+action[0],state[1]+action[1])
		if action not in self.A or desired_state not in self.S or desired_state in self.S_obs:
			return None
		return desired_state

	# Transition Probabilities (from Icecream GridWorld) <- Currently working, use this
	def pr(self, state, action, next_state):
		probability = 0
		# States must not be occupied by obstacle and they must be adjacent
		if next_state not in self.S_obs and state not in self.S_obs and next_state in self.S_adj(state):
			# If desired action succeeded
			if next_state == self.desired_state(state, action):
				# If the state changes
				if state == next_state:
					probability = 1
				else:
					probability = 1 - self.p_e
			# If desired action failed
			else:
				# If the state changes
				if state == next_state:
					probability = 1
					# Subtract the probability of all surrounding states
					for s in self.S_adj(state):
						if s != state:
							probability -= self.pr(state, action, s)
				else:
					if action != self.A[-1]:
						probability = self.p_e/(len(self.A)-1)
		return probability

	# Transition Probabilities (proximal action priority) <- Still developing, not working atm
	def new_pr(self, state, action, next_state):
		probability = 0
		if next_state not in self.S_obs and state not in self.S_obs and next_state in self.S_adj(state):
			desired_state = self.desired_state(state, action)
			desired_action = self.desired_action(state, next_state)
			if desired_state != None: # Action doesnt go out of bounds
				near_desired_state = self.S_adj(desired_state)
				if next_state == desired_state:
					if state == next_state:
						probability = 1
					else:
						probability = 4
				elif next_state in near_desired_state:
					if action in [(1,0),(-1,0),(0,1),(0,-1)]: # Sideways
						if desired_action not in [(1,0),(-1,0),(0,1),(0,-1)]:
							probability = 2
						else:
							probability = 1
					elif action in [(1,1),(-1,1),(1,-1),(-1,-1)]: # Diagonal
						probability = 2
				else:
					probability = 1
				probability /= 15
		return probability
	

	def observe_surroundings(self, state, dx, dy):
		"""
		Returns the intermediary graph/MDP object corresponding to the surroundings
		"""
		# Create the MDP object
		surroundings = DiscreteWorld(0, 0, [], [], [])
		
		# Computing the states in the surroundings, and filling the obstacle space
		x_s, y_s = state
		min_x, max_x, min_y, max_y = x_s, x_s, y_s, y_s
		for x in range(x_s-dx, x_s+dx):
			for y in range(y_s-dy, y_s+dy):
				if 0 <= x < self.X_dim and 0 <= y < self.Y_dim:
					surroundings.S.append((x,y))  # add the state
					min_x, max_x = min(x, min_x), max(x, max_x)
					min_y, max_y = min(y, min_y), max(y, max_y)
					if (x, y) in self.S_obs:  # if obstacle, also add it to the obstacle state
						surroundings.S_obs.append((x,y))

		surroundings.X_dim = max_x - min_x +1
		surroundings.Y_dim = max_y - min_y +1

		return surroundings

	def project_goal(self, goal):
		states = np.array([state for state in self.S if state not in self.S_obs])
		distances = np.linalg.norm(np.array(goal) - states, axis = 1)

		return tuple(states[np.argmin(distances)])

	# Plots visual representation of 2D gridworld environment with obstacles
	def plot(self, state, plot=True):
		fig, ax = plt.subplots(1, 1, figsize=(self.X_dim/self.Y_dim*self.figsize, self.figsize))

		# Plot State Space
		sx, sy = zip(*self.S)
		plt.scatter(sx,sy,1,color='k',edgecolor='k',zorder=-1)
		# Start Region
		for x,y in self.S_start:
			rx = [x-0.5,x+0.5,x+0.5,x-0.5]
			ry = [y-0.5,y-0.5,y+0.5,y+0.5]
			plt.fill(rx,ry,color='skyblue',zorder=-2)
		# End Region
		for x,y in self.S_end:
			rx = [x-0.5,x+0.5,x+0.5,x-0.5]
			ry = [y-0.5,y-0.5,y+0.5,y+0.5]
			plt.fill(rx,ry,color='lightcoral',zorder=-2)
		# Obstacles
		sx, sy = zip(*self.S_obs)
		plt.scatter(sx,sy,100,color='k',edgecolor='k',zorder=-1)
		# Agent
		if state in self.S:
			plt.scatter(state[0],state[1],100,color='w',edgecolor='k',zorder=1)
		else:
			sys.exit("Invalid State")

		# Formatting
		ax.set_xlim([-1,self.X_dim])
		ax.set_ylim([-1,self.Y_dim])
		ax.set_aspect('equal')
		plt.tight_layout()
		if plot:
			plt.show()
		else:
			plt.close()
		return fig, ax

	# Plots all transition probabilities
	def plotProbability(self, probabilities, plot=True):	
		fig, ax = plt.subplots(1,1)	
		matrix = probabilities.reshape(self.X_dim,self.Y_dim).transpose()
		for (j,i),label in np.ndenumerate(matrix):
		    ax.text(i,j,round(label,4),ha='center',va='center')
		im = ax.imshow(matrix, origin = 'lower')
		ax.set_xlim([-0.5,self.X_dim-0.5])
		ax.set_ylim([-0.5,self.Y_dim-0.5])
		ax.set_xticks(np.arange(-.5, self.X_dim, 1), minor=True)
		ax.set_yticks(np.arange(-.5, self.Y_dim, 1), minor=True)
		ax.grid(which='minor', color='k', linestyle='-', linewidth=1)
		ax.set_title('p_total = ' + str(round(sum(probabilities),4)))
		plt.colorbar(im)
		if plot:
			plt.show()
		else:
			plt.close()
		return fig, ax

	# Plots optimal value function across all states
	def plotValue(self, plot=True):
		fig, ax = plt.subplots(1,1)
		matrix = self.V.reshape(self.X_dim,self.Y_dim).transpose()
		for (j,i),label in np.ndenumerate(matrix):
		    ax.text(i,j,round(label,4),ha='center',va='center')
		im = ax.imshow(matrix, cmap="RdBu", origin = 'lower')
		ax.set_xlim([-0.5,self.X_dim-0.5])
		ax.set_ylim([-0.5,self.Y_dim-0.5])
		ax.set_xticks(np.arange(-.5, self.X_dim, 1), minor=True)
		ax.set_yticks(np.arange(-.5, self.Y_dim, 1), minor=True)
		ax.grid(which='minor', color='k', linestyle='-', linewidth=1)
		plt.colorbar(im)
		if plot:
			plt.show()
		else:
			plt.close()
		return fig, ax

	# Plots optimal policy across all states
	def plotPolicy(self, plot=True):
		fig, ax = plt.subplots(1,1)
		matrix = self.Pi.reshape(self.X_dim,self.Y_dim).transpose()
		for (j,i),direction in np.ndenumerate(matrix):
			if direction != (0,0):
				plt.quiver(i,j,direction[0],direction[1])
			else:
				plt.scatter(i,j,100,marker='x',color='k')
		for x,y in self.S_obs:
			rx = [x-0.5,x+0.5,x+0.5,x-0.5]
			ry = [y-0.5,y-0.5,y+0.5,y+0.5]
			plt.fill(rx,ry,color='lightgrey',zorder=-2)
		ax.set_xlim([-0.5,self.X_dim-0.5])
		ax.set_ylim([-0.5,self.Y_dim-0.5])
		ax.set_xticks(np.arange(-.5, self.X_dim, 1), minor=True)
		ax.set_yticks(np.arange(-.5, self.Y_dim, 1), minor=True)
		ax.grid(which='minor', color='k', linestyle='-', linewidth=1)
		ax.set_aspect('equal')
		if plot:
			plt.show()
		else:
			plt.close()
		return fig, ax

def main():
	print("Starting...")

	# Defining the grid world by indicating which states are starts, ends, and obstacles, etc.
	start_region = [(0,0)]
	end_region = [(10,10),(10,9),(9,10),(9,9)]
	obstacles = [(7,8),(8,8),(9,8),(10,8),(4,10),(4,9),(4,8),(4,7),(4,6),(4,5),(5,5),(6,5),(7,5),(8,5),(8,4),(8,3),(8,2),(5,2),(5,1),(5,0),(4,2)]
	world = DiscreteWorld(dx=11,dy=11,start=start_region,end=end_region,obs=obstacles)
	
	current_state = (2,2)

	world.plot(current_state)

	# Check validity of transition probabilities
	print(str(round(world.testTransitionProbability(),2)) + '% Accuracy')

	world.plotProbability(world.P[world.S.index(current_state), world.A.index((1,0)),:])
	world.plotValue()
	world.plotPolicy()

	print("Ending...")


if __name__ == '__main__':
	main()

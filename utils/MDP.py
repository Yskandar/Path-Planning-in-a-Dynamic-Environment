#!/usr/bin/env python

""" MDP.py """

__author__ = "Hayato Kato"

from copy import copy
import math
import numpy as np
import matplotlib.pyplot as plt

class MDP(object):
	def __init__(self, S, A, P, R, H, g):
		self.S = S
		self.A = A
		self.P = P
		self.R = R
		self.H = H
		self.gamma = g

		N_S = len(self.S)
		N_A = len(self.A)
		self.V = np.zeros(N_S)
		self.Q = np.zeros((N_S,N_A))
		self.threshold = 0.0001

		# Compute Optimal Value using Policy Iteration
		self.policyIteration()

		temp = []
		for s in range(N_S):
			temp.append(self.policy(self.S[s]))
		self.Pi = np.empty(len(temp),dtype=object)
		self.Pi[:] = temp

	# Value Iteration
	def valueIteration(self):
		N_S = len(self.S)
		N_A = len(self.A)
		self.V = np.zeros(N_S)
		self.Q = np.zeros((N_S,N_A))
		new_V = np.zeros(N_S)
		diff = []
		k = 0
		# Iterate until convergence
		while True:
			# Iterate through all transitions
			for s in range(N_S):
				max_V = 0
				for a in range(N_A):
					q_val = 0
					for s_ in range(N_S):
						# Compute the max value
						q_val += self.P[s,a,s_]*(self.R[s,a,s_]+self.gamma*self.V[s_])
					self.Q[s,a] = q_val
				new_V = np.amax(self.Q, axis=-1)
			temp = np.linalg.norm(self.V - new_V)
			diff.append(temp)
			if temp < self.threshold:
				break
			self.V = copy(new_V)
			if k == self.H:
				break
			k += 1
		return diff, k

	# Policy Iteration
	def policyIteration(self):
		N_S = len(self.S)
		N_A = len(self.A)
		self.V = np.zeros(N_S)
		self.Pi = np.zeros(N_S)
		self.Q = np.zeros((N_S,N_A))
		new_V = np.zeros(N_S)
		diff = []
		k = 0
		# Iterate until convergence
		while True:
			for s in range(N_S):
				a = int(self.Pi[s])
				q_val = 0
				for s_ in range(N_S):
					# Compute both the max value and optimal policy
					q_val += self.P[s,a,s_]*(self.R[s,a,s_]+self.gamma*self.V[s_])
				self.Q[s,a] = q_val
			self.Pi = np.argmax(self.Q,axis=-1)
			new_V = np.amax(self.Q, axis=-1)
			temp = np.linalg.norm(self.V - new_V)
			diff.append(temp)
			if temp < self.threshold:
				break
			self.V = copy(new_V)
			k += 1
			if k == self.H:
				break
		return diff,k

	# Returns desired state given a certain action (Implement in child class)
	def desired_state(self, state, action):
		return None

	# Returns ideal action that would result in next_state given state
	def desired_action(self, state, next_state):
		for action in self.A:
			if next_state == self.desired_state(state,action):
				return action
		return None

	# Returns set of all states reachable by a single action
	def S_adj(self, state):
		space = [] 
		for action in self.A:
			new_state = self.desired_state(state,action)
			if new_state != None:
				space.append(new_state)
		return space

	# Return optimal policy given state based off optimal value
	def policy(self, state):
		if state in self.S_obs:
			return (0,0)
		max_V = -math.inf
		desired_state = state
		for s in self.S_adj(state):
			state_index = self.S.index(s)
			value = self.V[state_index]
			if max_V < value:
				max_V = value
				desired_state = self.S[state_index]
		return self.desired_action(state, desired_state)

	# Returns the percentage of valid transition probabilities (must add up to either 0 or 1)
	def testTransitionProbability(self):
		N_S = len(self.S)
		N_A = len(self.A)
		testCount = N_S*N_A
		correctCount = 0
		for s in range(N_S):
			for a in range(N_A):
				total_p = round(sum(self.P[s,a,:]),4)
				if total_p == 1 or total_p == 0:
					correctCount += 1
		return correctCount / testCount * 100



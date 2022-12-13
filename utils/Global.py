#!/usr/bin/env python

__author__ = "david kao"

import sys
import numpy as np
from utils import move
import matplotlib.pyplot as plt
from utils.Obstacle import Obstacle, Agent

#todo: add obstacles from image

#global map that will track all agents
#size: a tuple with the dimensions
#obstacles: list of all the obstacles in the world
#time: this class keeps track of the simulation's time
class Global(object):
    def __init__(self, size):
        self.x_dim, self.y_dim = size[0], size[1]
        self.grid = np.zeros(size)
        self.obstacles = []
        self.agents = []
        self.time = 0
        self.S = self.allStates()
        self.complete_path = []
    
    def allStates(self):
        S = []
        for i in range(self.x_dim):
            for j in range(self.y_dim):
                S.append((i,j))
        return S

    def createAgent(self, location):
        self.grid[self.xy(location)] = 2
        a = Agent(location)
        self.agents.append(a)
        return self.agents[-1]
    
    def createObstacle(self, movement, location):
        self.grid[self.xy(location)] = 1
        o = Obstacle(movement, location)
        self.obstacles.append(o)
        return self.obstacles[-1]
        
    def moveObstacle(self, o: Obstacle):
        next_move = o.next(self.time-1) #next_move is the actual movement, e.g. (-1, 0)
        current_location = o.location

        if self.is_in_statespace(current_location+next_move) and self.grid[self.xy(current_location+next_move)] != 2.0:
            self.grid[self.xy(current_location)] = 0
            self.grid[self.xy(current_location+next_move)] = 1
            o.location = current_location + next_move
        else:
            print("Obstacle trying to leave the world or bumping into agent, staying still")
        return o

    def moveAgent(self, a: Agent):
        next_move = a.next()
        current_location = a.location
        self.grid[self.xy(current_location)] = 0
        self.grid[self.xy(current_location+next_move)] = 2
        a.location = current_location + next_move
        return a

    def update_grid(self, previous_location, new_location):
        self.grid[self.xy(previous_location)] = 0
        self.grid[self.xy(new_location)] = 2
        print("grid updated")

    def is_in_statespace(self, location):
        x, y = location
        return (0 <= x < self.x_dim) and (0 <= y < self.y_dim)

    def next(self):
        self.time += 1
        agent_location = ()
        for a in self.agents:
            self.moveAgent(a)
            agent_location = a.location
        for o in self.obstacles:
            self.moveObstacle(o)
            if (o.location[0] == agent_location[0]) and (o.location[1] == agent_location[1]):
                print("obstacle has collided with agent")
        if np.sum(self.grid) != len(self.obstacles) + 2*len(self.agents):
            print("careful, obstacles just collided")
        return self.time

    def xy(self,arr): #numpy arrays need to be in this form in order to index
        return ((arr[0]),(arr[1])) #tuple of : tuple of x coordinates, tuple of y coordinates

    def observe(self): #i wrote this assuming there's only one agent
        a = self.agents[0]
        range = a.range
        ax, ay = a.location
        ax_min = max(0, ax-range)
        ax_max = min(self.x_dim-1, ax+range)
        ay_min = max(0, ay-range)
        ay_max = min(self.y_dim-1, ay+range)
        return self.grid[ax_min:ax_max+1, ay_min:ay_max+1]

    def plot(self):
        figsize = 5
        fig, ax = plt.subplots(1, 1, figsize=(self.x_dim/self.y_dim*figsize, figsize))

		# Plot State Space
        sx, sy = zip(*self.S)
        #plt.scatter(sx,sy,1,color='k',edgecolor='k',zorder=-1)

#warning: the graph axes and the np.array axes are different
#so some magic is needed, TRUST THE GRAPH AND NOT THE NP ARRAY
        for o in self.obstacles: 
            x,y = o.location #lol just don't ask
            rx = [x-0.5,x+0.5,x+0.5,x-0.5]
            ry = [y-0.5,y-0.5,y+0.5,y+0.5]
            plt.fill(rx,ry,color='red',zorder=-2, alpha=0.8)
        for a in self.agents:
            x,y = a.location
            rx = [x-0.5,x+0.5,x+0.5,x-0.5]
            ry = [y-0.5,y-0.5,y+0.5,y+0.5]
            plt.fill(rx,ry,color='skyblue',zorder=-2, alpha=0.8)

            ax_min = max(0, x-a.range)
            ax_max = min(self.x_dim-1, x+a.range)
            ay_min = max(0, y-a.range)
            ay_max = min(self.y_dim-1, y+a.range)
            rx = [ax_min-0.5, ax_max+0.5, ax_max+0.5, ax_min-0.5]
            ry = [ay_min-0.5, ay_min-0.5, ay_max+0.5, ay_max+0.5]
            plt.fill(rx, ry, color='lightgreen', zorder=-2, alpha=0.4)
		# Formatting
        ax.set_xlim([-0.5,self.x_dim-0.5])
        ax.set_ylim([-0.5,self.y_dim-0.5])
        ax.set_aspect('equal')
        ax.invert_yaxis()
        plt.title("time = " + str(self.time))
        plt.tight_layout()
        plt.show()
        return fig, ax

    def plot_agent(self, i, ax):
        
        a = self.agents[0]
        x,y = self.complete_path[i]
        rx = [x-0.5,x+0.5,x+0.5,x-0.5]
        ry = [y-0.5,y-0.5,y+0.5,y+0.5]
        plt.fill(rx,ry,color='skyblue',zorder=-2, alpha=0.8)
        ax_min = max(0, x-a.range)
        ax_max = min(self.x_dim-1, x+a.range)
        ay_min = max(0, y-a.range)
        ay_max = min(self.y_dim-1, y+a.range)
        rx = [ax_min-0.5, ax_max+0.5, ax_max+0.5, ax_min-0.5]
        ry = [ay_min-0.5, ay_min-0.5, ay_max+0.5, ay_max+0.5]
        plt.fill(rx, ry, color='lightgreen', zorder=-2, alpha=0.4)
        ax.set_xlim([-0.5,self.x_dim-0.5])
        ax.set_ylim([-0.5,self.y_dim-0.5])
        ax.set_aspect('equal')
        plt.title("time = " + str(self.time))
        plt.tight_layout()


def main():
    g = Global((10,10))
    g.createObstacle(move.LR, (3,5))
    g.createObstacle(move.UD, (1,2))
    agent = g.createAgent((2,1))
    agent.range = 2
    print(g.grid,'\n')
    g.plot()
    duration = 3
    for t in range(duration):
        print(g.observe())
        g.next()
        g.plot()

if __name__ == '__main__':
    main()
    


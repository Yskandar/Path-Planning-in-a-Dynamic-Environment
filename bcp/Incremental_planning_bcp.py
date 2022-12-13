
__author__ = "Yska"

import sys
import argparse
import numpy as np
import math
import move
import matplotlib.pyplot as plt
from QuadMDP.QuadTree import Point, Rect
from QuadMDP.QuadMDP import QuadMDP
from Obstacle import Obstacle, Agent
from Global import Global
from agent_graph import *
from matplotlib.animation import FuncAnimation


def main(args):

    # Initializing the world from the picture

    filename = args.input
    image = plt.imread('map/' + filename)
    mapWidth, mapHeight = image.shape
    print("Map Width:", mapWidth)
    print("Map height:", mapHeight)
    mapDepth = (int)(math.log(mapWidth,2))
    searchDepth = 2 # Search depth within QuadMDP

    # Global class for the whole environment
    g = Global((mapWidth,mapHeight))

    # QuadMDP class for generating the quad decomposed states
    quad = QuadMDP(Rect(mapWidth/2-0.5,mapHeight/2-0.5,mapWidth,mapHeight),mapDepth)

    # Load all obstacles
    for x in range(mapWidth):
        for y in range(mapHeight):
            if image[y,x] == 0:
                g.createObstacle(move.NONE,(x,y))
                quad.insert(Point(x,y,True))

    # Generate quad decomposed states
    S = quad.findEmptySpace(searchDepth)
    graph = quad.generateGraph(S, searchDepth)

    startPos = (8,8)
    goalPos = (127,0)
    startQuadMDP = quad.findContainedQuadMDP(startPos)
    goalQuadMDP = quad.findContainedQuadMDP(goalPos)
    path = quad.getOptimalPath(S,searchDepth,startQuadMDP.getTuple(),goalQuadMDP.getTuple())
    path.append(goalPos)

    # Create the agent
    agent = g.createAgent(startPos)
    agent.range = 3
    print(g.grid,'\n')
    print("Initial position")
    g.plot()

    path = np.floor(path)
    print("Path we want to take: ", path)

    Xlim, Ylim = mapWidth, mapHeight
    obs = g.observe()
    # build observation environment
    Graph = agent_graph(agent, obs, Xlim, Ylim)
    idx_goal = 0

    complete_path = []
    current_state = tuple(agent.location)

    # Plot Quad MDP Decomposition
    figsize = 5
    fig, ax = plt.subplots(1, 1, figsize=(mapWidth/mapHeight*figsize, figsize))
    quad.draw(ax,True)
    startQuadMDP.draw(ax)
    goalQuadMDP.draw(ax)
    ax.set_xlim([-0.5,mapWidth-0.5])
    ax.set_ylim([-0.5,mapHeight-0.5])
    ax.set_aspect('equal')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()

    # Plot Projected High-level Path
    figsize = 5
    fig, ax = plt.subplots(1, 1, figsize=(mapWidth/mapHeight*figsize, figsize))
    quad.draw(ax,True)
    startQuadMDP.draw(ax)
    goalQuadMDP.draw(ax)
    ax.plot(*zip(*path),'--o')
    ax.set_xlim([-0.5,mapWidth-0.5])
    ax.set_ylim([-0.5,mapHeight-0.5])
    ax.set_aspect('equal')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()

    while idx_goal < len(path):

        # Grab the current goal and project it to the surroundings
        goal = tuple(path[idx_goal])
        print("goal: ", goal)
        projected_goal = Graph.project_to_surroundings(goal)

        # Get the optimal path on the low level
        inter_path = Graph.get_optimal_path(current_state, projected_goal)
        complete_path += inter_path

        # Simulate that we follow that path and get to the intermediary node
        current_state = inter_path[-1]
        print(current_state)
        print(goal)

        # Update the world accordingly
        g.update_grid(tuple(agent.location), current_state) # I created that function because I was too lazy to implement all the sequences of actions. It justs update the grid with the current state of the agent
        agent.location = current_state
        
        # Check how the environment look now & get new observation
        #g.plot()
        obs = g.observe()

        # Load the new graph corresponding to the new positions of the agent and its new observation
        Graph = agent_graph(agent, obs, Xlim, Ylim)
        if current_state == goal: # If we have reached the goal, we go to the next one
            idx_goal += 1

    # Plotting the trajectory
    g.complete_path = complete_path

    figsize = 5
    fig, ax = plt.subplots(1, 1, figsize=(mapWidth/mapHeight*figsize, figsize))
    plt.title("time = " + str(g.time))
    ax.set_xlim([-0.5,mapWidth-0.5])
    ax.set_ylim([-0.5,mapHeight-0.5])
    ax.set_aspect('equal')
    ax.invert_yaxis()
    plt.tight_layout()

    def plot_agent(i):
        a = g.agents[0]
        x,y = g.complete_path[i]
        rx = [x-0.5,x+0.5,x+0.5,x-0.5]
        ry = [y-0.5,y-0.5,y+0.5,y+0.5]
        plt.fill(rx,ry,color='skyblue',zorder=-2, alpha=0.8)
        ax_min = max(0, x-a.range)
        ax_max = min(g.x_dim-1, x+a.range)
        ay_min = max(0, y-a.range)
        ay_max = min(g.y_dim-1, y+a.range)
        rx = [ax_min-0.5, ax_max+0.5, ax_max+0.5, ax_min-0.5]
        ry = [ay_min-0.5, ay_min-0.5, ay_max+0.5, ay_max+0.5]
        plt.fill(rx, ry, color='lightgreen', zorder=-2, alpha=0.4)

    for o in g.obstacles:
        x,y = o.location
        rx = [x-0.5,x+0.5,x+0.5,x-0.5]
        ry = [y-0.5,y-0.5,y+0.5,y+0.5]
        plt.fill(rx,ry,color='red',zorder=-2, alpha=0.8)
    
    for state in path:
        x, y = state
        rx = [x-0.5,x+0.5,x+0.5,x-0.5]
        ry = [y-0.5,y-0.5,y+0.5,y+0.5]
        plt.fill(rx,ry,color='blue',zorder=-2, alpha=1)

    
    ani = FuncAnimation(fig, plot_agent, frames=len(complete_path), interval=10, repeat=False)

    plt.show()
    


    print(complete_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Quad Decomposition Demo')
    parser.add_argument('-i', '--input', help='Grayscale PNG map file name', default='worldmap.png')
    args = parser.parse_args()
    main(args)
    
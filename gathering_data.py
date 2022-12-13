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
from utils.graph_utils import Graph
from utils.RandomMap import RandomMap
import pandas as pd




def main(args):
    print("starting...")

    dim = 64
    f = open('data.csv','w')
    writer = csv.writer(f)
    header = ['Threshold', 'Resolution', 'Obstacle %', 'Base V #', 'Base E #']
    for searchDepth in range((int)(math.log(dim,2))):
        header.append('D(' + str(searchDepth) + ') V #')
        header.append('D(' + str(searchDepth) + ') E #')
    writer.writerow(header)

    # Setup Figure
    fig = plt.figure()
    ax = plt.subplot()

    searchDepth = 0
    num_paths = 1
    num_iter = 1000
    thresholds = np.linspace(0.5, 0.7, 6)

    # Generating Dictionnary
    dict_times = dict()
    for threshold in thresholds:

        dict_times['Vanilla path planning - threshold {}'.format(threshold)] = []
        dict_times['QuadTree path planning - threshold {}'.format(threshold)] = []
    Nonecount = 0

    k = 0
    tot = num_iter * num_paths * len(thresholds)

    # Generating random maps
    for threshold in thresholds:
        for iter in range(num_iter):
            resolution = random.uniform(5,15)
            thresh = round(threshold,2)
            randImage = RandomMap((dim,dim),thresh,resolution)
            image = randImage.image
            width,height = randImage.xDim, randImage.yDim
            
            
            obsCount = 0
            non_obstacles = []  # stores the points that are not obstacles so that we can select start state and end goal in here
            obstacles = []

            """
            if iter % 250 == 0:
                fig, ax = plt.subplots(1, 1, figsize=(5, 5))
                randImage.plot(ax)
                plt.show()
            """

            # compute the time to build the quadtree
            start = time.time()
            depth = (int)(math.log(width,2))
            quad_decomposed = QuadMDP(Rect(width/2-0.5,height/2-0.5,width,height),depth)
            for x in range(width):
                for y in range(height):
                    if image[x,y] == 0:
                        #quad_original.insert(Point(x,y,True))
                        quad_decomposed.insert(Point(x,y,True))
            S = quad_decomposed.findEmptySpace(searchDepth)
            delta = time.time() - start

            for x in range(width):
                for y in range(height):
                    if image[x,y] == 0:
                        obsCount += 1
                        obstacles.append((x,y))
                    else:
                        non_obstacles.append((x,y))
            
            
            time_original = 0
            time_decomposed = delta  # take into account the time to pay because of the quadtree building, set to 0 if you just want to compare the path planning only
            for iternum in range(num_paths):

                print('Advancement: ', 100*k/tot, '%')

                if non_obstacles == []:
                    print('skipping')
                    continue

                # Pick the start state and end state we want to follow
                startPos_idx, goalPos_idx = np.random.choice(len(non_obstacles), 2, replace = False)
                startPos = non_obstacles[startPos_idx]
                goalPos = non_obstacles[goalPos_idx]

                

                # Low level MDP (graph)
                start = time.time()
                graph = Graph(dim, dim, obstacles)
                path = graph.get_optimal_path(startPos, goalPos)
                if path is not None:
                    time_original += (time.time() - start)
                else:
                    k += 1
                    continue
                print(path)
                

                # High level MDP

                start = time.time()
                startQuadMDP = quad_decomposed.findContainedQuadMDP(startPos)
                goalQuadMDP = quad_decomposed.findContainedQuadMDP(goalPos)
                path = quad_decomposed.getOptimalPath(S,searchDepth,startQuadMDP.getTuple(),goalQuadMDP.getTuple())
                if path is not None:
                    time_decomposed += (time.time() - start)
                else:
                    time_original = 0
                    k+=1
                    continue
                
                print(path)



                
                if path == None:
                    Nonecount += 1

                k += 1
            
            # Add results
            dict_times['Vanilla path planning - threshold {}'.format(threshold)].append(time_original)
            dict_times['QuadTree path planning - threshold {}'.format(threshold)].append(time_decomposed)




    print("None paths count:", Nonecount)

    df = pd.DataFrame.from_dict(dict_times)
    df.to_csv("GraphData/comparing_times_with_env_building_{}.csv".format(time.time()))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Quad Decomposition Demo')
    parser.add_argument('-i', '--input', help='Grayscale PNG map file name', default='1.png')
    args = parser.parse_args()
    main(args)


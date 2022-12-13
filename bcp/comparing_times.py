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
    print("starting...")

    dim = 32
    f = open('data.csv','w')
    writer = csv.writer(f)
    header = ['Threshold','Resolution', 'Obstacle %','Base V #', 'Base E #']
    for searchDepth in range((int)(math.log(dim,2))):
        header.append('D(' + str(searchDepth) + ') V #')
        header.append('D(' + str(searchDepth) + ') E #')
    writer.writerow(header)

    # Setup Figure
    fig = plt.figure()
    ax = plt.subplot()

    searchDepth = 0
    num_paths = 1

    dict_times = dict()
    dict_times['original'] = []
    dict_times['decomposed'] = []
    Nonecount = 0

    k = 0
    tot = len(np.linspace(0.3, 0.7, 5)) * 500 * num_paths

    # Generating random maps
    for thresh in np.linspace(0.3, 0.7, 5):
        for iter in range(500):
            resolution = random.uniform(5,15)
            thresh = round(thresh,2)
            randImage = RandomMap((dim,dim),thresh,resolution)
            image = randImage.image
            width,height = randImage.xDim, randImage.yDim
            depth = (int)(math.log(width,2))
            quad_original = QuadMDP(Rect(width/2-0.5,height/2-0.5,width,height),depth)
            quad_decomposed = QuadMDP(Rect(width/2-0.5,height/2-0.5,width,height),depth)
            obsCount = 0
            non_obstacles = []  # stores the points that are not obstacles so that we can select start state and end goal in here
            obstacles = []

            if iter % 250 == 0:
                fig, ax = plt.subplots(1, 1, figsize=(5, 5))
                randImage.plot(ax)
                plt.show()

            for x in range(width):
                for y in range(height):
                    if image[x,y] == 0:
                        obsCount += 1
                        quad_original.insert(Point(x,y,True))
                        quad_decomposed.insert(Point(x,y,True))
                        obstacles.append((x,y))
                    else:
                        quad_original.insert(Point(x,y,False))
                        #quad_decomposed.insert(Point(x,y,False))
                        non_obstacles.append((x,y))
            
            for iternum in range(num_paths):

                print('Advancement: ', 100*k/tot, '%')

                if non_obstacles == []:
                    continue

                # Pick the start state and end state we want to follow
                startPos_idx, goalPos_idx = np.random.choice(len(non_obstacles), 2, replace = False)
                startPos = non_obstacles[startPos_idx]
                goalPos = non_obstacles[goalPos_idx]

                

                # Low level MDP
                start = time.time()
                S = quad_original.findEmptySpace(0)
                startQuadMDP = quad_original.findContainedQuadMDP(startPos)
                goalQuadMDP = quad_original.findContainedQuadMDP(goalPos)
                path = quad_original.getOptimalPath(S,0,startQuadMDP.getTuple(),goalQuadMDP.getTuple())
                time_original = time.time() - start
                #print(path)



                # Low level MDP (graph)
                start = time.time()
                graph = Graph(dim, dim, obstacles)
                path = Graph.get_optimal_path(startPos, goalPos)
                time_original_graph = time.time() - start
                

                # High level MDP
                start = time.time()
                S = quad_decomposed.findEmptySpace(searchDepth)
                startQuadMDP = quad_decomposed.findContainedQuadMDP(startPos)
                goalQuadMDP = quad_decomposed.findContainedQuadMDP(goalPos)
                path = quad_decomposed.getOptimalPath(S,searchDepth,startQuadMDP.getTuple(),goalQuadMDP.getTuple())
                time_decomposed = time.time() - start
                print(path)

                # Add results
                dict_times['original'].append(time_original)
                dict_times['decomposed'].append(time_decomposed)

                
                if path == None:
                    Nonecount += 1

                k += 1




    print("None paths count:", Nonecount)

    plt.figure()
    X = np.linspace(0, len(dict_times['original']), len(dict_times['original']))
    plt.plot(X, dict_times['original'], label = 'No decomposition')
    plt.plot(X, dict_times['decomposed'], label = 'Decomposition')
    plt.legend()
    plt.title("Comparing Vanilla Path Search and Quadtree Path Search")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Quad Decomposition Demo')
    parser.add_argument('-i', '--input', help='Grayscale PNG map file name', default='1.png')
    args = parser.parse_args()
    main(args)






            


            
            
            


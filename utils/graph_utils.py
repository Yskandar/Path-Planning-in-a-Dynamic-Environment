import numpy as np
from . MDP import *
from utils import MDP, DiscreteWorld, move

class Graph(DiscreteWorld.DiscreteWorld):

    def __init__(self, dx, dy, obs):
        #super().__init__(dx, dy, start, end, obs)
        self.edges = []
        self.obstacle_matrix = np.zeros(shape = (dx, dy))
        self.actions_array = RANDOM
        self.X_dim = dx
        self.Y_dim = dy
        self.S_obs = obs
        for obstacle in self.S_obs:  # building an obstacle matrix for efficient checking of obstacle collision
            x, y = obstacle
            self.obstacle_matrix[x][y] = 1
        
    
    def add_state(self, coords):
        self.S.append(coords)

    def add_edges(self, list_of_edges):
        self.edges += list_of_edges
    
    def add_edge(self, s1, s2):
        self.edges.append((s1, s2))

    def is_in_statespace(self, state):
        return (0 <= state[0] < self.X_dim) and (0 <= state[1] < self.Y_dim)

    def is_in_obstacles(self, state):
        return bool(self.obstacle_matrix[state[0], state[1]])


    def get_adjacent_states(self, state):
        next_states = np.array(state) + self.actions_array
        #adjacent_states = [list(state) for state in next_states]
        adjacent_states = [tuple(state) for state in next_states if self.is_in_statespace(state) and not self.is_in_obstacles(state)]
        return adjacent_states



    def get_edges(self, state):  # get the edges leaving that node - Actually this is probably useless

        adj_states = self.get_adjacent_states(state)

        r = [(state, tuple(adj_state)) for adj_state in adj_states]  # one edge = (node1, node2)

        return r
    
    
    def get_optimal_path(self, start_state, end_state):
        path_list = [[start_state]]
        k = 0
        visited = []
        while path_list and k < 1000000:
            path = path_list.pop(0)
            last_node = path[-1]
            if last_node not in visited:
                neighbors = self.get_adjacent_states(last_node)
                for neighbor in neighbors:
                    if neighbor == end_state:
                        return path + [neighbor]
                    else:
                        path_list.append(path + [neighbor])
                visited.append(last_node)
                k += 1
        
        #print("no path detected")
        return None


def main():
    print("Starting Path Planning with obstacle avoidance")
    # Defining the grid world by indicating which states are starts, ends, and obstacles, etc.
    start_region = [(0,0)]
    end_region = [(5,5)]
    obstacles = [(0,1), (1,1), (2,1), (3,1), (4,1), (5,3), (4,3), (3,3), (2,3),(1,3)]
    world = Graph(dx=6,dy=6,start=start_region,end=end_region,obs=obstacles)

    initial_state = (0,0)
    goal_state = (5,5)

    surroundings = world.observe_surroundings(initial_state, 4, 4)
    surroundings.S_start.append(initial_state)

    projected_goal = surroundings.project_goal(goal_state)
    surroundings.S_end.append(projected_goal)
    
    print(surroundings.X_dim)
    print(surroundings.Y_dim)
    surroundings.plot(initial_state)
    world.plot(initial_state)
    path = world.get_optimal_path(initial_state, goal_state)

    for state in path:
        world.plot(state)
    
    print("Ending...")
    


if __name__ == '__main__':
    main()


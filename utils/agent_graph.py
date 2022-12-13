# create a Agentclass for the agent to build a graph of its surroundings
# 
# 
# Inputs: the agent, the observation
# Outputs: the graph or the optimal path, we'll see
# 
import sys
import numpy as np
from utils import move
import matplotlib.pyplot as plt
from utils.Obstacle import Obstacle, Agent
import copy


class agent_graph():
    def __init__(self, agent, observation, Xlim, Ylim):
        self.agent = agent
        self.observation = np.copy(observation)
        self.observation_expanded = np.copy(observation)
        self.S = []
        self.Xlim = Xlim
        self.Ylim = Ylim
        self.Xlim_up = min(Xlim-1, self.agent.location[0] + self.agent.range)
        self.Ylim_up = min(Ylim-1, self.agent.location[1] + self.agent.range)
        self.Xlim_down = max(0, self.agent.location[0] - self.agent.range)
        self.Ylim_down = max(0, self.agent.location[1] - self.agent.range)

        self.create_states()

    def is_in_statespace(self, state):
        x, y = state
        return (self.Xlim_down <= x <= self.Xlim_up) and (self.Ylim_down <= y <= self.Ylim_up)

    def project_to_surroundings(self, goal):

        x, y = goal
        # First test if the state is actually in the global state space
        if not (0 <= x < self.Xlim and 0 <= y < self.Ylim):
            print("state not in the global environment")
            raise ValueError

        if self.is_in_statespace(goal) and not self.is_in_obstacles(goal):
            return goal

        else:
            states = np.array([state for state in self.S if not self.is_in_obstacles_expanded(state)])
            distances = np.linalg.norm(np.array(goal) - states, axis = 1)

            return tuple(states[np.argmin(distances)])

    def create_states(self):
        x_a, y_a = self.agent.location
        dx, dy = self.agent.range, self.agent.range
        
        for x in range(-dx, dx + 1):
            for y in range(-dy, dy + 1):
                if self.is_in_statespace((x_a + x, y_a + y)):
                    self.S.append((x_a + x, y_a + y))
                    if self.is_in_obstacles((x_a + x, y_a + y)):
                        x_obs, y_obs = int(x_a + x - self.Xlim_down), int(y_a + y - self.Ylim_down)
                        try:
                            self.observation_expanded[x_obs + 1, y_obs] = 1.0
                        except:
                            pass
                        try:
                            self.observation_expanded[x_obs, y_obs + 1] = 1.0
                        except:
                            pass
                        if x_obs - 1 >= 0:
                            try:
                                self.observation_expanded[x_obs - 1, y_obs] = 1.0
                            except:
                                pass
                        if y_obs - 1 >= 0:
                            try:
                                self.observation_expanded[x_obs, y_obs - 1] = 1.0
                            except:
                                pass

    
    def is_in_obstacles(self, state):
        x, y = state

        x_obs, y_obs = int(x - self.Xlim_down), int(y - self.Ylim_down)
        return self.observation[x_obs, y_obs] == 1.0

    def is_in_obstacles_expanded(self, state):
        x, y = state
        x_obs, y_obs = int(x - self.Xlim_down), int(y - self.Ylim_down)
    
        return self.observation_expanded[x_obs, y_obs] == 1.0

    def get_adjacent_states(self, state):
        next_states = np.array(state) + move.RANDOM
        #adjacent_states = [list(state) for state in next_states]
        adjacent_states = [tuple(state) for state in next_states if self.is_in_statespace(state) and not self.is_in_obstacles_expanded(state)]
        return adjacent_states

    def get_optimal_path(self, start_state, end_state):
        path_list = [[start_state]]
        k = 0
        if self.is_in_obstacles_expanded(start_state) or self.is_in_obstacles_expanded(end_state):
            print("start state or end state in obstacles")
            print(" Don't move")
            return [start_state, start_state]

        visited = []
        while path_list and k < 1000000:
            path = path_list.pop(0)
            
            last_node = path[-1]
            if last_node not in visited:
                neighbors = self.get_adjacent_states(last_node)
                np.random.shuffle(neighbors)
                for neighbor in neighbors:
                    if neighbor == end_state:
                        return path + [neighbor]
                    else:
                        path_list.append(path + [neighbor])
                visited.append(last_node)
                k += 1
        
        print("no path detected")
        return [start_state, start_state]


    def get_path_DFS_bcp(self, start_state, end_state):
        path_list = [[start_state]]
        k = 0

        if self.is_in_obstacles_expanded(start_state) or self.is_in_obstacles_expanded(end_state):
            print("start state or end state in obstacles")
            print(" Don't move")
            return [start_state, start_state]
        
        visited = []
        while path_list and k < 1000000:
            path = path_list.pop(0)
            node = path[-1]
            if node not in visited:
                neighbors = self.get_adjacent_states(node)
                for neighbor in neighbors:
                    if neighbor == end_state:
                        return path + [neighbor]
                    else:
                        new_path = path + [neighbor]
                        path_list = [new_path] + path_list
                    visited.append(node)
                    k += 1
        
        print("no path detected")
        return [start_state, start_state]

    def get_path_DFSV2(self, start_state, end_state, path = [], visited = set()):

        new_path = path
        new_visited = visited

        new_path.append(start_state)
        new_visited.add(start_state)
        if start_state == end_state:
            return path
        for neighbor in self.get_adjacent_states(start_state):
            if neighbor not in visited:
                result = self.get_path_DFSV2(neighbor, end_state, new_path, new_visited)
                if result is not None:
                    return result
        new_path.pop()
        return None

    def get_path_DFS(self, start_state, end_state, path = [], visited = set()):
        if self.is_in_obstacles_expanded(start_state) or self.is_in_obstacles_expanded(end_state):
            print("start state or end state in obstacles")
            print(" Don't move")
            return [start_state, start_state]
        new_path = path
        new_visited = visited
        result = self.get_path_DFSV2(start_state, end_state, new_path, new_visited)

        if result == None or result == []:
            return [start_state, start_state]
        else:
            return result


        


    





    


    





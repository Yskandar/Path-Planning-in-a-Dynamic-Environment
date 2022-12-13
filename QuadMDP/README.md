## Quadtree Decomposition
- Reduces number of total states needed to represent the gridworld environment (reduces computation cost, which is highly valued for path planning of dynamic environments)

## Problem:
1. Full knowledge of static world before first step
	- entire quadtree would be built before starting
	- a graph is generated based on the adjacent states in the quadtree
	- the generated graph could ignore lower-level nodes which contain states with obstacles and in general would reduce computation if a path is found for a more sparse representation of the world (possibly going through each level of tree to find a path in the most sparse representation of the world)
	- this graph is used to run basic graph-based search to find a valid path (DFS, BFS, A-star, etc.)
2. No prior knowledge of static world before first step
	- must map out the obstacles as it goes
	- the quadtree would be built as the agent navigates the environment, but since the obstacles themselves are static, no relocation of branches are necessary but merely adding new branches and leaves
	- a graph needs to be generated each time so that the agent can still navigate the incomplete map of the world
	- a clever method of removing and adding new edges based off new entries into the quadtree could reduce computation instead of entire graph being remade each iteration

Both problems can be solved while benefiting from the pros of using a Quadtree data structure to hold onto the valid state spaces that the agent can traverse by consolidating empty spaces together and representing them as a large safe state that the agent can go to.

## Issues:
- Difficult to come up with the transition probabilities
	• There is a intersects() function as part of the Rect class, which can be used to find adjacent rectangles
	• If we consider a deterministic environment with no movement error the system becomes considerably simple since it would only move to a desired state
- Actual computation cost cut might be neglegible/not even worth it due to bad implementation
- Must distinguish between static and dynamic obstacles in the environment when mapping so that the quadtree structure does not get messed up (could look into smart quadtree implementations which are robust to moving particles)


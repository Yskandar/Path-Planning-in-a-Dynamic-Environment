
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("collected_data1000.csv")

obsPercentage = df['Obstacle %']
baseVertex = df['Base V #']
d0Vertex = df['D(0) V #']

plt.rcParams['text.usetex'] = True

x = np.arange(101)
#y1 = 6*np.power(x,0.59)
#y2 = 0.8*np.power(x,0.8)
y1 = 6*np.power(x,0.59)
y2 = 1.2*np.power(x,0.75)

fig = plt.figure(figsize=(5,5))
ax = plt.subplot()

ax.scatter(100*obsPercentage,100*d0Vertex/baseVertex,5)
ax.fill_between(x,y1,y2,alpha=0.25,zorder=-2)
ax.set_xlim([0,100])
ax.set_ylim([0,100])
ax.set_xlabel('Empty State Occupancy Percentage (\%)')
ax.set_ylabel('Vertex Count Reduction Rate (\%)')
ax.set_aspect('equal')
ax.grid()
plt.tight_layout()
plt.show()
#plt.savefig('vertex_new.png')

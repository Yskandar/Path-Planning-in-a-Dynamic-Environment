
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

df = pd.read_csv("collected_data1000.csv")

thresh = df['Threshold']
resolution = df['Resolution']
baseVertex = df['Base V #']
d0Vertex = df['D(0) V #']
d1Vertex = df['D(1) V #']
d2Vertex = df['D(2) V #']
d3Vertex = df['D(3) V #']
d4Vertex = df['D(4) V #']

#z3 = np.polyfit(xPlot[0.3], yPlot[0.3], 2)

plt.rcParams['text.usetex'] = True

fig = plt.figure(figsize=(8,5))
ax = plt.subplot()

#data = np.random.normal(100, 20, 200)

ax.boxplot([baseVertex,d0Vertex,d1Vertex,d2Vertex,d3Vertex], showfliers=False)

#ax.scatter(xPlot[0.7],yPlot[0.7],5)
#ax.set_xlim([5,15])
#ax.set_ylim([0,100])
ax.set_xlabel('Different Levels of QuadTree Decomposition')
ax.set_ylabel('Total Vertex Count')
ax.set_xticklabels(['Original','Decompose(0)','Decompose(1)','Decompose(2)','Decompose(3)'])
#ax.set_aspect('equal')
ax.grid()
plt.tight_layout()
#plt.show()
plt.savefig('vertex_box.png')

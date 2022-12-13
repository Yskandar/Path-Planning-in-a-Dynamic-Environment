
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

df = pd.read_csv("data.csv")

thresh = df['Threshold']
resolution = df['Resolution']
baseVertex = df['Base V #']
d0Vertex = df['D(0) V #']

xPlot = defaultdict(list)
yPlot = defaultdict(list)
for i,t in enumerate(thresh):
	value = 100*d0Vertex[i]/baseVertex[i]
	xPlot[t].append(resolution[i])
	yPlot[t].append(value)
print(xPlot)

#z3 = np.polyfit(xPlot[0.3], yPlot[0.3], 2)


plt.rcParams['text.usetex'] = True


fig = plt.figure(figsize=(5,5))
ax = plt.subplot()

#ax.scatter(xPlot[0.2],yPlot[0.2],5)

thresh = 0.3
x = range(len(xPlot[thresh]))
poly_fit = np.poly1d(np.polyfit(xPlot[thresh], yPlot[thresh], 2))
ax.plot(x,poly_fit(x),5,color='tab:blue')
thresh = 0.5
x = range(len(xPlot[thresh]))
poly_fit = np.poly1d(np.polyfit(xPlot[thresh], yPlot[thresh], 2))
ax.plot(x,poly_fit(x),5,color='tab:orange')
thresh = 0.7
x = range(len(xPlot[thresh]))
poly_fit = np.poly1d(np.polyfit(xPlot[thresh], yPlot[thresh], 2))
ax.plot(x,poly_fit(x),5,color='tab:green')

ax.legend(['Threshold = 0.3','Threshold = 0.5','Threshold = 0.7'])
leg = ax.get_legend()
leg.legendHandles[0].set_color('tab:blue')
leg.legendHandles[1].set_color('tab:orange')
leg.legendHandles[2].set_color('tab:green')

ax.scatter(xPlot[0.3],yPlot[0.3],5,color='tab:blue')
ax.scatter(xPlot[0.5],yPlot[0.5],5,color='tab:orange')
ax.scatter(xPlot[0.7],yPlot[0.7],5,color='tab:green')
#ax.scatter(xPlot[0.7],yPlot[0.7],5)
ax.set_xlim([5,15])
ax.set_ylim([0,100])
ax.set_xlabel('Perlin Noise Octave Level (Frequency)')
ax.set_ylabel('Vertex Count Reduction Rate (\%)')
#ax.set_aspect('equal')
ax.grid()
plt.tight_layout()
#plt.show()
plt.savefig('vertex_octave.png')

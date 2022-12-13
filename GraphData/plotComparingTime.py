from statistics import mean 
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("compareTimeData_dim2.csv")

thresh = df['Dimension']
vanillaPath = df['VanillaPathTime']
generateTime = df['QuadGenTime']
quadPath = df['QuadPathTime']



plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 14})

window = 10
threshold = []
vanillaMean = []
quadMean = []
genMean = []


for i in range(len(vanillaPath)//10):
	threshold.append(mean(thresh[10*i:10*i+10]))
	vanillaMean.append(mean(vanillaPath[10*i:10*i+10]))
	quadMean.append(mean(quadPath[10*i:10*i+10]))
	genMean.append(mean(generateTime[10*i:10*i+10]))
	#print(threshold)


'''
for ind in range(len(vanillaPath)-window + 1):
	vanillaMean.append(np.mean(vanillaPath[ind:ind+window]))
	quadMean.append(np.mean(quadPath[ind:ind+window]))
	genMean.append(np.mean(generateTime[ind:ind+window]))
for ind in range(window-1):
	vanillaMean.insert(0,np.nan)
	quadMean.insert(0,np.nan)
	genMean.insert(0,np.nan)
'''
fig = plt.figure(figsize=(10,5))
ax = plt.subplot()
plt.yscale('log')

#ax.scatter(thresh,vanillaPath,50,color='tab:blue',marker='.')
#ax.scatter(thresh,quadPath,50,color='tab:orange',marker='.')
#ax.scatter(thresh,generateTime,50,color='tab:green',marker='.')
ax.scatter(threshold,vanillaMean,50,marker='x',color='tab:blue')
ax.scatter(threshold,quadMean,50,marker='x',color='tab:orange')
ax.scatter(threshold,genMean,50,marker='x',color='tab:green')

ax.set_xlabel('World Map Dimensions (pixels)')
ax.set_ylabel('Time to Generate A Valid Path (seconds)')
#ax.set_aspect('equal')
#ax.set_ylim([0.0001,10^1])
ax.legend([
	'Path Finding Time on Original State Space',
	'Path Finding Time + QuadTree Generation Time',
	'QuadTree Generation Time'
	#'Moving Average of Computation Time Using Original State Space',
	#'Moving Average of Computation Time Using Quad Decomposed State Space',
	#'Moving Average of Computation Time Required to Generate QuadTree'
	],loc='lower right')
#leg = ax.get_legend()
#leg.legendHandles[0].set_color('tab:blue')
#leg.legendHandles[1].set_color('tab:orange')
#leg.legendHandles[2].set_color('tab:green')

ax.grid('both')
plt.tight_layout()
plt.show()
#plt.savefig('compareTime.png')

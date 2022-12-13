import math
import numpy as np
import matplotlib.pyplot as plt
from perlin_noise import PerlinNoise

class RandomMap():
	def __init__(self,dimension, thresh, scale):
		self.xDim, self.yDim = dimension
		self.image = np.zeros((self.xDim,self.yDim))
		self.thresh = thresh
		self.scale = scale
		noise = PerlinNoise()
		for x in range(self.xDim):
			for y in range(self.yDim):
				value = noise([x/self.scale,y/self.scale])+0.5
				self.image[x,y] = math.ceil(value) if value < self.thresh else math.floor(value)

	def plot(self, ax):
		im = ax.imshow(self.image, cmap='gray')
		plt.colorbar(im)

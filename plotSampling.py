import matplotlib.pyplot as plt
import numpy as np

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

x = np.arange(0, 3.0, 0.001)
atomPositionX = 1.5
sigma = 0.5

y = gaussian(x, atomPositionX, sigma)

plt.plot(x,y)
heads = gaussian(np.array(range(0, 4)), atomPositionX, sigma)
plt.stem(np.array(range(0, 4)),heads)
plt.axis('off')
plt.show()
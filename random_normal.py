import matplotlib.pyplot as plt
import numpy as np



mu, sigma = 1, 10
s = np.random.normal(mu, sigma, 100)
count, bins, ignored = plt.hist(s, 100, density=True)
print(s)
print("X:", bins, "\n")
print("Y:", 1/(sigma * np.sqrt(2 * np.pi)) *
               np.exp( - (bins - mu)**2 / (2 * sigma**2)
               ))

plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
               np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
         linewidth=2, color='green')
plt.show()
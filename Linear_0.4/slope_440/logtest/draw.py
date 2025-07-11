import numpy as np
import matplotlib.pyplot as plt
data = np.loadtxt("xe.txt")
sample_indices = np.random.choice(data.shape[0],60000,replace=False)
x = data[:,1]
y = data[:,2]
x = x[sample_indices]
y = y[sample_indices]
data_sample = np.vstack([x,y])
from scipy.stats import gaussian_kde
kde = gaussian_kde(data_sample)
density = kde(data_sample)
density_norm = (density - density.min()) #/(density.max() - density.min())
fig = plt.figure()
plt.subplot(1,3,1)
ax = fig.add_subplot(111)
sc = ax.scatter(x,y, c=density_norm, cmap = 'jet', vmin = 0, vmax = 1, alpha = 0 + 0.3*(density_norm), s = 3, label='xy')
cbar = plt.colorbar(sc, ax = ax , pad=0.5)
plt.legend()
plt.grid()
plt.subplot(1,3,2)
bins_x = np.linspace(x.min(),x.max(),30)
bins_y = np.linspace(y.min(),y.max(),30)
plt.hist(x,bins=bins_x,label='x_plot')
plt.legend()
plt.grid()
plt.subplot(1,3,3)
plt.hist(y,bins=bins_y,label='y_plot')
plt.legend()
plt.grid()
plt.show()

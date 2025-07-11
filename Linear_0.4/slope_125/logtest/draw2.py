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
xmin,xmax = x.min()-0.1, x.max()+0.1
ymin,ymax = y.min()-0.1, y.max()+0.1
print(xmin,xmax)
print(ymin,ymax)
X,Y = np.meshgrid(np.linspace(xmin,xmax,100), np.linspace(ymin,ymax,100))
position = np.vstack([X.ravel(),Y.ravel()])
Z = kde(position).reshape(X.shape)
#density = kde(data_sample)
#density_norm = (density - density.min())  #/(density.max() - density.min())
Z = Z - Z.min()
fig = plt.figure(figsize=(18,6))
plt.subplot(1,3,1)
im = plt.imshow(Z, origin='lower',extent=[xmin,xmax,ymin,ymax],cmap='jet',aspect='auto')
plt.gca().set_autoscale_on(False)
plt.xlim(xmin,xmax)
plt.ylim(ymin,ymax)
plt.colorbar(im,label="density")
#ax = fig.add_subplot(111)
#sc = ax.scatter(x,y, c=density_norm, cmap = 'jet', vmin = 0, vmax = 1, alpha = 0 + 0.4*(density_norm), s = 5, label='xy')
#sc = ax.scatter(X,Y, c=Z, cmap = 'jet', vmin = 0, vmax = 1,  s = 5, label='xy')
#cbar = plt.colorbar(sc, ax = ax , pad=0.3)
#plt.legend()
plt.grid()
##############################33
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

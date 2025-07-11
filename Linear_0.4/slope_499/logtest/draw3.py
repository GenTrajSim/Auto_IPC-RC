import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter
data = np.loadtxt("xe.txt")
sample_indices = np.random.choice(data.shape[0],60000,replace=False)
x = data[:,1]
y = data[:,2]
x = x[sample_indices]
y = y[sample_indices]
data_sample = np.vstack([y,x])
from scipy.stats import gaussian_kde
kde = gaussian_kde(data_sample)
xmin,xmax = -100,120#x.min()-0.1, x.max()+0.1
ymin,ymax = -2,2 #y.min()-0.1, y.max()+0.1
print(xmin,xmax)
print(ymin,ymax)
X,Y = np.meshgrid(np.linspace(xmin,xmax,100), np.linspace(ymin,ymax,100))
position = np.vstack([X.ravel(),Y.ravel()])
Z = kde(position).reshape(X.shape)
#density = kde(data_sample)
#density_norm = (density - density.min())  #/(density.max() - density.min())
Z = (Z)/( (4)*(220)/10000 )
Z = Z - Z.min()
#Z = np.ma.masked_where(Z<=0.001,Z)
fig = plt.figure(figsize=(6,6))
#plt.subplot(1,3,1)
#im = plt.imshow(Z, origin='lower',extent=[xmin,xmax,ymin,ymax],cmap='jet',aspect='auto')
norm_z = Z/(Z.max()-Z.min())
im = plt.imshow(Z,vmin=0, vmax=0.22,origin='lower',extent=[-100,120,-2,2],cmap='jet',aspect='auto')
plt.gca().set_autoscale_on(False)
plt.gca().yaxis.set_major_locator(MultipleLocator(1))
plt.gca().yaxis.set_minor_locator(MultipleLocator(0.5))
plt.gca().xaxis.set_major_locator(MultipleLocator(40))
plt.gca().xaxis.set_minor_locator(MultipleLocator(20))
plt.tick_params(axis='y',which='minor',labelbottom=False)
plt.tick_params(axis='x',which='minor',labelbottom=False)
#plt.gca().set_facecolor((1,1,1,0))
plt.xlim(-100,120)
plt.ylim(-2,2)
plt.colorbar(im,label="density")
#ax = fig.add_subplot(111)
#sc = ax.scatter(x,y, c=density_norm, cmap = 'jet', vmin = 0, vmax = 1, alpha = 0 + 0.4*(density_norm), s = 5, label='xy')
#sc = ax.scatter(X,Y, c=Z, cmap = 'jet', vmin = 0, vmax = 1,  s = 5, label='xy')
#cbar = plt.colorbar(sc, ax = ax , pad=0.3)
#plt.legend()
#plt.grid()
##############################33
#plt.subplot(1,3,2)
#bins_x = np.linspace(x.min(),x.max(),30)
#bins_y = np.linspace(y.min(),y.max(),30)
#plt.hist(x,bins=bins_x,label='x_plot')
#plt.legend()
#plt.grid()
#plt.subplot(1,3,3)
#plt.hist(y,bins=bins_y,label='y_plot')
#plt.legend()
#plt.grid()
plt.show()

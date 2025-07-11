import numpy as np
import matplotlib.pyplot as plt
data = np.genfromtxt("train.log",delimiter=' ',invalid_raise=False)
loss2 = data[:,5]
loss1 = data[:,2]
plt.plot(loss1)
plt.plot(loss2)
plt.yscale('log')
plt.grid()
plt.show()


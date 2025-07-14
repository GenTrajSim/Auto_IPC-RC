import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from scipy.stats import gaussian_kde
PT = "_1800_188"
detai = 300*10
#########################
data1 = np.loadtxt("Linear_0.2/slope_455/logtest/xe"+PT+".txt")
data2 = np.loadtxt("Linear_0.4/slope_490/logtest/xe"+PT+".txt")
x = data1[:,1]
y = data1[:,2]
z = data2[:,2]
# Create sample data
num_frames = x.shape[0]//detai
#x = np.random.rand(num_frames * 100)
#y = np.random.rand(num_frames * 100)
#z = np.random.rand(num_frames * 100)
averages = [np.mean(x[i*detai:(i+1)*detai]) for i in range(num_frames)]
# Set up the figure and 3D axis
fig = plt.figure(figsize=(15,6))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122)
# Create a grid of points
#xp = np.linspace(-2, 2, 100)  # 1D array for X
#yp = np.linspace(-60, 90, 100)  # 1D array for Y
#xp, yp = np.meshgrid(xp, yp)     # Create a 2D grid
# Define a fixed plane (e.g., z = 0)
#zp = np.zeros_like(xp)         # Create a 2D array for Z with the same shape as X and Y
#zp = (1-xp/0.25 -yp/4)*(13)+10
#print(zp)
#ax1.plot_surface(xp, yp, zp, alpha=0.45, color='lightblue')
#fig, (ax1,ax2) = plt.subplots(1,2, subplot_kw{'projection':'3d'})
#fig, (ax1, ax2) = plt.subplots(1,2,subplot_kw={'projection': '3d'})
#fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={'projection': '3d'})
#fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [2, 1]})
# Function to update the scatter plot for each frame
def update(frame):
    ax1.clear()  # Clear the previous frame
    xi=None
    yi=None
    zi=None
    datai=None
    xi = x[frame*detai:(frame+1)*detai]
    yi = y[frame*detai:(frame+1)*detai]
    zi = z[frame*detai:(frame+1)*detai]
    datai = np.vstack([xi,yi,zi])
    kdei = None
    densityi = None
    densityi_nor = None
    kdei = gaussian_kde(datai)
    densityi = kdei(datai)
    densityi_nor = (densityi -densityi.min())/(densityi.max()-densityi.min())
    maski= None
    maski = densityi_nor > 0.15
    ax1.scatter(xi[maski], yi[maski], zi[maski], c=densityi_nor[maski], vmin=0, cmap='jet', alpha=1*((densityi_nor[maski])**2), depthshade=False, s=20, antialiased=True)
    #ax1.set_title(f'time {(frame1)*0.02} -> {frame2*0.02} us')
    ###
    #ax1.scatter(x[frame*detai:(frame+1)*detai], y[frame*detai:(frame+1)*detai], z[frame*detai:(frame+1)*detai], alpha=0.1, depthshade=False, s=20, antialiased=True)
    ax1.set_title(f' {PT} time {(frame + 1)*200} x0.001 us')
    ax1.set_xlabel('norm. rho Axis')
    ax1.set_ylabel('RCII Axis')
    ax1.set_zlabel('RCI Axis')
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-40, 40)
    ax1.set_zlim(-40, 80)
    # Create a grid of points
    #xp = np.linspace(-2, 2, 100)  # 1D array for X
    #yp = np.linspace(-60, 90, 100)  # 1D array for Y
    #xp, yp = np.meshgrid(xp, yp)     # Create a 2D grid
    # Define a fixed plane (e.g., z = 0)
    #zp = np.zeros_like(xp)         # Create a 2D array for Z with the same shape as X and Y
    #zp = (1-xp/0.25 -yp/4)*(13)+10
    #print(zp)
    #ax1.plot_surface(xp, yp, zp, alpha=0.35, color='sandybrown')
    ax1.view_init(azim=-153,elev=35)
    
    ax2.clear()
    ax2.set_title(' time vs Average of norm. rho')
    ax2.set_xlabel('time x0.001 us')
    ax2.set_ylabel('Average of rho (norm.)')
    ax2.set_xlim(0, (num_frames - 1)*200)
    ax2.set_ylim(-1, 1)
    ax2.plot(range(0,(frame + 1)*200,200), averages[:(frame + 1)], color='blue', marker='o')
    #ax2.axhline(y=averages[frame], color='red', linestyle='--', label=f'Avg: {averages[frame]:.2f}')
    #ax2.legend()

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=num_frames, repeat=True)

# Save the animation as a video file
#ani.save('3d_scatter_movie.mp4', writer='ffmpeg', fps=60,dpi=300, 
#        extra_args=["-vcodec","libx264","-crf","18","-preset","slow"])
ani.save("3d"+PT+"_scatter_movie_7-14.mp4", writer='ffmpeg', fps=4 ,
         extra_args=["-crf", "18", "-preset", "ultrafast"])
plt.rcParams['figure.dpi']=300
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import gaussian_kde
from scipy.special import erf
import math
from matplotlib.colors import Normalize
from scipy.spatial import cKDTree
#import os
import sys
arg1 = sys.argv[1]
PT = "100_280"
PT = arg1
data1 = np.loadtxt("Linear_0.2/slope_455/logtest/xe_"+PT+".txt")#2200_185.txt")
data2 = np.loadtxt("Linear_0.4/slope_490/logtest/xe_"+PT+".txt")#_2200_185.txt")
#data1 = np.loadtxt("Linear_0.2/slope_455/logtest/xe_2200_185.txt")
#data2 = np.loadtxt("Linear_0.4/slope_490/logtest/xe_2200_185.txt")
print(data1.shape)
#sample_i = np.random.choice(data1.shape[0],200000,replace=False)
sample_i = np.random.choice(data1.shape[0],200000,replace=False)
x = data1[:,1]
y = data1[:,2]
z = data2[:,2]
x = x[sample_i]
y = y[sample_i]
z = z[sample_i]
bins = 85
hist,edges = np.histogramdd((x,y,z),bins=bins)
x_idx = np.digitize(x,edges[0])-1
y_idx = np.digitize(y,edges[1])-1
z_idx = np.digitize(z,edges[2])-1
x_dix = np.clip(x_idx,0,bins-1)
y_dix = np.clip(y_idx,0,bins-1)
z_dix = np.clip(z_idx,0,bins-1)
density = hist[x_dix,y_dix,z_dix]/200000
line_point1 = np.array([0.15,-5,48])
line_point2 = np.array([-0.6754,5, 23])
A_point = np.array([0.16, 15, 48])
B_point = np.array([-0.5, -10, -16])
#data = np.vstack([x,y,z])
#kde = gaussian_kde(data)
##density = kde(data)
#v = kde(data)
#norm = Normalize(vmin=v.min(), vmax=v.max())
#v_normalized = norm(v)
#cmap = plt.cm.jet#Reds#YlOrRd
#colors = cmap(v_normalized)  
#colors[:, 3] = (v_normalized**2)* 0.3+0.020 
#ss = 5#(v_normalized)*0.12+0.001
#mask = v_normalized > 0.10###########
#fig = plt.figure(figsize=(8, 6),dpi=300)
#ax = fig.add_subplot(111, projection='3d')
#sc = ax.scatter(x[mask], y[mask], z[mask], c=colors[mask],  s = ss , depthshade=True)
#mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
#mappable.set_array([]) 
#plt.colorbar(mappable,ax=ax, label='Density')
#plt.show()

density_nor = (density -density.min())#/(density.max()-density.min())
density_alp = (density -density.min())/(density.max()-density.min())
print(density_nor.max())
R_sate = 0.01
#mask_point1x = ((((x-line_point1[0])/(x.max()-x.min()))**2)+(((y-line_point1[1])/(y.max()-y.min()))**2)+(((z-line_point1[2])/(z.max()-z.min()))**2)) < (R_sate**2)
mask = density_alp > 0.1 #0.0458  ##0.18, 0.1for 3000
fig = plt.figure(figsize=(8,6),dpi=300)
ax = fig.add_subplot(111,projection='3d')
sc = ax.scatter(x[mask],y[mask],z[mask], c=density_nor[mask], cmap='jet', vmin=0,vmax=0.0003, alpha=(0.2*density_alp[mask]**3),s=5,depthshade=False, antialiased=True)
#ax.scatter(line_point1[0], line_point1[1], line_point1[2],color='red')
#ax.scatter(line_point2[0], line_point2[1], line_point2[2],color='red')
####################################################################################
mask_point1x = ((((x-line_point1[0])/(x.max()-x.min()))**2)+(((y-line_point1[1])/(y.max()-y.min()))**2)+(((z-line_point1[2])/(z.max()-z.min()))**2)) < (R_sate**2)
print("x[mask_point1x]:",x[mask_point1x].shape,"ave: ",np.mean(density_nor[mask_point1x]),"-ln: ",-1*np.log(np.mean(density_nor[mask_point1x])))
#ax.scatter(x[mask_point1x], y[mask_point1x], z[mask_point1x],color='black')
mask_point2x = ((((x-line_point2[0])/(x.max()-x.min()))**2)+(((y-line_point2[1])/(y.max()-y.min()))**2)+(((z-line_point2[2])/(z.max()-z.min()))**2)) < (R_sate**2)
print("x[mask_point2x]:",x[mask_point2x].shape,"ave: ",np.mean(density_nor[mask_point2x]),"-ln: ",-1*np.log(np.mean(density_nor[mask_point2x])))
#ax.scatter(x[mask_point2x], y[mask_point2x], z[mask_point2x],color='black')
mask_A = ((((x-A_point[0])/(x.max()-x.min()))**2)+(((y-A_point[1])/(y.max()-y.min()))**2)+(((z-A_point[2])/(z.max()-z.min()))**2)) < (R_sate**2)
print("x[A_point]:",x[mask_A].shape,"ave: ",np.mean(density_nor[mask_A]),"-ln: ",-1*np.log(np.mean(density_nor[mask_A])))
#ax.scatter(x[mask_A], y[mask_A], z[mask_A],color='black')
mask_B = ((((x-B_point[0])/(x.max()-x.min()))**2)+(((y-B_point[1])/(y.max()-y.min()))**2)+(((z-B_point[2])/(z.max()-z.min()))**2)) < (R_sate**2)
print("x[B_point]:",x[mask_B].shape,"ave: ",np.mean(density_nor[mask_B]),"-ln: ",-1*np.log(np.mean(density_nor[mask_B])))
#ax.scatter(x[mask_B], y[mask_B], z[mask_B],color='black')
#sc = ax.scatter(x[mask],y[mask],z[mask], c=density_nor[mask], cmap='jet', vmin=0,vmax=0.0003, alpha=0.1,s=0.1,depthshade=False, antialiased=True)
xp = np.linspace(-1.5, 1.5, 50)  # 
yp = np.linspace(-30, 30, 50)  # 
xp, yp = np.meshgrid(xp, yp)      #
z_plane1 = (1-(xp/0.2)-(yp/2.))*21 - 16
z_plane2 = (1-(xp/0.2)-(yp/2.))*21 + 24
params2 = [-21/0.4,-21/4.,-1.,21-16]
params1 = [-21/0.4,-21/4.,-1.,21+23]
theta = 3.*np.pi / 6
plane_normal = np.array([params1[0], params1[1], params1[2]])
plane_point = np.array([0, 0, (-params1[3] / params1[2])-20])
R = 2.4
#
def dist1(param1,param2):
    A1,B1,C1,D1 = param1
    A2,B2,C2,D2 = param2
    Tdata = np.vstack([x,y,z])
    Tdata = Tdata.T
    distance1 = (np.dot(Tdata,[A1,B1,C1])+D1)/np.sqrt((A1**2)+(B1**2)+(C1**2))
    distance2 = (np.dot(Tdata,[A2,B2,C2])+D2)/np.sqrt((A2**2)+(B2**2)+(C2**2))
    return distance1,distance2
#
def find_plane_basis(normal):
    if normal[0] != 0:
        v = np.array([-normal[1], normal[0], 0]) 
    else:
        v = np.array([0, -normal[2], normal[1]])
    u1 = np.cross(normal, v)
    u1 = u1 / np.linalg.norm(u1)
    u2 = np.cross(normal, u1)
    u2 = u2 / np.linalg.norm(u2)  
    return u1, u2
def generate_line_in_plane(plane_normal, plane_point, theta, t_range,points):
    u1, u2 = find_plane_basis(plane_normal)
    direction = np.cos(theta) * u1 + np.sin(theta) * u2
    t = np.linspace(t_range[0], t_range[1], 100)
    x = plane_point[0] + direction[0] * t
    y = plane_point[1] + direction[1] * t
    z = plane_point[2] + direction[2] * t
    direction_norm_sq = np.dot(direction, direction)
    t_values = np.dot(points - plane_point, direction) / direction_norm_sq
    return x, y, z, t_values
def line_directions(plane_normal,plane_point,theta):
    u1, u2 = find_plane_basis(plane_normal)
    direction = np.cos(theta) * u1 + np.sin(theta) * u2
    line_direction = direction / np.linalg.norm(direction)
    return line_direction
line_direction = line_directions(plane_normal,plane_point,theta)
R = 0.5
line_point = plane_point
line_point1 = np.array([0.15,-5,48])
line_point2 = np.array([-0.6754,5, 23])
line_line_direction = line_point1 - line_point2 #np.array([0.75+1.2,-30-30,42-20])
line_line_direction = line_line_direction/np.linalg.norm(line_line_direction)
line_point = line_point1 #np.array([0.75,-30,42])
#ax.scatter(line_point1[0], line_point1[1], line_point1[2],color='red')
#ax.scatter(line_point2[0], line_point2[1], line_point2[2],color='red')
def project_point_to_line(point, line_point, line_direction):
    vector = point - line_point
    projection = line_point + np.dot(vector, line_direction) * line_direction
    return projection

def distance_to_line(point, line_point, line_direction):
    vector = point - line_point
    cross_p = np.cross(vector,line_direction)
    distance = np.linalg.norm(cross_p)/np.linalg.norm(line_direction)
    perpendicular_vector = vector - np.dot(vector, line_direction) * line_direction
    return distance#np.linalg.norm(perpendicular_vector)
xm = x[mask]
ym = y[mask]
zm = z[mask]
points = np.vstack([xm, ym, zm])
points = points.T
projection_positions = []
ponits_p = []
for point in points:
    distance = distance_to_line(point, line_point2, line_line_direction)
    if distance <= R:
        #print(distance)
        projection = project_point_to_line(point, line_point2, line_line_direction)
        projection_position = np.dot(projection - line_point2, line_line_direction)
        projection_positions.append(projection_position)
        ponits_p.append(point)
projection_positions = np.array(projection_positions)
#plane_normal = np.array([params1[0], params1[1], params1[2]])
#plane_point = np.array([0, 0, (-params1[3] / params1[2])-10]) 
#theta = 4*np.pi / 5########################################
#
ponits_p = np.array(ponits_p)
print("points_p.shape=",ponits_p.shape)
#ax.scatter(ponits_p[:,0],ponits_p[:,1],ponits_p[:,2],c='black',s=5)
mask_2dist1,mask_2dist2 = dist1(params1,params2)
mask_21 = (mask_2dist1>0) #and (mask_2dist2>0)
mask_22 = (mask_2dist2<0)
mask_2 = mask_21*mask_22
test_n_x = x[mask_2]
test_n_y = y[mask_2]
test_n_z = z[mask_2]
test_n = np.vstack([test_n_x, test_n_y, test_n_z])
test_n = test_n.T
print('N:',test_n.shape[0])
#point_line = np.array([0,0,0])
#directions = np.array([1,1,1])
#t_range = (-20,20)
t = np.linspace(-50, 50, 100)
lline = line_point + t[:, np.newaxis] * line_line_direction
#x_line, y_line, z_line, t_values = generate_line_in_plane(plane_normal, plane_point, theta, t_range, test_n)
#x_line = point_line[0] + directions[0]*t
#y_line = point_line[1] + directions[1]*t
#z_line = point_line[2] + directions[2]*t

#ax.plot(lline[:,0],lline[:,1],lline[:,2],color='black')
#ax.scatter(plane_point[0],plane_point[1],plane_point[2],color='red')

#
#print('N:',test_n.shape[0])
#ax.scatter(x[mask_2],y[mask_2],z[mask_2],color='black',s=5,alpha=0.5)
#
#plt.figure(figsize=(8, 6))
#plt.subplot(1,3,2)
#plt.hist(t_values, bins=100, color='blue', alpha=0.7, edgecolor='black')
#plt.xlabel('t (Projection on Line)', fontsize=12)
#plt.ylabel('Density (Number of Points)', fontsize=12)
#plt.title('Projection Density on Line', fontsize=14)
#plt.grid(alpha=0.3)
#plt.show()
#
#ax.plot_surface(xp, yp, z_plane1, alpha=0.5, rstride=100, cstride=100, color='cyan')
#ax.plot_surface(xp, yp, z_plane2, alpha=0.5, rstride=100, cstride=100, color='cyan')
new_points = np.array([
    [-0.6,8.,30],
    [-0.53,6,31.25],
    [-0.49,4,32.5],
    [-0.37,2,33.25],
    [-0.35,0,35],
    [-0.26,-2,36.25],
    [-0.055,-4,37.5],
    [0.15,-6,45.25],
    [0.24,-8,50]
    ])
new_x,new_y,new_z = new_points[:,0],new_points[:,1],new_points[:,2]
original_points = np.vstack([x,y,z]).T
#new_points = np.vstack([new_x,new_y,new_z])
tree = cKDTree(original_points)
nearest_10_ave_v = []
for new_point in new_points:
    distance,indices = tree.query(new_point,k=5000)
    avg_v = np.mean(density_nor[indices])
    nearest_10_ave_v.append(avg_v)
#for i,avg_v in enumerate(nearest_10_ave_v):
#    print(new_points[i],avg_v)
#ax.scatter(new_x,new_y,new_z,c='black',s=2)
#ax.plot(new_x,new_y,new_z,c='black',linewidth=1)
ax.set_xlabel('norm. rho Axis')
ax.set_ylabel('RCII Axis')
ax.set_zlabel('RCI Axis')
ax.set_xlim([-1.5,1.5])
ax.set_ylim([-30,30])
ax.set_zlim([-30,70])
ax.view_init(azim=-174,elev=49)#-148 49# azim=-174,elev=49
plt.rcParams['figure.dpi']=300
cbar = plt.colorbar(sc, ax=ax, pad=0.2)
##

#plt.subplot(1,2,2)
#print("N: ",projection_positions.shape)
#plt.hist(projection_positions, bins=80, color='blue', alpha=0.7, edgecolor='black')
#plt.xlabel('t (Projection on Line)', fontsize=12)
#plt.ylabel('Density (Number of Points)', fontsize=12)
#plt.title('Projection Density on Line', fontsize=14)
#plt.grid(alpha=0.3)

#plt.show()
#ax.grid(False)
#ax.set_axis_off()
#np.savetxt('hist_prefect'+PT+'20.txt', projection_positions)
plt.savefig("./3D_P_"+PT+".tif", dpi=300, transparent=False)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
#PT = "1_260"
PT = "P1800_T188"
#data1 = np.loadtxt("Linear_0.2/slope_455/logtest/xe_"+PT+".txt")
#data2 = np.loadtxt("Linear_0.4/slope_490/logtest/xe_"+PT+".txt")
data1 = np.loadtxt("Linear_0.2/slope_455/big_system/xe"+PT+".txt")
data2 = np.loadtxt("Linear_0.4/slope_490/big_system/xe"+PT+".txt")

x = data1[:,1]
y = data1[:,2]
z = data2[:,2]

def RC_function(params):
    A,B,C = params
    Tdata = np.vstack([x,y,z])
    Tdata = Tdata.T
    distance = (np.dot(Tdata,[A,B,C])-1)/np.sqrt((A**2)+(B**2)+(C**2))
    return distance

params = [1/(0.5*(1+(5/21))),1/(5*(1+(5/21))),1/(21*(1+(5/21)))]

RC = RC_function(params)

RC_r1 = RC.reshape(RC.shape[0]//30000,30000)
xe = (RC_r1>0).sum(axis=1)
RC_r2 = RC_r1.sum(axis=1)
x_r1 = x.reshape(x.shape[0]//30000,30000)
y_r1 = y.reshape(y.shape[0]//30000,30000)
z_r1 = z.reshape(z.shape[0]//30000,30000)
x_r2 = x_r1.sum(axis=1)
y_r2 = y_r1.sum(axis=1)
z_r2 = z_r1.sum(axis=1)
print(RC.shape[0])
print(RC_r2.shape)
RCmax = np.column_stack((RC_r2/30000,x_r2/30000,y_r2/30000,z_r2/30000,xe/30000))
np.savetxt("RC_"+PT+".txt",RCmax)
print("maxRCmix:",RC.max(),"minRCmin",RC.min())

#with open("Linear_0.2/slope_455/logtest/xe.log", "r", encoding="utf-8") as file, open("new.lammpstrj","w",encoding="utf-8") as tgt:
with open("Linear_0.2/slope_455/big_system/xe"+PT+".log", "r", encoding="utf-8") as file, open("new_big_model"+PT+".lammpstrj","w",encoding="utf-8") as tgt:
    linei = 0
    lineij =0
    timestep=0
    for line in file:
        #if linei % 309 == 0:
        #    lineij = 0
        #else:
        linei = linei + 1
        if (linei %30009>9) or (linei %30009==0):
            tgt.write(f"{line.strip()} {RC[lineij]}\n")
            lineij = lineij + 1
            if linei %30009==0:
                timestep = timestep + 1
        else:
            if linei %30009==9:
                tgt.write(line.strip() + " RCmix\n")
            else:
                if linei %30009==2:
                    tgt.write(f"{timestep*100}\n")
                else:
                    tgt.write(line.strip() + "\n")

#data2_lammps = np.loadtxt("Linear_0.4/slope_490/logtest/xe_1800_188.log")


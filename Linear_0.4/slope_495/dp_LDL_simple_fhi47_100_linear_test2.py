import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random
import numpy as np
import pandas as dp
#import matplotlib as plt
import glob
import os
import sys
import math
import pandas as pd
from scipy.spatial.distance import pdist,squareform
from scipy.stats import rankdata
AUTO = tf.data.experimental.AUTOTUNE

#path = glob.glob("../../../dp_LDL/Dataset/P1800_T188/data1/new_coord/*npy")#+glob.glob("/kaggle/input/newldldata/data2/data2/new_coord/*npy")

path = glob.glob("../../../dp_LDL/BalanceDataset/P100_T235/data1/new_coord/*npy")

pi = math.acos(-1)
#pi
data_rho_ave = pd.read_csv('../../../dp_LDL/Dataset/P1725_T190/data1/boxdata.csv')
data_rho_ave2= pd.read_csv('../../../dp_LDL/Dataset/P1725_T190/data2/boxdata.csv')
data_rho_ave3= pd.read_csv('../../../dp_LDL/Dataset/P1600_T195/data1/boxdata.csv')
data_rho_ave4= pd.read_csv('../../../dp_LDL/Dataset/P1600_T198/data1/boxdata.csv')
data_rho_ave5= pd.read_csv('../../../dp_LDL/Dataset/P1725_T186/data1/boxdata.csv')
data_rho_ave6= pd.read_csv('../../../dp_LDL/Dataset/P1725_T188/data1/boxdata.csv')
data_rho_ave7= pd.read_csv('../../../dp_LDL/Dataset/P1500_T195/data1/boxdata.csv')
data_rho_ave8= pd.read_csv('../../../dp_LDL/Dataset/P1500_T198/data1/boxdata.csv')
data_rho_ave9= pd.read_csv('../../../dp_LDL/Dataset/P1600_T200/data1/boxdata.csv')
data_rho_ave10=pd.read_csv('../../../dp_LDL/Dataset/P1600_T193/data1/boxdata.csv')
data_rho_ave11= pd.read_csv('../../../dp_LDL/Dataset/P1725_T186/data2/boxdata.csv')
data_rho_ave12= pd.read_csv('../../../dp_LDL/Dataset/P1500_T198/data2/boxdata.csv')
data_rho_ave13= pd.read_csv('../../../dp_LDL/Dataset/P1600_T193/data2/boxdata.csv')
data_rho_ave14= pd.read_csv('../../../dp_LDL/Dataset/P1500_T195/data2/boxdata.csv')
data_rho_ave15= pd.read_csv('../../../dp_LDL/Dataset/P1600_T200/data2/boxdata.csv')
data_rho_ave16= pd.read_csv('../../../dp_LDL/Dataset/P1725_T188/data2/boxdata.csv')###########
data_rho_ave17= pd.read_csv('../../../dp_LDL/Dataset/P1500_T201/data1/boxdata.csv')
data_rho_ave18= pd.read_csv('../../../dp_LDL/Dataset/P1600_T190/data1/boxdata.csv')
data_rho_ave19= pd.read_csv('../../../dp_LDL/Dataset/P1725_T183/data1/boxdata.csv')
data_rho_ave20= pd.read_csv('../../../dp_LDL/Dataset/P1500_T192/data1/boxdata.csv')
data_rho_ave21= pd.read_csv('../../../dp_LDL/Dataset/P1600_T202/data1/boxdata.csv')
data_rho_ave22= pd.read_csv('../../../dp_LDL/Dataset/P1800_T185/data1/boxdata.csv')
data_rho_ave23= pd.read_csv('../../../dp_LDL/Dataset/P1500_T204/data1/boxdata.csv')
data_rho_ave24= pd.read_csv('../../../dp_LDL/Dataset/P1400_T198/data1/boxdata.csv')
data_rho_ave25= pd.read_csv('../../../dp_LDL/Dataset/P1250_T198/data1/boxdata.csv')
data_rho_ave26= pd.read_csv('../../../dp_LDL/Dataset/P1725_T193/data1/boxdata.csv')
data_rho_ave27= pd.read_csv('../../../dp_LDL/Dataset/P1800_T180/data1/boxdata.csv')
data_rho_ave28= pd.read_csv('../../../dp_LDL/Dataset/P1400_T200/data1/boxdata.csv')
data_rho_ave29= pd.read_csv('../../../dp_LDL/Dataset/P1725_T195/data1/boxdata.csv')
data_rho_ave30= pd.read_csv('../../../dp_LDL/Dataset/P1250_T200/data1/boxdata.csv')
data_rho_ave31= pd.read_csv('../../../dp_LDL/Dataset/P1800_T190/data1/boxdata.csv')
data_rho_ave32= pd.read_csv('../../../dp_LDL/Dataset/P1400_T202/data1/boxdata.csv')
data_rho_ave33= pd.read_csv('../../../dp_LDL/Dataset/P1725_T188/data3/boxdata.csv')
data_rho_ave34= pd.read_csv('../../../dp_LDL/Dataset/P1250_T205/data1/boxdata.csv')
data_rho_ave35= pd.read_csv('../../../dp_LDL/Dataset/P1800_T188/data1/boxdata.csv')
data_rho_ave36= pd.read_csv('../../../dp_LDL/Dataset/P1400_T207/data1/boxdata.csv')
data_rho_ave37= pd.read_csv('../../../dp_LDL/Dataset/P1725_T180/data1/boxdata.csv')
data_rho_ave38= pd.read_csv('../../../dp_LDL/Dataset/P1800_T195/data1/boxdata.csv')
data_rho_ave39= pd.read_csv('../../../dp_LDL/Dataset/P1250_T210/data1/boxdata.csv')
data_rho_ave40= pd.read_csv('../../../dp_LDL/Dataset/P1400_T193/data1/boxdata.csv')
data_rho_ave41= pd.read_csv('../../../dp_LDL/Dataset/P1400_T195/data1/boxdata.csv')
data_rho_ave42= pd.read_csv('../../../dp_LDL/Dataset/P1800_T178/data1/boxdata.csv')
data_rho_ave43= pd.read_csv('../../../dp_LDL/Dataset/P1250_T195/data1/boxdata.csv')
data_rho_ave44= pd.read_csv('../../../dp_LDL/Dataset/P1250_T202/data1/boxdata.csv')
data_rho_ave45= pd.read_csv('../../../dp_LDL/BalanceDataset/P3000_T230/data1/boxdata.csv')
data_rho_ave46= pd.read_csv('../../../dp_LDL/BalanceDataset/P100_T195/data1/boxdata.csv')
data_rho_ave47= pd.read_csv('../../../dp_LDL/BalanceDataset/P1_T200/data1/boxdata.csv')
data_rho_ave = np.array(data_rho_ave)
data_rho_ave2= np.array(data_rho_ave2)
data_rho_ave3= np.array(data_rho_ave3)
data_rho_ave4= np.array(data_rho_ave4)
data_rho_ave5= np.array(data_rho_ave5)
data_rho_ave6= np.array(data_rho_ave6)
data_rho_ave7= np.array(data_rho_ave7)
data_rho_ave8= np.array(data_rho_ave8)
data_rho_ave9= np.array(data_rho_ave9)
data_rho_ave10= np.array(data_rho_ave10)
data_rho_ave11= np.array(data_rho_ave11)
data_rho_ave12= np.array(data_rho_ave12)
data_rho_ave13= np.array(data_rho_ave13)
data_rho_ave14= np.array(data_rho_ave14)
data_rho_ave15= np.array(data_rho_ave15)
data_rho_ave16= np.array(data_rho_ave16)
data_rho_ave17= np.array(data_rho_ave17)
data_rho_ave18= np.array(data_rho_ave18)
data_rho_ave19= np.array(data_rho_ave19)
data_rho_ave20= np.array(data_rho_ave20)
data_rho_ave21= np.array(data_rho_ave21)
data_rho_ave22= np.array(data_rho_ave22)
data_rho_ave23= np.array(data_rho_ave23)
data_rho_ave24= np.array(data_rho_ave24)
data_rho_ave25= np.array(data_rho_ave25)
data_rho_ave26= np.array(data_rho_ave26)
data_rho_ave27= np.array(data_rho_ave27)
data_rho_ave28= np.array(data_rho_ave28)
data_rho_ave29= np.array(data_rho_ave29)
data_rho_ave30= np.array(data_rho_ave30)
data_rho_ave31= np.array(data_rho_ave31)
data_rho_ave32= np.array(data_rho_ave32)
data_rho_ave33= np.array(data_rho_ave33)
data_rho_ave34= np.array(data_rho_ave34)
data_rho_ave35= np.array(data_rho_ave35)
data_rho_ave36= np.array(data_rho_ave36)
data_rho_ave37= np.array(data_rho_ave37)
data_rho_ave38= np.array(data_rho_ave38)
data_rho_ave39= np.array(data_rho_ave39)
data_rho_ave40= np.array(data_rho_ave40)
data_rho_ave41= np.array(data_rho_ave41)
data_rho_ave42= np.array(data_rho_ave42)
data_rho_ave43= np.array(data_rho_ave43)
data_rho_ave44= np.array(data_rho_ave44)
data_rho_ave45= np.array(data_rho_ave45)
data_rho_ave46= np.array(data_rho_ave46)
data_rho_ave47= np.array(data_rho_ave47)
####
data_rhobox = np.vstack((data_rho_ave,data_rho_ave2))
data_rhobox = np.vstack((data_rhobox, data_rho_ave3))
data_rhobox = np.vstack((data_rhobox, data_rho_ave4))
data_rhobox = np.vstack((data_rhobox, data_rho_ave5))
data_rhobox = np.vstack((data_rhobox, data_rho_ave6))
data_rhobox = np.vstack((data_rhobox, data_rho_ave7))
data_rhobox = np.vstack((data_rhobox, data_rho_ave8))
data_rhobox = np.vstack((data_rhobox, data_rho_ave9))
data_rhobox = np.vstack((data_rhobox, data_rho_ave10))
data_rhobox = np.vstack((data_rhobox, data_rho_ave11))
data_rhobox = np.vstack((data_rhobox, data_rho_ave12))
data_rhobox = np.vstack((data_rhobox, data_rho_ave13))
data_rhobox = np.vstack((data_rhobox, data_rho_ave14))
data_rhobox = np.vstack((data_rhobox, data_rho_ave15))
data_rhobox = np.vstack((data_rhobox, data_rho_ave16))
data_rhobox = np.vstack((data_rhobox, data_rho_ave17))
data_rhobox = np.vstack((data_rhobox, data_rho_ave18))
data_rhobox = np.vstack((data_rhobox, data_rho_ave19))
data_rhobox = np.vstack((data_rhobox, data_rho_ave20))
data_rhobox = np.vstack((data_rhobox, data_rho_ave21))
data_rhobox = np.vstack((data_rhobox, data_rho_ave22))
data_rhobox = np.vstack((data_rhobox, data_rho_ave23))
data_rhobox = np.vstack((data_rhobox, data_rho_ave24))
data_rhobox = np.vstack((data_rhobox, data_rho_ave25))
data_rhobox = np.vstack((data_rhobox, data_rho_ave26))
data_rhobox = np.vstack((data_rhobox, data_rho_ave27))
data_rhobox = np.vstack((data_rhobox, data_rho_ave28))
data_rhobox = np.vstack((data_rhobox, data_rho_ave29))
data_rhobox = np.vstack((data_rhobox, data_rho_ave30))
data_rhobox = np.vstack((data_rhobox, data_rho_ave31))
data_rhobox = np.vstack((data_rhobox, data_rho_ave32))
data_rhobox = np.vstack((data_rhobox, data_rho_ave33))
data_rhobox = np.vstack((data_rhobox, data_rho_ave34))
data_rhobox = np.vstack((data_rhobox, data_rho_ave35))
data_rhobox = np.vstack((data_rhobox, data_rho_ave36))
data_rhobox = np.vstack((data_rhobox, data_rho_ave37))
data_rhobox = np.vstack((data_rhobox, data_rho_ave38))
data_rhobox = np.vstack((data_rhobox, data_rho_ave39))
data_rhobox = np.vstack((data_rhobox, data_rho_ave40))
data_rhobox = np.vstack((data_rhobox, data_rho_ave41))
data_rhobox = np.vstack((data_rhobox, data_rho_ave42))
data_rhobox = np.vstack((data_rhobox, data_rho_ave43))
data_rhobox = np.vstack((data_rhobox, data_rho_ave44))
data_rhobox = np.vstack((data_rhobox, data_rho_ave45))
data_rhobox = np.vstack((data_rhobox, data_rho_ave46))
data_rhobox = np.vstack((data_rhobox, data_rho_ave47))
rho_max = np.max(data_rhobox[:,6]) #+10
#rho_max
rho_min = np.min(data_rhobox[:,6]) #-10
#rho_min
#rho_ave = np.mean(data_rho_ave[:,6])
rho_ave = (rho_max + rho_min)/2
pot_max = np.max(data_rhobox[:,7])
pot_min = np.min(data_rhobox[:,7])
pot_ave = (pot_max + pot_min)/2

class LDL_Data_Feeder:
    def __init__(self,
                 total_path,
                 max_batch_system = 1,
                 local_atom_maxN = 90, system_OatomN = 300, system_all = 900,
                 len_dictionary = 4,
                 d_cut = 5,
                 Rcs = 4,
                 RC = 8,
                 Training = False):
        self.total_path = total_path
        #tf.print("===path===")
        #tf.print(total_path)
        self.max_batch_system = max_batch_system
        self.local_atom_maxN = local_atom_maxN
        self.system_OatomN = system_OatomN
        self.system_all = system_all
        self.len_dictionary = len_dictionary
        self.d_cut = d_cut
        self.Class_id = len_dictionary - 1
        self.Training = Training
        self.Rcs = Rcs
        self.RC = RC

    def read_box(self, filename):
        temp = float((filename.split('/data')[0]).split('_T')[-1])
        press= float((filename.split('_T')[0]).split('/P')[-1])
        data = np.load(filename)
        return data[0], data[1], data[2], data[3], data[4], 2*(data[5]-rho_ave)/(rho_max-rho_min), 2*(data[6]-pot_ave)/(pot_max-pot_min), data[7], data[8]
    ###########boxx,     boxy,    boxz,    temp,     press,   Rho,     Pot,     Enp,     Committor

    def read_npy_file(self, filename1,filename2,filename3):
        filename1 = str(filename1)
        filename2 = str(filename2)
        filename3 = str(filename3)
        data1 = np.load(filename1)
        data3 = np.load(filename3)
        boxx, boxy, boxz, temp, press, rho, pot, ent, pb = self.read_box(filename2) # simluation data
        old_coord = []
        BOXX = []
        BOXY = []
        BOXZ = []
        LOCLA_coord = []
        new_coord = data1
        Temperature = []
        Pressure = []
        Density = []
        Potential = []
        Entropy = []
        Committor = []
        #############################################
        old_coord.append(data3)
        BOXX.append(boxx)
        BOXY.append(boxy)
        BOXZ.append(boxz)
        LOCLA_coord.append(new_coord)
        Density.append(rho)
        Pressure.append(press)
        Temperature.append(temp)
        Potential.append(pot)
        Entropy.append(ent)
        Committor.append(pb)
        ##################################################
        old_coord=np.array(old_coord)
        BOXX=np.array(BOXX)
        BOXY=np.array(BOXY)
        BOXZ=np.array(BOXZ)
        LOCLA_coord=np.array(LOCLA_coord)
        Density=np.array(Density)
        Temperature=np.array(Temperature)
        Pressure=np.array(Pressure)
        Potential=np.array(Potential)
        Entropy=np.array(Entropy)
        Committor=np.array(Committor)
        data1 = None
        data3 = None
        boxx = None
        boxy = None
        boxz = None
        temp = None
        press = None
        rho = None
        pot = None
        ent = None
        pb = None
        return (LOCLA_coord.astype(np.float32),
                Density.astype(np.float32),
                Potential.astype(np.float32),
                Entropy.astype(np.float32),
                Committor.astype(np.float32),
                Temperature.astype(np.float32),
                Pressure.astype(np.float32),
                old_coord.astype(np.float32),
                BOXX.astype(np.float32),
                BOXY.astype(np.float32),
                BOXZ.astype(np.float32))

    def Generator_Dataset(self):
        idn = []
        while (len(idn)<self.max_batch_system):
            #tf.print("======total_path=====")
            #tf.print(self.total_path)
            path_id = random.randint(0,len(self.total_path)-1)
            if path_id not in idn:
                idn.append(path_id)
        for i in range(len(idn)):
            path = np.array(self.total_path[idn[i]])
            coord_path = np.array(path)
            sysin_path = np.char.replace(coord_path,'new_coord','box')
            old_c_path = np.char.replace(coord_path,'new_coord','coord')
            out = self.read_npy_file(coord_path,sysin_path,old_c_path)
            #tf.print(out[0][0].shape,',', out[1][0].shape,',', out[2][0].shape,',', out[3][0].shape,',', out[4][0].shape)
            yield out[0][0], out[1][0], out[2][0], out[3][0], out[4][0], out[5][0], out[6][0],  out[7][0], out[8][0], out[9][0], out[10][0]

    def Generator_test_Dataset(self,deal_filename,beigin,deta_k):
        path = np.array(self.total_path)
        deal_filename = str(deal_filename)
        #tf.print(path.shape)
        #tf.print(path.shape[0])
        #suf_path = '../../../dp_LDL/Dataset/P1800_T188/data1/new_coord'#####################
        #suf_path = '../../../dp_LDL/'+'BalanceDataset/P100_T235/data1'+'/new_coord'#####################
        suf_path = '../../../dp_LDL/'+deal_filename+'/new_coord'#####################
        for i in range(path.shape[0]):
            #filename = str((i*20000)+500000)                     ############################
            #filename = str((i*4000)+200)
            filename = str((i*deta_k)+beigin)
            coord_path = suf_path+'/'+filename+'.npy'
            tf.print(coord_path)
            coord_path = np.array(coord_path)
            sysin_path = np.char.replace(coord_path,'new_coord','box')
            old_c_path = np.char.replace(coord_path,'new_coord','coord')
            out = self.read_npy_file(coord_path,sysin_path,old_c_path)
            yield out[0][0], out[1][0], out[2][0], out[3][0], out[4][0], out[5][0], out[6][0],  out[7][0], out[8][0], out[9][0], out[10][0]

    def Generate_batch(self, batch_size, Repeat_size, shuffle_size):
        dataset = tf.data.Dataset.from_generator(
            self.Generator_Dataset,
            output_types=(tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32),
            output_shapes=((self.system_OatomN,self.local_atom_maxN-1,4),
                           (),
                           (),
                           (),
                           (),
                           (),
                           ())
        )
        #dataset = data.map(lambda x: tf.reshape(x,(-1,300,)))
        dataset = dataset.repeat(Repeat_size)
        dataset = dataset.shuffle(shuffle_size).batch(batch_size)
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        dataset = dataset.prefetch(AUTOTUNE)
        return dataset

    def Generate_test_batch(self,batch_size):
        dataset = tf.data.Dataset.from_generator(
            self.Generator_test_Dataset,
            output_types=(tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32),
            output_shapes=((self.system_OatomN,self.local_atom_maxN-1,4),
                           (),
                           (),
                           (),
                           (),
                           (),
                           (),
                           (self.system_OatomN,4),
                           (),(),()),
            args=("BalanceDataset/P100_T235/data1",200,4000)
        )
        dataset = dataset.batch(batch_size)
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        dataset = dataset.prefetch(AUTOTUNE)
        return dataset

class Embedding_descriptor(tf.keras.layers.Layer):
    def __init__(self, M1_model, M2_model, Natom):
        super(Embedding_descriptor, self).__init__()
        self.M1_model = M1_model
        self.M2_model = M2_model
        self.Natom = Natom
    def build(self, input_shape):
        self.WM1 = self.add_weight(shape=(self.Natom, self.M1_model),
                                  initializer='random_normal',
                                  trainable=True, name='WM1')
        self.WM2 = self.add_weight(shape=(self.Natom, self.M2_model),
                                  initializer='random_normal',
                                  trainable=True, name='WM2')
    def call(self, R):
        batch_size = tf.shape(R)[0]
        R_T = tf.transpose(R, perm=[0, 1, 3, 2])
        D_2 = tf.matmul(R_T,self.WM2)
        D_1 = tf.matmul(self.WM1, R, transpose_a=True)
        D = tf.matmul(D_1, D_2)
        return D

class encoder_ANN_rho(tf.keras.layers.Layer):
    def __init__(self, M1_model, M2_model, Natom, inner_dim, out_dim, dropout):
        super(encoder_ANN_rho, self).__init__()
        self.M1_model = M1_model
        self.M1_model = M1_model
        self.Natom = Natom
        self.inner_dim = inner_dim
        self.out_dim = out_dim
        #self.rho0 = rho0
        self.embedding = Embedding_descriptor(self.M1_model, self.M1_model, self.Natom)
        ################################################################################
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)
        self.dropout3 = tf.keras.layers.Dropout(dropout)
        #self.dropout4 = tf.keras.layers.Dropout(dropout)
        self.activation_fn1 = tf.keras.layers.Activation('gelu')
        self.activation_fn2 = tf.keras.layers.Activation('gelu')
        #self.activ ation_fn3 = tf.keras.layers.Activation('gelu')
        self.dense1 = tf.keras.layers.Dense(self.inner_dim)
        self.dense2 = tf.keras.layers.Dense(self.inner_dim)
        self.dense3 = tf.keras.layers.Dense(self.out_dim)
        ###############################################################################
        self.dropout11 = tf.keras.layers.Dropout(dropout)
        self.dropout12 = tf.keras.layers.Dropout(dropout)
        self.dropout13 = tf.keras.layers.Dropout(dropout)
        #self.dropout4 = tf.keras.layers.Dropout(dropout)
        self.activation_fn11 = tf.keras.layers.Activation('gelu')
        self.activation_fn12 = tf.keras.layers.Activation('gelu')
        #self.activ ation_fn3 = tf.keras.layers.Activation('gelu')
        self.dense11 = tf.keras.layers.Dense(self.inner_dim)
        self.dense12 = tf.keras.layers.Dense(self.inner_dim)
        self.dense13 = tf.keras.layers.Dense(self.out_dim)
        ###############################################################################
    def call(self, x, temp, press, training=True):
        bsz = tf.shape(x)[0]
        AtomN = tf.shape(x)[1]
        x = self.embedding(x)
        x = tf.reshape(x, (bsz,AtomN,-1))
        x = tf.reshape(x, (bsz*AtomN,-1))
        infor = self.dropout1(x,training=training)
        ################################################up embeding 
        x1= self.dense1(infor) # (bsz*AtomN, inner_dim)
        x = self.dropout2(x1,training=training)
        x = self.activation_fn1(x)
        x2= self.dense2(x) # (bsz*AtomN, inner_dim)
        x = self.dropout3(x2,training=training)
        x = self.activation_fn2(x)
        x3= self.dense3(x)
        x = tf.reshape(x3, (bsz,AtomN))
        rho = tf.reduce_mean(x,axis=-1)
        #tf.print(rho.shape)
        ###############################################up rho
        y1= self.dense11(infor)
        y = self.dropout12(y1,training=training)
        y = self.activation_fn11(y)
        y2= self.dense12(y) # (bsz*AtomN, inner_dim)
        y = self.dropout13(y2,training=training)
        y = self.activation_fn12(y)
        y3= self.dense13(y)
        y = tf.reshape(y3, (bsz,AtomN))
        pot = tf.reduce_sum(y,axis=-1)
        return rho,x3,x2,x1, pot,y3,y2,y1

class non_linear(tf.keras.layers.Layer):
    def __init__(self, inner_dim, out_dim):
        super(non_linear, self).__init__()
        self.inner_dim = inner_dim
        self.out_dim = out_dim
        self.dense1 = tf.keras.layers.Dense(self.inner_dim)
        self.dense2 = tf.keras.layers.Dense(self.inner_dim)
        self.dense3 = tf.keras.layers.Dense(self.out_dim)
        self.activation_fn1 = tf.keras.layers.Activation('gelu')
        self.activation_fn2 = tf.keras.layers.Activation('gelu')
        #self.dropout1 = tf.keras.layers.Dropout(dropout)
    def call(self, x, training=True):
        x = self.dense1(x)
        #x = self.activation_fn1(x)
        x = self.dense2(x)
        #x = self.activation_fn2(x)
        x = self.dense3(x)
        return x

class decoder_ANN_rho(tf.keras.layers.Layer):
    def __init__(self, inner_dim, out_dim, atomON, dropout):
        super(decoder_ANN_rho, self).__init__()
        self.inner_dim = inner_dim
        self.out_dim = out_dim
        self.atomON = atomON
        self.mid_dim = int(256/2)
        self.dropout1 = tf.keras.layers.Dropout(dropout)        #
        self.dropout11= tf.keras.layers.Dropout(dropout)        #
        self.dropout2 = tf.keras.layers.Dropout(dropout)        #
        self.dense00 = tf.keras.layers.Dense(self.inner_dim, kernel_regularizer=tf.keras.regularizers.L2(0.01), bias_regularizer=tf.keras.regularizers.L2(0.01))    #
        self.dense01 = tf.keras.layers.Dense(self.inner_dim, kernel_regularizer=tf.keras.regularizers.L2(0.01), bias_regularizer=tf.keras.regularizers.L2(0.01))    #
        self.dropout00 = tf.keras.layers.Dropout(dropout)       #
        self.dropout01 = tf.keras.layers.Dropout(dropout)        #
        self.activation_fn00 = tf.keras.layers.Activation('gelu')#
        self.activation_fn01 = tf.keras.layers.Activation('gelu')#
        self.dense1 = tf.keras.layers.Dense(self.inner_dim, kernel_regularizer=tf.keras.regularizers.L2(0.01), bias_regularizer=tf.keras.regularizers.L2(0.01))     #
        self.dense11 = tf.keras.layers.Dense(self.inner_dim, kernel_regularizer=tf.keras.regularizers.L2(0.01), bias_regularizer=tf.keras.regularizers.L2(0.01))    #
        self.double_inner_dim = int(self.inner_dim * 2)
        self.dense2 = tf.keras.layers.Dense(self.double_inner_dim, kernel_regularizer=tf.keras.regularizers.L2(0.01), bias_regularizer=tf.keras.regularizers.L2(0.01))       #
        self.dense12 = tf.keras.layers.Dense(self.double_inner_dim, kernel_regularizer=tf.keras.regularizers.L2(0.01), bias_regularizer=tf.keras.regularizers.L2(0.01))      #
        self.dense3 = tf.keras.layers.Dense(self.out_dim, kernel_regularizer=tf.keras.regularizers.L2(0.01), bias_regularizer=tf.keras.regularizers.L2(0.01))
        ##########################################################
        #self.temp_embed = tf.keras.layers.Dense(self.mid_dim, kernel_regularizer=tf.keras.regularizers.L2(0.01), bias_regularizer=tf.keras.regularizers.L2(0.01))   # 
        #self.press_embed = tf.keras.layers.Dense(self.mid_dim, kernel_regularizer=tf.keras.regularizers.L2(0.01), bias_regularizer=tf.keras.regularizers.L2(0.01))  # 
        #self.dense4 = tf.keras.layers.Dense(self.inner_dim, kernel_regularizer=tf.keras.regularizers.L2(0.01), bias_regularizer=tf.keras.regularizers.L2(0.01))     # 
        #self.dense44 = tf.keras.layers.Dense(self.inner_dim, kernel_regularizer=tf.keras.regularizers.L2(0.01), bias_regularizer=tf.keras.regularizers.L2(0.01))   
        ##########################################################
        self.activation_fn1 = tf.keras.layers.Activation('gelu') #
        self.activation_fn11= tf.keras.layers.Activation('gelu') #
        self.activation_fn2 = tf.keras.layers.Activation('gelu') #
        self.activation_fnv = tf.keras.layers.Activation('gelu') #
        self.dropoutv = tf.keras.layers.Dropout(dropout)         #
        self.dropout44 = tf.keras.layers.Dropout(dropout)
        self.activation44 = tf.keras.layers.Activation('gelu')
        self.dropout_12 = tf.keras.layers.Dropout(dropout)
        self.activation_12 = tf.keras.layers.Activation('gelu')
        #self.dense_a = tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.L2(0.01), bias_regularizer=tf.keras.regularizers.L2(0.01))                 #
        #self.dense_aa = tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.L2(0.01), bias_regularizer=tf.keras.regularizers.L2(0.01))                #
        #self.dense_b = tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.L2(0.01), bias_regularizer=tf.keras.regularizers.L2(0.01))                 #
    def call(self, x3, x2, x1, y3,y2, y1, temp, press, training=True):
        #bsz = tf.shape(x3)[0] # (bsz*AtomN,1)
        bsz = tf.shape(temp)[0]
#        temp = tf.reshape(temp,(bsz,1))
#        press= tf.reshape(press,(bsz,1))
#        valueT = self.temp_embed(temp)
#        valueP = self.press_embed(press)
#        value  = tf.concat([valueT,valueP],axis=-1)
#        #
#        #tf.print('value1.shape',value.shape)
#        value  = self.dropout2(value, training=training)
#        value  = self.activation_fn2(value)
#        value  = self.dense4(value)
#        #tf.print('value2.shape',value.shape)
#        value  = self.dropoutv(value, training=training)
#        value  = self.activation_fnv(value)
#        #value  = tf.tile(tf.expand_dims(value, 1),[1,self.atomON,1])
#        #value  = tf.reshape(value,(bsz*self.atomON,-1))
#        #
#        #tf.print('value.shape',value.shape,'v:',value)
#        #a = tf.keras.activations.tanh(self.dense_a(value)) #+ 1e-9
#        aa= (self.dense_aa(value)) #+ 1e-9
#        b = self.dense_b(value)
#        #tf.print('a,aa,b.shape',a.shape,aa.shape,b.shape)
#        #a = tf.tile(tf.expand_dims(a, 1),[1,self.atomON,1])
#        aa= tf.tile(tf.expand_dims(aa,1),[1,self.atomON,1])
#        b = tf.tile(tf.expand_dims(b, 1),[1,self.atomON,1])
#        #a = tf.reshape(a,(bsz*self.atomON,-1))
#        aa= tf.reshape(aa,(bsz*self.atomON,-1))
#        b = tf.reshape(b,(bsz*self.atomON,-1))
        #tf.print('a,aa,b.shape',a.shape,aa.shape,b.shape)
        #tf.print('a:',a)
        #tf.print('aa:',aa)
        #tf.print('b:',b)
        #
        x = self.dense1(x3) + x1 #+ value
        y = self.dense11(y3)+ y1
        x = self.dropout1(x, training=training)
        x = self.activation_fn1(x)
        y = self.dropout11(y, training=training)
        y = self.activation_fn11(y)
        x = self.dense00(x)+ x2
        x = self.dropout00(x, training=training)
        x = self.activation_fn00(x)
        y = self.dense01(y)+ y2
        y = self.dropout01(y, training=training)
        y = self.activation_fn01(y)
        #
        x4= tf.concat([x,y],axis=-1)
        x4= self.dense2(x4)#+x3
        x4= self.dropout44(x4, training=training)
        x4= self.activation44(x4)
        x4= self.dense12(x4)#+y3
        x4= self.dropout_12(x4, training=training)
        x4= self.activation_12(x4)
        x4= self.dense3(x4)
        #x4= tf.concat([x4,y4],axis=-1)
        #
        normal_x3 = x3 #
        normal_y3 = tf.cast(self.atomON,dtype=tf.float32)*y3
        #x0 = (normal_x3*1)  +  (normal_y3*a)#+ c3
        #x0 = x0/(tf.math.sqrt((a*a)+(1*1)))
        #tf.print('x0,shape',x0.shape)
        #tf.print(x0)
        #tf.print('x4,shape',x0.shape)
        #tf.print(x4)
        #noise = tf.random.normal(shape=tf.shape(x3), mean=0.0, stddev=1.5)
        #tf.print(noise)
        return x3,y3,x4, normal_x3,normal_y3#+noise           #####################    

model_encoder=encoder_ANN_rho(100,4,30,250,1,0.1)
model_decoder=decoder_ANN_rho(250,1,300,0.1)

def numpy_rankdata(x):
    return rankdata(x).astype(np.float32)

def spearman_rank_correlation(x, y):
    x_rank = tf.numpy_function(numpy_rankdata,[x],tf.float32)
    y_rank = tf.numpy_function(numpy_rankdata,[y],tf.float32)
    d = x_rank - y_rank
    d_squared = d**2 #np.square(d)
    n = x.shape[0]
    rho = 1 - (6 * tf.reduce_sum(d_squared)) / (n * (n**2 - 1))
    return rho

def loss_function_1(pred_rho,real_rho,pred_pot,real_pot,normal_y3,x4,x3):
    loss_rho = tf.keras.losses.MSE(pred_rho,real_rho)
    loss_pot = tf.keras.losses.MSE(pred_pot,real_pot)
    loss1 = loss_rho + loss_pot #+ linear_axb 
    #############################################
    tf.print('x3.shape, x4.shape',x3.shape,x4.shape)
    correlation = ( tf.reduce_sum((x4-tf.reduce_mean(x4))*(x3-tf.reduce_mean(x3)) ) )
    tf.print('correlation.shape',correlation.shape)
    correlation = correlation/( tf.math.sqrt( tf.reduce_sum((x3-tf.reduce_mean(x3))**2) )* tf.math.sqrt( tf.reduce_sum((x4-tf.reduce_mean(x4))**2) ) )#, 1e-9)
    tf.print('correlation.shape',correlation.shape)
    spearman_X = tf.reshape(x3, (-1,))
    spearman_Y = tf.reshape(x4, (-1,))
    #spearman_X = spearman_X.numpy()
    #spearman_Y = spearman_Y.numpy()
    spearman_cor = spearman_rank_correlation(spearman_X, spearman_Y)
    #
    loss_correlation = (tf.reduce_mean((correlation-0.50)**2) )
    loss_spearman_cor = (tf.reduce_mean((spearman_cor-0.50)**2) )
    loss_correlation = tf.cast(loss_correlation,dtype=tf.float32) + tf.cast(loss_spearman_cor,dtype=tf.float32)
    x3 = tf.reshape(x3,(-1,300,1))
    x4 = tf.reshape(x4,(-1,300,1))
    bsz= x4.shape[0]
    k_cor = correlation*(( tf.math.sqrt( tf.reduce_sum((x4-tf.reshape(tf.reduce_mean(x4,axis=1),(bsz,1,1)))**2,axis=1) ) )/( tf.math.sqrt( tf.reduce_sum((x3-tf.reshape(tf.reduce_mean(x3,axis=1),(bsz,1,1)))**2, axis=1) ) ))
    tf.print('k_cor',k_cor.shape)
    #
    aa_loss1 = tf.reduce_mean( (k_cor - tf.math.tan(499.999*pi/1000.0))**2 )  #aa_loss1 + aa_loss2
    #aa_loss2 = 1*tf.reduce_mean( tf.math.minimum(-k_cor,-1))
    aa_loss = aa_loss1 # + aa_loss2
    #k1 = tf.math.abs(1-loss_correlation)
    #k2 = *tf.math.abs(9-aa_loss)
    #ka1 = k2/(k1+k2)
    #ka2 = k1/(k1+k2)
    #tf.print(ka1, ka2)
    tf.print(loss_correlation,aa_loss)
    tf.print(tf.reduce_mean(correlation), tf.reduce_mean(spearman_cor), tf.reduce_mean(k_cor))
    tf.print("======")
    #
    loss2 = (aa_loss) + loss_correlation*100 #+ (1/loss_std) +(1/tf.math.reduce_std(x3))  +(1/tf.math.reduce_std(normal_y3)) #+ loss2_mean
    return loss1, loss2, loss_rho, loss_pot ,  aa_loss , tf.reduce_mean(k_cor*1), loss_correlation*100

optimizer1 = tf.keras.optimizers.Adam(0.0001) #0.00001
optimizer2 = tf.keras.optimizers.Adam(0.0001) #0.0001
@tf.function
def train_step(coord,rho,pot,temp,press):
    filename = './log_simple_fhi47_100_linear/train.log'
    flt = open(filename,'a')
    with tf.GradientTape() as encoder_tape, tf.GradientTape() as decoder_tape:
        Prho,x3,x2,x1, Ppot, y3,y2,y1 = model_encoder(coord, temp, press, training=True)
        x3,y3,x4, normal_x3,normal_y3 = model_decoder(x3,x2,x1, y3,y2,y1, temp,press,training=True)
        loss1, loss2, loss_rho, loss_pot, aa_loss,linear_aax4,loss_correlation = loss_function_1(Prho,rho, Ppot,pot, normal_y3,x4,x3)
        tf.print('loss_rho: ',loss_rho,',loss_pot: ', loss_pot,'aa_loss',aa_loss,'linear_aax4:',linear_aax4,'loss_correlation:',loss_correlation,output_stream = 'file://'+flt.name)
    gradients_encoder = encoder_tape.gradient(loss1, model_encoder.trainable_variables)
    gradients_decoder = decoder_tape.gradient(loss2, model_decoder.trainable_variables)
    optimizer1.apply_gradients(zip(gradients_encoder, model_encoder.trainable_variables))
    optimizer2.apply_gradients(zip(gradients_decoder, model_decoder.trainable_variables))
    #train_rho_loss(loss1)
    #train_landscape_loss(loss_landscape)
    #train_loss_std(loss_std)
    #train_xe(pred_xe)
    #train_mean(loss2_mean)
    flt.close()

@tf.function
def test_step(coord,rho,pot,temp,press,epoch,oldc,boxx,boxy,boxz):
    filename = './logtest/test.log'
    fle = open(filename,'a')
    Prho,x3,x2,x1, Ppot,y3,y2,y1 = model_encoder(coord, temp, press, training=True)
    x3,y3,x4, normal_x3,normal_y3 = model_decoder(x3,x2,x1, y3,y2,y1, temp,press,training=True)
    loss1, loss2, loss_rho, loss_pot, aa_loss,linear_aax4,loss_correlation = loss_function_1(Prho,rho, Ppot,pot, normal_y3,x4,x3)
    tf.print('loss_rho: ',loss_rho,',loss_pot: ', loss_pot,'aa_loss',aa_loss,'linear_aax4',linear_aax4,'loss_correlation:',loss_correlation,"real_rho:", tf.reduce_mean(rho), ", pred_rho:", tf.reduce_mean(Prho), "real_pot:", tf.reduce_mean(pot), ", pred_pot:", tf.reduce_mean(Ppot),output_stream = 'file://'+fle.name)
    #pathname = './'
    #filename_x = 't.x.log'
    #filename_x3= 't.x3.log'
    #filename_x = pathname + str(epoch) + filename_x
    #filename_x3= pathname + str(epoch) + filename_x3
    #fx = open(filename_x ,'w')
    #fx3= open(filename_x3,'w')
    filename1 = './logtest/xe.log'
    flx = open(filename1,'a')
    tf.print("ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n300\nITEM: BOX BOUNDS xy xz yz pp pp pp",output_stream = 'file://'+flx.name)
    tf.print("0 ", boxx*10," 0",output_stream = 'file://'+flx.name)
    tf.print("0 ", boxy*10," 0",output_stream = 'file://'+flx.name)
    tf.print("0 ", boxz*10," 0",output_stream = 'file://'+flx.name)
    tf.print("ITEM: ATOMS id type x y z label rho rx4",output_stream = 'file://'+flx.name)
    oldc = tf.reshape(oldc,(300,4))
    sequence = tf.range(1,301, dtype=tf.float32)
    sequence = tf.reshape(sequence, (300,1))
    sequence = tf.floor(sequence)
    linx = None
    linx3= None
    linx4= None
    lint = None
    linx = y3
    linx3= x3
    linx4 = x4
    lin = tf.concat([linx,linx3],axis=-1)
    lin2 =tf.concat([sequence,oldc],axis=-1)
    lint =tf.concat([lin2,lin],axis=-1)
    lint =tf.concat([lint,linx4],axis=-1)
    tf.print(lint,summarize=300,output_stream = 'file://'+flx.name)
    #tf.print("===================================", output_stream = 'file://'+flx.name)
    #tf.print(x,summarize=300,output_stream = 'file://'+flx.name)
    #test_rho_loss(loss1)
    #test_landscape_loss(loss_landscape)
    #test_loss_std(loss_std)
    #test_xe(pred_xe)
    #test_mean(loss2_mean)
    fle.close()
    flx.close()

#for layer in model_encoder.layers:
#    if layer.name is None:
#        layer._name = f"encoder_layer_{layer.name}"
#
#for layer in model_decoder.layers:
#    if layer.name is None:
#        layer._name = f"decoder_layer_{layer.name}"

checkpoint_path = "./checkpoint_LDL_vsimple_fhi47_100_linear/cpkt"
ckpt = tf.train.Checkpoint(model_encoder=model_encoder, model_decoder=model_decoder)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

model_encoder.trainable=False
model_decoder.trainable=False

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print ('Latest checkpoint restored!!')

for epoch in range(1):
    #start = time.time()
    #train_rho_loss.reset_states()
    #train_landscape_loss.reset_states()
    #train_loss_std.reset_states()
    #train_xe.reset_states()
    #train_mean.reset_states()
    ###
    #test_rho_loss.reset_states()
    #test_landscape_loss.reset_states()
    #test_loss_std.reset_states()
    #test_xe.reset_states()
    #test_mean.reset_states()
    ###
    #data_Feeder = LDL_Data_Feeder(
    #    path, max_batch_system = 1000, local_atom_maxN = 31, system_OatomN = 300
    #)
    #batch_dataset = data_Feeder.Generate_batch(batch_size=100,Repeat_size=1, shuffle_size=1)
    #for (batch,(coord,rho,pot,ent,pb,temp,press)) in enumerate(batch_dataset):
    #    train_step(coord,rho,pot,temp,press)
    #    tf.print("EPOCH: ",epoch," loss_rho: ", train_rho_loss.result(), " loss_landscape: ",train_landscape_loss.result(),
    #             " loss_std: ", train_loss_std.result(), " xe: ", train_xe.result(), " train_mean: ",train_mean.result(),
    #             output_stream = 'file://'+filename_train.name)
    #ckpt_save_path = ckpt_manager.save()
    test_Feeder = LDL_Data_Feeder(
        path, max_batch_system = 1, local_atom_maxN = 31, system_OatomN = 300
    )
    test_dataset = test_Feeder.Generate_test_batch(1)
    for (batch,(coord,rho,pot,ent,pb,temp,press,oldc,boxx,boxy,boxz)) in enumerate(test_dataset):
        test_step(coord,rho,pot,temp,press,epoch,oldc,boxx,boxy,boxz)
    #    tf.print("EPOCH: ",epoch," loss_rho: ",test_rho_loss.result()," loss_landscape: ",test_landscape_loss.result(),
    #             " loss_std: ",test_loss_std.result()," xe: ",test_xe.result()," train_mean: ",test_mean.result(),
    #             output_stream = 'file://'+filename_test.name)


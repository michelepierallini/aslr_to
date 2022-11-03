import os
import sys
from operator import delitem
from pickletools import float8
from typing_extensions import Self
import crocoddyl
import pinocchio
import numpy as np
import example_robot_data
import aslr_to
import math
import time
from scipy.io import savemat
from numpy import linalg as LA
import pathlib
from pinocchio.visualize import GepettoVisualizer
# seed the pseudorandom number generator
from numpy.random import seed
from numpy.random import rand
import matplotlib.pyplot as plt
params = {'text.latex.preamble' : [r'\usepackage{siunitx}', r'\usepackage{amsmath}']}
plt.rcParams.update(params)
import csv
import shutil
from ILC_OnlyPinocchio import ILC_OnlyPinocchio

dt = 1e-4
r = 2
m = 1
q0 = np.zeros(21)
q0[0] = np.pi/2
T = 10000
max_step = T
T_f = T * dt
t = np.arange(0,T*dt, dt)
FISHING_STRING = 'fishing_rod_two'

k_ii = 1.3 * np.array([0,34.6099,26.8363,17.2029,11.8908,8.9892,\
        12.6067,8.8801,4.0401,3.6465,3.0458,6.3951,3.6361,2.6307,\
        3.0194,1.1336,1.5959,1.3670,0.7066,0.5041,0.412])

d_ii = 3e1 * np.array([0.0191,0.0164,0.0127,0.0082,0.0056,0.0043,\
        0.0060,0.0042,0.0019,0.0017,0.0014,0.0030, 0.0017,0.0012,0.0014,\
        0.0005, 0.0008, 0.0006,0.0003,0.0002,0.0000])

directory = 'ILC_DATA_{}_{}_5'.format(max_step, dt)
dir_plot_ddp = './' + directory + '/DDP_/'
U_DDP = dir_plot_ddp + 'u_ddp.csv'
POS_X = dir_plot_ddp + 'y_des.csv'
POS_Y = dir_plot_ddp + 'pos_y.csv'
POS_Z = dir_plot_ddp + 'pos_z.csv'
VEL_X = dir_plot_ddp + 'y_dot_des.csv'
ACC_X = dir_plot_ddp + 'y_ddot_des.csv'
# K_FB_DDP = dir_plot_ddp + 'k_fb.csv'

target_pos = np.array([3.13679016, 0, 0.96838924]) # np.array([3.06556153, 0, 1.07999153]) #np.loadtxt(TARGET)
y_des = np.loadtxt(POS_X, delimiter=',')
pos_y = np.loadtxt(POS_Y, delimiter=',')
pos_z = np.loadtxt(POS_Z, delimiter=',')
y_dot_des = np.loadtxt(VEL_X, delimiter=',')
y_ddot_des = np.loadtxt(ACC_X, delimiter=',')
# Gamma_fb = np.loadtxt(K_FB_DDP, delimiter=',')
u0 = np.loadtxt(U_DDP, delimiter=',')

ilc = ILC_OnlyPinocchio(q0,
                k_ii, 
                d_ii, 
                m, 
                r, 
                u0, 
                target_pos, 
                y_des, 
                y_dot_des, 
                y_ddot_des, 
                dt, 
                FISHING_STRING, 
                max_step, 
                t, 
                pos_y,
                pos_z, 
                directory)

ilc.main()

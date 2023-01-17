import os
import sys
from operator import delitem
from pickletools import float8
from typing_extensions import Self
from sympy import QQ_gmpy

from torch import sgn
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
from termcolor import colored
import csv
import shutil

SAVE_ALL = False

class ILC_OnlyPinocchioFishing():
    """_summary_
    This class manages to deal with SISO system.
    One could use a cartesian coordinate or position of the motor q_i 
    In order to do so, one must change the definition of:
        -) self.y_des
        -) self.y_dot_des
        -) self.y_ddot_des
    """ 
    
    ###################
    ################### ToDo fare un keyboard exception which save all I want
    def __init__(self, 
                q0,
                k_ii,
                d_ii,
                m,
                r,
                u0,  # torque control 
                target_pose, 
                y_des,
                y_dot_des,
                y_ddot_des, 
                dt, 
                FISHING_STRING, 
                max_step,
                time_span, 
                pos_x,
                pos_y,
                pos_z, 
                q1, # position control 
                directory, 
                control_mode = 'Cartesian'):
        print('=================================================================')
        # ToDo fix this path 
        self.path_current = pathlib.Path().resolve()
        ########################### This to modify the urdf a little bit 
        # self.urdf_file = str(self.path_current) + '/fishing_rod_description2/urdf/fishing_rod2.urdf'
        
        # self.urdf_file = str(self.path_current) + '/fishing_rod_description2_hat/urdf/fishing_rod2.urdf'
        # self.mesh_path = str(self.path_current) + '/fishing_rod_description2/meshes/'
        
        self.urdf_file = str(self.path_current) + '/fishing_rod_description4_hat/urdf/fishing_rod4.urdf'
        self.mesh_path = str(self.path_current) + '/fishing_rod_description4/meshes/'
        self.task = control_mode
        # self.task = 'Joint'
        self.robot = pinocchio.buildModelFromUrdf(self.urdf_file) # example_robot_data.load('fishing_rod')
        self.robot_display = example_robot_data.load(FISHING_STRING)
        # self.cameraTF = [2., 2.68, 0.54, 0.2, 0.62, 0.72, 0.22]
        # self.cameraTF = [5.2, 5.2, 1.8, 0.2, 0.62, 0.72, 0.22]
        self.cameraTF = [5.4, 5.4, 2.0, 0.2, 0.62, 0.72, 0.22]
        self.data = self.robot.createData()
        self.robot.gravity.linear = np.array([0, 0, -9.81]) # w.r.t. global frame
        self.nq = self.robot.nv
        self.target_pos = target_pose
        self.data.xout = []
        self.data.q = []
        self.data.dq = []
        self.data.pos = []
        self.data.vel = []
        self.data.acc = []
        self.q_vett = []
        self.q_dot_vett = []
        self.pos_vett = []
        self.vel_vett = []
        self.acc_vett = []
        self.u_fb_vett = []
        self.control_action_vett = []
        self.uncer_mag = 0.5 # 0.2
        self.delta_k_ii = k_ii * self.uncer_mag # np.linalg.norm(k_ii,2) * np.ones(self.nq)/200
        self.delta_d_ii = d_ii * self.uncer_mag # self.uncer_mag * np.linalg.norm(d_ii,2) * np.ones(self.nq)
        print('|| \Delta K ||: {}\n|| \Delta D ||: {}'.format(np.linalg.norm(self.delta_k_ii), np.linalg.norm(self.delta_d_ii)))
        # self.k_ii = k_ii - self.delta_k_ii
        # self.d_ii = d_ii - self.delta_d_ii
        self.k_ii = k_ii + self.delta_k_ii
        self.d_ii = d_ii + self.delta_d_ii

        self.K = np.diag(self.k_ii)
        self.D = np.diag(self.d_ii)
        # D = np.diag(d_ii) + np.diag(d_ii[:-1]/2e1, k=-1) + np.diag(d_ii[:-1]/2e1, k=1)  # not implemented in the class
        # K = np.diag(k_ii) + np.diag(k_ii[:-1]/2e1, k=-1) + np.diag(k_ii[:-1]/2e1, k=1)  # not implemented in the class 
        self.m = m
        # self.m = self.robot.nu
        self.underactuation_matrix = np.zeros(self.nq)
        self.nu = m # number of actuators
        self.actuation_index = 0
        self.underactuation_matrix[self.actuation_index] = 1
        # for i in range(0,self.nu - 1):
        #     self.underactuation_matrix[i] = 1

        self.dt = dt
        self.T = dt * max_step
        self.steps = max_step #int(self.T/self.dt)
        self.time = time_span
        self.u_new = np.zeros(self.steps)
        self.u_per_iter_ff = np.zeros(self.steps)
        self.err = np.zeros(self.steps)
        self.u0_guess = u0

        # plt.figure(clear=True)
        # plt.ylabel(r'Control Action $[Nm]$')
        # plt.xlabel(r'Time $[s]$')
        # plt.plot(self.time, self.u0_guess, color='green', linestyle='-',linewidth=2)
        # plt.grid()
        # plt.show()
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.pos_z = pos_z
        
        self.q1 = q1
    
        self.y_des = y_des
        self.y_dot_des = y_dot_des
        self.y_ddot_des = y_ddot_des
        self.FINISH_LEARNING = False
        self.directory = directory
        self.max_actuator = 200 # 300

        self.q0 = q0
        self.dq0 = np.zeros(self.robot.nv)
        self.max_iter = 200 
        self.toll = 0.015 # 0.04
        self.r = r
        self.conta_incr = 0
        self.sign_u = 1
        self.Gamma_fb = 0.1 #-0.5 # 0.01
        self.Gamma_ff = -0.5 # fishing2 is +0.03 
        # THE SIGN OF Gamma_ff DEPENDS ON THE SIGN OF THE ERR, IF THE 
        # INITIAL GUESS IS ABOVE OR UNDER THE REFERENCE
        
        # ff gains 
        self.gamma0 = self.sign_u * 0.1 #5
        self.gamma1 = self.sign_u * 0.001
        self.gamma2 = self.sign_u * 0.0001
        # fb gains frank 
        self.K_P = self.sign_u * 0.1
        self.K_V = self.sign_u * 0.001
        self.K_A = self.sign_u * 0.0001
        self.error2print = np.zeros(self.max_iter)
        self.err_rms = np.zeros(self.max_iter)
        self.error_in_iter = np.zeros(self.steps)
        self.err_t_f = np.zeros(self.max_iter)
        # self.gamma = np.zeros(r)
        # for i in range(0,self.r):
        #     self.gamma[i] = i/r
        # seed(1)
        self.getFolderPlot()

        self.dir_plot_ddp = './' + self.directory + '/DDP_/'
        if os.path.exists(self.dir_plot_ddp):
            shutil.rmtree(self.dir_plot_ddp)
        os.makedirs(self.dir_plot_ddp)
        
        u_ddp = np.array(self.u0_guess)
        y_des = np.array(self.y_des)
        y_dot_des = np.array(self.y_dot_des)
        y_ddot_des = np.array(self.y_ddot_des)
        q1 = np.array(self.q1)
        pos_x = np.array(self.pos_x)
        pos_y = np.array(self.pos_y)
        pos_z = np.array(self.pos_z)
        target = np.array(self.target_pos)
        # Gamma_fb = np.array(self.Gamma_fb) # to add as input 
        U_DDP = self.dir_plot_ddp + 'u_ddp.csv'
        POS_X = self.dir_plot_ddp + 'pos_x.csv'
        POS_Y = self.dir_plot_ddp + 'pos_y.csv'
        POS_Z = self.dir_plot_ddp + 'pos_z.csv'
        VEL_X = self.dir_plot_ddp + 'y_dot_des.csv'
        ACC_X = self.dir_plot_ddp + 'y_ddot_des.csv'
        TARGET = self.dir_plot_ddp + 'target.csv'
        Y_DES = self.dir_plot_ddp + 'y_des.csv'
        Y_DOT_DES = self.dir_plot_ddp + 'y_dot_des.csv'
        Y_DDOT_DES = self.dir_plot_ddp + 'y_ddot_des.csv'
        Q1 = self.dir_plot_ddp + 'q1.csv'
        # K_FB_DDP = self.dir_plot_ddp + 'k_fb.csv'
        
        y_des.tofile(Y_DES, sep=',')
        y_dot_des.tofile(Y_DOT_DES, sep=',')
        y_ddot_des.tofile(Y_DDOT_DES, sep=',')
        
        pos_x.tofile(POS_X, sep=',')
        pos_y.tofile(POS_Y, sep=',')
        pos_z.tofile(POS_Z, sep=',')
        y_dot_des.tofile(VEL_X, sep=',')
        y_ddot_des.tofile(ACC_X, sep=',')
        u_ddp.tofile(U_DDP, sep=',')
        target.tofile(TARGET)
        q1.tofile(Q1, sep=',')
        # Gamma_fb.tofile(K_FB_DDP, sep='')
        t = np.arange(0,max_step * dt,self.dt)

        plt.figure(clear=True)
        plt.ylabel(r'References')
        plt.xlabel(r'Time $[s]$')
        plt.plot(t, y_des, color='red',label='pos', linestyle='-',linewidth=2)
        # plt.plot(t, vel_x, color='green',label='vel', linestyle='-',linewidth=2)
        # plt.plot(t, acc_x, label='acc')
        # plt.legend(loc='best')
        plt.grid()
        plt.savefig(self.dir_plot_ddp + 'desired_pos.svg', format='svg')
        plt.close()

        plt.figure(clear=True)
        plt.ylabel(r'Control Action $[Nm]$')
        plt.xlabel(r'Time $[s]$')
        plt.plot(t, u_ddp, color='green', linestyle='-',linewidth=2)
        plt.grid()
        plt.savefig(self.dir_plot_ddp + 'action.svg', format='svg')
        plt.close()
        time.sleep(0.1)

    def getFolderPlot(self):
        if os.path.exists(self.directory):
            shutil.rmtree(self.directory)
        os.makedirs(self.directory)
    
    @staticmethod
    def getFinalPoint(X_tip, Z_tip, v_x, v_z, t_f):
        X_d = X_tip + v_x * t_f
        Z_d = Z_tip + v_z * t_f - 0.5 * 9.81 * t_f**2 
        return X_d, Z_d

    def saveData(self, iteration, error, pos_x, vel_x, acc_x, pos_z, vel_z, acc_z, action_ff, action_fb, u_per_iter_ff, q, q_dot):
        self.dir_plot = './' + self.directory + '/Iter_hat_{}/'.format(iteration)
        if os.path.exists(self.dir_plot):
            shutil.rmtree(self.dir_plot)
        os.makedirs(self.dir_plot)
        
        error = np.array(error)
        pos_x = np.array(pos_x)
        vel_x = np.array(vel_x)
        acc_x = np.array(acc_x)
        pos_z = np.array(pos_z)
        vel_z = np.array(vel_z)
        acc_z = np.array(acc_z)
        q = np.array(q)
        q_dot = np.array(q_dot)
        # print('type ff {}\ntype fb{}'.format(type(action_ff),type(action_fb)))
        action_ff = np.asarray(action_ff)
        action_fb = np.asarray(action_fb)
        u_per_iter_ff = np.array(u_per_iter_ff)
        ERROR = self.dir_plot + 'error.csv'
        POS_X = self.dir_plot + 'pos_X.csv'
        VEL_X = self.dir_plot + 'vel_X.csv'
        ACC_X = self.dir_plot + 'acc_X.csv'
        POS_Z = self.dir_plot + 'pos_Z.csv'
        VEL_Z = self.dir_plot + 'vel_Z.csv'
        ACC_Z = self.dir_plot + 'acc_Z.csv'
        ACTION_FF = self.dir_plot + 'action_ff.csv'
        ACTION_FB = self.dir_plot + 'action_fb.csv'
        ACTION_NEW_PER_ITER = self.dir_plot + 'u_per_iter_ff.csv'
        # Q = self.dir_plot + 'q.csv'
        Q_1 = self.dir_plot + 'q.csv'
        # Q_DOT = self.dir_plot + 'q_dot.csv'

        error.tofile(ERROR, sep=',')
        pos_x.tofile(POS_X, sep=',')
        vel_x.tofile(VEL_X, sep=',')
        acc_x.tofile(ACC_X, sep=',')
        pos_z.tofile(POS_Z, sep=',')
        vel_z.tofile(VEL_Z, sep=',')
        acc_z.tofile(ACC_Z, sep=',')
        action_ff.tofile(ACTION_FF, sep=',')
        action_fb.tofile(ACTION_FB, sep=',')
        u_per_iter_ff.tofile(ACTION_NEW_PER_ITER, sep=',')
        np.savetxt(Q_1, q[:,0])
        # np.savetxt(Q_DOT, q_dot)
    
    def savePlot(self, iteration):
        dir_plot = './' + self.directory + '/Iter_hat_{}/'.format(iteration)
        if os.path.exists(dir_plot + "plot/"):
            shutil.rmtree(dir_plot + "plot/")
        os.makedirs(dir_plot + "plot/")

        ERROR = dir_plot + 'error.csv'
        POS_X = dir_plot + 'pos_X.csv'
        VEL_X = dir_plot + 'vel_X.csv'
        ACC_X = dir_plot + 'acc_X.csv'
        POS_Z = dir_plot + 'pos_Z.csv'
        VEL_Z = dir_plot + 'vel_Z.csv'
        ACC_Z = dir_plot + 'acc_Z.csv'
        ACTION_FF = dir_plot + 'action_ff.csv'
        ACTION_FB = dir_plot + 'action_fb.csv'
        ACTION_NEW_PER_ITER = dir_plot + 'u_per_iter_ff.csv'
        Q = self.dir_plot + 'q.csv'
        Q_DOT = self.dir_plot + 'q_dot.csv'
        
        err = np.loadtxt(ERROR, delimiter=',')
        pos_x = np.loadtxt(POS_X, delimiter=',')
        vel_x = np.loadtxt(VEL_X, delimiter=',')
        acc_x = np.loadtxt(ACC_X, delimiter=',')
        pos_z = np.loadtxt(POS_Z, delimiter=',')
        vel_z = np.loadtxt(VEL_Z, delimiter=',')
        acc_z = np.loadtxt(ACC_Z, delimiter=',')
        action_ff = np.loadtxt(ACTION_FF, delimiter=',')
        action_fb = np.loadtxt(ACTION_FB, delimiter=',')
        u_per_iter_ff = np.loadtxt(ACTION_NEW_PER_ITER, delimiter=',')
        # q = np.loadtxt(Q, delimiter=',')
        # q_dot = np.loadtxt(Q_DOT, delimiter=',')    

        action = action_ff + action_fb
        
        if self.task == 'Joint':
            # print('Shape q:{}'.format(np.shape(q)))
            # y = q[:,0]
            # y_dot = q_dot[:,0]
            # y_ddot = self.data.xout[:]
            label_vel_str = r'Vel $\dot{q}_1$ $[rad/s]$'
            label_acc_str = r'Acc $\ddot{q}_1$ $[rad/s^2]$'
            y_label_y_des = r'$q_1$ $[rad]$'
        else:
            # y = pos_x
            # y_dot = vel_x
            # y_ddot = acc_x
            label_vel_str = r'Tip Vel Cartesian $X$ $[m/s]$'
            label_acc_str = r'Tip Acc Cartesian $X$ $[m/s^2]$'
            y_label_y_des = r'Tip Pos Cartesian $X$ $[m]$'
        plt.figure(clear=True)
        plt.ylabel(r'Action $[Nm]$')
        plt.xlabel(r'Time $[s]$')
        plt.plot(self.time, action_ff, color='red', label='ff',linewidth=2)
        plt.plot(self.time, action_fb, color='green', label='fb',linewidth=2)
        plt.plot(self.time, action, color='blue', label='ff + fb',linewidth=2)
        plt.grid()
        plt.legend(loc='best', shadow=True, fontsize='10')
        plt.savefig(dir_plot + 'plot/' + 'action.svg', format='svg')
        plt.close()

        plt.figure(clear=True)
        plt.ylabel(r'Action Per Iter$[Nm]$')
        plt.xlabel(r'Time $[s]$')
        plt.plot(self.time, u_per_iter_ff, color='red', label='ff',linewidth=2)
        plt.grid()
        plt.legend(loc='best', shadow=True, fontsize='10')
        plt.savefig(dir_plot + 'plot/' + 'action_new_per_iter.svg', format='svg')
        plt.close()

        plt.figure(clear=True)
        plt.ylabel(r'Tip Evolution $X-$axis')
        plt.xlabel(r'Time $[s]$')
        plt.plot(self.time, self.y_dot_vett, color='green', label=label_vel_str, linewidth=2)
        plt.plot(self.time, self.y_ddot_vett, color='blue', label=label_acc_str, linewidth=2)
        plt.grid()
        plt.legend(loc='best', shadow=True, fontsize='10')
        plt.savefig(dir_plot + 'plot/' + 'derivative.svg', format='svg')
        plt.close()

        plt.figure(clear=True)
        plt.ylabel(y_label_y_des)
        plt.xlabel(r'Time $[s]$')
        plt.plot(self.time, self.y_des,color='blue',label='Des',linewidth=2)
        plt.plot(self.time, self.y_vett, color='red',label='Real',linewidth=2)
        plt.grid()
        plt.legend(loc='best', shadow=True, fontsize='10')
        plt.savefig(dir_plot + 'plot/' + 'tip.svg', format='svg')
        plt.close()

        plt.figure(clear=True)
        plt.ylabel(r'Error $[m]$')
        plt.xlabel(r'Time $[s]$')
        plt.plot(self.time, err, color='red',linewidth=2)
        plt.grid()
        plt.savefig(dir_plot + 'plot/' + 'error.svg', format='svg')
        plt.close()

    def computeWhatINeed(self, q, dq, action):
        q = np.array(q)
        dq = np.array(dq)
        pinocchio.computeAllTerms(self.robot, self.data, q, dq)
        self.data.Minv = np.linalg.inv(self.data.M)
        # Computing the cost value and residuals
        pinocchio.forwardKinematics(self.robot, self.data, q, dq)
        pinocchio.updateFramePlacements(self.robot, self.data)

        self.data.pos = self.data.oMf[self.robot.getFrameId("Link_EE")].translation.T
        self.data.vel = pinocchio.getFrameVelocity(self.robot, \
            self.data, \
            self.robot.getFrameId("Link_EE"), pinocchio.WORLD).linear
        self.data.acc = pinocchio.getFrameAcceleration(self.robot,\
            self.data,\
            self.robot.getFrameId("Link_EE"),pinocchio.WORLD).linear
        self.data.xout = self.getStateDot(action, q, dq)

    def getStateDot(self, action, q, dq):
        K_q = np.dot(self.K,q)[:] # np.transpose(np.squeeze(np.dot(K,q)))
        D_q_dot = np.dot(self.D,dq)[:]
        D_q_dot = np.squeeze(np.asarray(D_q_dot))

        q_ddot = np.dot(self.data.Minv, (self.underactuation_matrix * action \
                - self.data.nle - K_q - D_q_dot))
        self.data.xout[:] = q_ddot
        return q_ddot

    def getNewControlEachTime(self, y, y_dot, y_ddot, u_old, i):
        '''
        u_old scalr control action for each time
        i time index
        y, y_dot, y_ddot at each time 
        j the iteration index
        '''
        u_new = 0
        err = 0
        u_new_ff = 0
        err = self.gamma0 * (self.y_des[i] - y) + \
            self.gamma1 * (self.y_dot_des[i] - y_dot) + \
            self.gamma2 * (self.y_ddot_des[i] - y_ddot)

        # Jacobian_ee = np.max(pinocchio.getJointJacobian(self.robot, \
        #     self.data,\
        #     self.robot.getFrameId("Link_EE"),\
        #     pinocchio.WORLD))
    
        # max_J_hat = 1
        # self.Gamma_ff = -self.Gamma_ff * max_J_hat
        # self.Gamma_ff = -self.Gamma_ff / (max_J_hat * np.sum(self.data.M[:,1]))
        # print('1/Gamma: {}\nJacobian: {}'.format(np.sum(self.data.M[:,1]), max_J))

        u_new_ff = - np.sign(err) * self.Gamma_ff * err
        # u_new_ff = - self.Gamma_ff * err # classic 
        # u_new_ff = - np.sign(err) * np.sign(u_old) * self.Gamma_ff * err #

        u_new = u_old + u_new_ff # better but diverges after some times

        self.error_in_iter[i] = abs(y - self.y_des[i]) # + \
                # self.y_dot_des[i] - y_dot+ \
                # self.y_ddot_des[i] - y_ddot
        return u_new, err, u_new_ff

    def forwardEulerMethod(self, q, dq):
        q = np.array(q)
        dq = np.array(dq)
        self.data.q = (q + np.multiply(dq, self.dt)).tolist()
        new_q = self.data.q

        self.data.dq = (dq + np.multiply(self.data.xout[:], self.dt)).tolist()
        new_dq = self.data.dq
        return new_q, new_dq

    def main(self):
        viz = crocoddyl.GepettoDisplay(self.robot_display, 4, 4, self.cameraTF)
        action = self.u0_guess
        # print(self.u0_guess, np.shape(self.u0_guess))

        # plt.figure(clear=True)
        # plt.ylabel(r'Control Action $[Nm]$')
        # plt.xlabel(r'Time $[s]$')
        # plt.plot(self.time, action, color='green', linestyle='-',linewidth=2)
        # plt.grid()
        # plt.show()
        print('=================================================================')
        for j in range(0, self.max_iter): # iteration
            action = action
            # print('shape controller {}'.format(np.shape(action)))
            q = list(self.q0)
            q_dot = list(self.dq0)
            self.computeWhatINeed(self.q0, self.dq0, action[0]) 
            
            if self.task == 'Cartesian':
                self.y = self.data.pos[0]
                self.y_dot = self.data.vel[0]
                self.y_ddot = self.data.acc[0]
            else:
                self.y = self.q0[0]
                self.y_dot = self.dq0[0]
                self.y_ddot = self.data.xout[0]
            
            self.q_vett = []
            self.q_dot_vett = []
            self.q_ddot_vett = []
            
            self.pos_vett_x = []
            self.vel_vett_x = []
            self.acc_vett_x = []
            self.pos_vett_z = []
            self.vel_vett_z = []
            self.acc_vett_z = []
            
            self.u_fb_vett = []
            self.y_vett = []
            self.y_dot_vett = []
            self.y_ddot_vett = []
            self.control_action_vett = []
            self.q_1_vett = []
            try: 
                for i in range(0, self.steps): # time 
                    action_plug = action[i]
                    u_fb = self.K_P * (self.y_des[i] - self.y) + self.K_V * (self.y_dot_des[i] - self.y_dot) + self.K_A * (self.y_ddot_des[i] - self.y_ddot)
                    u_fb = - self.Gamma_fb * u_fb
                    # u_fb = self.Gamma_fb * u_fb
                    control_action = action_plug + u_fb                
                    # if abs(control_action) > self.max_actuator:
                    #     control_action = np.sign(control_action) * self.max_actuator

                    self.computeWhatINeed(q, q_dot, control_action)
                    q, q_dot = self.forwardEulerMethod(q, q_dot)
                    self.q_vett.append(q) # \in\mathbbf{R}^{n_jointsxself.steps} vector
                    self.q_dot_vett.append(q_dot)
                    self.q_ddot_vett.append(self.data.xout)
                    self.pos_vett_x.append(self.data.pos[0])
                    self.vel_vett_x.append(self.data.vel[0])
                    self.acc_vett_x.append(self.data.acc[0])
                    self.pos_vett_z.append(self.data.pos[2])
                    self.vel_vett_z.append(self.data.vel[2])
                    self.acc_vett_z.append(self.data.acc[2])
                    
                    self.u_fb_vett.append(u_fb)
                    self.control_action_vett.append(control_action)
                    viz.robot.display(np.array(q))
                    viz.robot.viewer.gui.addSphere('world/point', 0.05, [1.0, 0.0, 0.0, 1.0])  # radius = .1, RGBA=1001
                    viz.robot.viewer.gui.applyConfiguration('world/point',
                                    self.target_pos.tolist() + [0.0, 0.0, 0.0, 1.0])  # xyz+quaternion
                    # traj
                    viz.robot.viewer.gui.addSphere('world/point', 0.01, [13.0, 180.0, 185.0, 1.0])  # radius = .1, RGBA=1001
                    viz.robot.viewer.gui.applyConfiguration('world/point',
                                    np.array([self.pos_x[i], self.pos_y[i], self.pos_z[i]]).tolist() + [0.0, 0.0, 0.0, 1.0])  # xyz+quaternion
                                    # np.array([self.data.pos[0], self.data.pos[1], self.data.pos[2]]).tolist() + [0.0, 0.0, 0.0, 1.0])  # xyz+quaternion
                                    # np.array([self.y_des[i], self.pos_y[i], self.pos_z[i]]).tolist() + [0.0, 0.0, 0.0, 1.0])  # xyz+quaternion
                    viz.robot.viewer.gui.refresh()
                    
                    if self.task == 'Cartesian':
                        ###### Controlling the X-coordinate of the robot tip 
                        self.y = self.data.pos[0]
                        self.y_dot = self.data.vel[0]
                        self.y_ddot = self.data.acc[0]
                    else:
                        ###### Controlling position of the first joint
                        self.y = q[0]
                        self.y_dot = q_dot[0]
                        self.y_ddot = self.data.xout[0]
                    self.q_1_vett.append(q[0])
                    
                    ###### Storing for plots later on 
                    self.y_vett.append(self.y)
                    self.y_dot_vett.append(self.y_dot)
                    self.y_ddot_vett.append(self.y_ddot)
                    # time.sleep(self.dt) 
                    self.u_new[i], self.err[i], self.u_per_iter_ff[i] = self.getNewControlEachTime(self.y, self.y_dot, self.y_ddot, action_plug, i)
                    self.u_new[i] = self.u_new[i] + u_fb # iterative ff + fb
                    if (i == self.steps - 1):
                        self.err_t_f[j] = self.error_in_iter[-1]
                if j%5 == 0:            
                    self.saveData(j, 
                                self.error_in_iter, 
                                self.pos_vett_x, 
                                self.vel_vett_x,
                                self.acc_vett_x, 
                                self.pos_vett_z, 
                                self.vel_vett_z,
                                self.acc_vett_z,
                                self.control_action_vett, 
                                self.u_fb_vett, 
                                self.u_per_iter_ff, 
                                self.q_vett, 
                                self.q_dot_vett)

                    self.savePlot(j)
                self.error2print[j] = np.linalg.norm(self.error_in_iter,2)
                MSE = np.square(self.error_in_iter).mean()   
                self.err_rms[j] = math.sqrt(MSE) 
                print('Iteration number:', str(j))
                print('Final Err: {}'.format(np.round(self.err_t_f[j],4)))
                print('Error RMS: {}'.format(np.round(self.err_rms[j],4))) 
                if j >= 1:
                    if self.err_rms[j] > self.err_rms[j-1]:
                        print(colored('================= Error is increasing ===========================','red'))
                        self.conta_incr += 1
                        if self.err_t_f[j] > self.err_t_f[j-1]:
                            print('=============== Tip error is increasing =========================')
                    else:
                        print(colored('================= Error is decreasing ===========================','green'))
                        if self.err_t_f[j] > self.err_t_f[j-1]:
                            print('=============== Tip error is increasing =========================')
                else:
                    print('================================================================')
                    
                action = self.u_new
                    
                # if self.err_rms[j] < self.toll or self.err_t_f[j] < self.toll:
                if self.err_rms[j] < self.toll or j == self.max_iter - 1: # and self.err_t_f[j] < self.toll:
                    print(colored('============================ FINISH =============================', 'green'))
                    self.FINISH_LEARNING = True
                    dir_plot = './' + self.directory
                    
                    ERR_RMS = dir_plot + '/err_rms.csv'
                    ERR_T_F = dir_plot + '/err_t_f.csv'
                    
                    self.err_rms.tofile(ERR_RMS, sep=',')
                    self.err_t_f.tofile(ERR_T_F, sep=',')

                    plt.figure(clear=True)
                    plt.ylabel(r'RMS $[m]$')
                    plt.xlabel(r'Iterations')
                    plt.plot(range(0,j + 1), self.err_rms[0:j+1], color='red', marker='o', linestyle='-.', linewidth=2, markersize=12)
                    plt.grid()
                    plt.savefig(dir_plot + '/' +'rms.svg', format='svg')
                    plt.show()

                    plt.figure(clear=True)
                    plt.ylabel(r'Final Error $[m]$')
                    plt.xlabel(r'Iterations')
                    plt.plot(range(0,j + 1), self.err_t_f[0:j+1], color='red', marker='*',linestyle='-',linewidth=2, markersize=12)
                    plt.grid()
                    plt.savefig(dir_plot + '/' +'err_tf_final.svg', format='svg')
                    plt.show()
            
                if self.FINISH_LEARNING:
                    print(colored('         LEARNING HAS BEEN SUCCESSUFULLY ACHIVED!        ', 'green'))

                    dir_plot = './' + self.directory
                    # plt.figure(clear=True)
                    # plt.ylabel(r'RMS $[m]$')
                    # plt.xlabel(r'Iterations')
                    # plt.plot(range(0,j), self.err_rms[0:j], color='red', marker='o', linestyle='-.', linewidth=2, markersize=12)
                    # plt.grid()
                    # plt.savefig(dir_plot + '/' +'rms.svg', format='svg')
                    # plt.show()

                    # plt.figure(clear=True)
                    # plt.ylabel(r'Final Error $[m]$')
                    # plt.xlabel(r'Iterations')
                    # plt.plot(range(0,j), self.err_t_f[0:j], color='red', marker='*',linestyle='-',linewidth=2, markersize=12)
                    # plt.grid()
                    # plt.savefig(dir_plot + '/' + 'err_tf_final.svg', format='svg')
                    # plt.show()
                    
                    plt.figure(clear=True)
                    plt.ylabel(r'$q_1$ $[rad]$')
                    plt.xlabel(r'Time $[s]$')
                    plt.plot(self.time, self.q_1_vett, color='blue', linestyle='-.',linewidth=2)
                    plt.grid()
                    plt.savefig(dir_plot + '/' + 'q_1.svg', format='svg')
                    plt.show()

                    plt.figure(clear=True)
                    plt.ylabel(r'Final Ouput $[m]$')
                    plt.xlabel(r'Time $[s]$')
                    plt.plot(self.time, self.y_des, color='red', linestyle='-',linewidth=2, label= 'Des')
                    plt.plot(self.time, self.pos_vett_x, color='blue', linestyle='-.',linewidth=2, label= 'Real')
                    plt.grid()
                    plt.legend(loc='best', shadow=True, fontsize='10')
                    plt.savefig(dir_plot + '/' + 'final_track.svg', format='svg')
                    plt.show()
                    sys.exit()
            except KeyboardInterrupt:
                self.FINISH_LEARNING = True
                dir_plot = './' + self.directory
                    
                ERR_RMS = dir_plot + '/err_rms.csv'
                ERR_T_F = dir_plot + '/err_t_f.csv'
                    
                self.err_rms.tofile(ERR_RMS, sep=',')
                self.err_t_f.tofile(ERR_T_F, sep=',')

                plt.figure(clear=True)
                plt.ylabel(r'RMS $[m]$')
                plt.xlabel(r'Iterations')
                plt.plot(range(0,j + 1), self.err_rms[0:j+1], color='red', marker='o', linestyle='-.', linewidth=2, markersize=12)
                plt.grid()
                plt.savefig(dir_plot + '/' +'rms.svg', format='svg')
                plt.show()

                plt.figure(clear=True)
                plt.ylabel(r'Final Error $[m]$')
                plt.xlabel(r'Iterations')
                plt.plot(range(0,j + 1), self.err_t_f[0:j+1], color='red', marker='*',linestyle='-',linewidth=2, markersize=12)
                plt.grid()
                plt.savefig(dir_plot + '/' +'err_tf_final.svg', format='svg')
                plt.show()
            
                if self.FINISH_LEARNING:
                    print(colored('         LEARNING HAS BEEN SUCCESSUFULLY ACHIVED!        ', 'green'))

                    dir_plot = './' + self.directory
                    # plt.figure(clear=True)
                    # plt.ylabel(r'RMS $[m]$')
                    # plt.xlabel(r'Iterations')
                    # plt.plot(range(0,j), self.err_rms[0:j], color='red', marker='o', linestyle='-.', linewidth=2, markersize=12)
                    # plt.grid()
                    # plt.savefig(dir_plot + '/' +'rms.svg', format='svg')
                    # plt.show()

                    # plt.figure(clear=True)
                    # plt.ylabel(r'Final Error $[m]$')
                    # plt.xlabel(r'Iterations')
                    # plt.plot(range(0,j), self.err_t_f[0:j], color='red', marker='*',linestyle='-',linewidth=2, markersize=12)
                    # plt.grid()
                    # plt.savefig(dir_plot + '/' + 'err_tf_final.svg', format='svg')
                    # plt.show()
                    
                    plt.figure(clear=True)
                    plt.ylabel(r'$q_1$ $[rad]$')
                    plt.xlabel(r'Time $[s]$')
                    plt.plot(self.time, self.q_1_vett, color='blue', linestyle='-.',linewidth=2)
                    plt.grid()
                    plt.savefig(dir_plot + '/' + 'q_1.svg', format='svg')
                    plt.show()

                    plt.figure(clear=True)
                    plt.ylabel(r'Final Ouput $[m]$')
                    plt.xlabel(r'Time $[s]$')
                    plt.plot(self.time, self.y_des, color='red', linestyle='-',linewidth=2, label= 'Des')
                    plt.plot(self.time, self.pos_vett_x, color='blue', linestyle='-.',linewidth=2, label= 'Real')
                    plt.grid()
                    plt.legend(loc='best', shadow=True, fontsize='10')
                    plt.savefig(dir_plot + '/' + 'final_track.svg', format='svg')
                    plt.show()
                    sys.exit()
                

        


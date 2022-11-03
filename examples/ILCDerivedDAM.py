from inspect import getframeinfo
from math import gamma

from torch import _linalg_inv_out_helper_
import crocoddyl
import pinocchio
import numpy as np
import aslr_to
from free_ffdyn_ilc import DAMILC
import sys
import matplotlib.pyplot as plt
import time


class ILCDerivedDAM(crocoddyl.SolverAbstract):

    def __init__(self, 
                shootingProblem, 
                u_0, 
                traj_des, 
                traj_des_dot, 
                traj_des_ddot, 
                T, 
                robot_model, 
                dt, 
                xs, 
                fishing_rod, 
                x_des):
        crocoddyl.SolverAbstract.__init__(self, shootingProblem)
        self.T = T
        self.dt = dt
        self.maxiter = 10
        self.initial_guess = u_0

        # plt.figure(clear=True)
        # plt.ylabel(r'Action $[Nm]$')
        # plt.xlabel(r'Time $[s]$')
        # plt.plot(u_0, color='green')
        # plt.grid()
        # plt.show()
        
        self.toll = 1e-2
        self.des_traj = traj_des
        self.des_traj_dot = traj_des_dot
        self.des_traj_ddot = traj_des_ddot
        self.gamma0 = 1
        self.gamma1 = 0.00001
        self.gamma2 = 0.000001
        self.rel_deg = 2
        self.x0 = self.problem.x0
        self.x_des = x_des
        self.robot_model = robot_model
        self.temp_err = [np.zeros(1)]*self.T
        self.state = crocoddyl.StateMultibody(robot_model)
        self.data = robot_model.createData()
        self.actuation = aslr_to.ASRFishing(self.state)
        self.nu = self.actuation.nu
        self.Gamma = np.eye(self.nu)
        self.xs = xs
        self.err_naked = [np.zeros(1)]*self.T
        pinocchio.forwardKinematics(self.robot_model, self.data, self.x0[:self.state.nq])
        pinocchio.updateFramePlacements(self.robot_model, self.data)
        self.current_pos = self.data.oMf[self.robot_model.getFrameId("Link_EE")].translation.T
        cameraTF = [2., 2.68, 0.54, 0.2, 0.62, 0.72, 0.22]
        self.display = crocoddyl.GepettoDisplay(fishing_rod, 4, 4, cameraTF)
        self.current_vel = np.zeros(3)
        self.current_acc = np.zeros(3)
        self.under_matrix = np.append(1, np.zeros(self.state.nv-1))
        self.allocateData() 

        # print('-------------------') 
        # 'Initialization Completed',
        # print('-------------------')

    def solve(self):

        u_old = self.initial_guess
        # print(np.shape(u_old))
        Gamma_j = self.Gamma 

        for i in range(self.maxiter):

            # self.problem.calc(self.xs,u_old)
            u_old, Gamma_j = self.computeDirection(u_old, Gamma_j)

            # plt.figure(clear=True)
            # plt.ylabel(r'Action $[Nm]$')
            # plt.xlabel(r'Time $[s]$')
            # plt.plot(u_old, color='green')
            # plt.grid()
            # plt.show()
            
            # print('------------------------------------------------------------------------')
            # print(Gamma_j)

            self.iter = i
            self.err[i] = self.err_naked
            print('------------------------------------------------------------------------')
            print('Iteration number: ', str(i))
            print('Error is:', np.linalg.norm(self.err[i],2))
            # for k in range(0,self.T):
            #     print(np.linalg.norm(self.xs[k] - self.x_des[k],2))  
            print('------------------------------------------------------------------------')
            if np.linalg.norm(self.err[i],2) < self.toll:
                print('Finish\n')
                sys.exit()
                return self.xs, self.us
        return self.xs, self.us

    def computeDirection(self, u_old, Gamma_j):

        for t, (m, d) in enumerate(zip(self.problem.runningModels, self.problem.runningDatas)):

            self.temp_err[t] = self.gamma0 * (self.current_pos[0] - self.des_traj[t]) \
                    + self.gamma1 * (self.current_vel[0] - self.des_traj_dot[t]) \
                    + self.gamma2 * (self.current_acc[0] - self.des_traj_ddot[t])

            self.err_naked[t] = (self.current_pos[0] - self.des_traj[t]) # \
                            # + (self.current_vel[0] - self.des_traj_dot[t]) \
                            # + (self.current_acc[0] - self.des_traj_ddot[t])

            m.calc(d, self.xs[t], u_old[t])
            self.xs[t+1] = d.xnext  

            u_old[t] = u_old[t] + np.dot(Gamma_j, self.temp_err[t])

            q = d.xnext[:self.state.nv]
            q_dot = d.xnext[-self.state.nv:]

            self.display.robot.display(q)


            pinocchio.computeAllTerms(m.differential.state.pinocchio,d.differential.multibody.pinocchio,q,q_dot)
            pinocchio.forwardKinematics(m.differential.state.pinocchio, d.differential.multibody.pinocchio, q, q_dot)
            pinocchio.updateFramePlacements(m.differential.state.pinocchio, d.differential.multibody.pinocchio)
            # y_d_dq = pinocchio.getJointJacobian(m.differential.state.pinocchio, d.differential.multibody.pinocchio,  self.robot_model.getJointId("Joint_21"), pinocchio.WORLD)
            # M_iniertia = d.differential.multibody.pinocchio.M
            # # Gamma_j = 1/(np.dot(y_d_dq[1,:], np.dot(M_iniertia,self.under_matrix))) # controllling x_hat movement, singularity
            # # Gamma_j = self.Gamma
            # # if t == 10:
            # #     print('------ = \n', np.linalg.norm(y_d_dq,2))
            # #     print('M_inertia = \n', np.linalg.norm(M_iniertia,2))
            Gamma_j = 1 # 1/(np.linalg.norm(y_d_dq,2)*np.linalg.norm(M_iniertia,2))

            self.current_pos = d.differential.multibody.pinocchio.oMf[self.robot_model.getFrameId("Link_EE")].translation.T
            self.current_vel = pinocchio.getFrameVelocity(m.differential.state.pinocchio,d.differential.multibody.pinocchio, self.robot_model.getFrameId("Link_EE")).linear
            self.current_acc = pinocchio.getFrameAcceleration(m.differential.state.pinocchio,d.differential.multibody.pinocchio,self.robot_model.getFrameId("Link_EE"),pinocchio.WORLD).linear
            # # if t == 10:
            # #     print('Cartesian Pos: ', self.current_pos)
            # #     print('Cartesian Vel: ', self.current_vel)
            # #     print('Cartesian Acc: ', self.current_acc)
        return u_old, Gamma_j

    def computeGains(self):
        pass
        
    def allocateData(self):
        self.q = [np.zeros(self.state.nq) for i in range(self.T)]
        self.v = [np.zeros(self.state.nq) for i in range(self.T)]
        self.a = [np.zeros(self.state.nq) for i in range(self.T)]
        self.err = [np.zeros(self.maxiter) for i in range(self.T)]
        self.us = [np.zeros(self.actuation.nu) for i in range(self.T)]
        # self.v = [np.zeros([m.state.nv]) for m in zip(self.problem.runningModels)]


from __future__ import print_function
from cmath import inf
import os
import sys
from aslr_to import actuation_fishing
import crocoddyl
import pinocchio
import numpy as np
import example_robot_data
import aslr_to
import time
from scipy.io import savemat
from numpy import linalg as LA
import pathlib
from pinocchio.visualize import GepettoVisualizer
# seed the pseudorandom number generator
from numpy.random import seed
from numpy.random import rand
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import csv
from numdifftools import Jacobian, Hessian
import shutil


# class GetProjectileMotion():
#     # https://apmonitor.com/pdc/index.php/Main/NonlinearProgramming
#     def __init__(self, X_des, Z_des):
#         self.GRAVITY = 9.81
#         self.gravity = [0, 0, -self.GRAVITY]
#         self.X_des = X_des
#         self.Z_des = Z_des
#         self.theta_des = np.pi/4
#         self.inequalities = False
#         self.z0 = [0.01, 0, 0] # theta is NOT a state variables
#         self.bounds = ((0.2,inf),(1,2.5),(0.4,2.1)) # theta is NOT a state variables
#         # self.v_0 = 0       # z_0
#         # self.X_0 = 0       # z_1
#         # self.Z_0 = 0       # z_2
       
#     def getFun(self, z):
#         # v_0 = z[0]
#         # X_0 = z[1]
#         # Z_0 = z[2]
#         # T = z[3]
#         v_0 = z[0]
#         X_0 = z[1]
#         Z_0 = z[2]
#         # return 1e5 * (self.X_des - X_0)**2 + 1e5 * (self.Z_des - Z_0)**2 + 1e1 * v_0**2
#         J = (self.Z_des - Z_0) -  (self.X_des - X_0) * np.tan(self.theta_des) \
#             + 0.5 * self.GRAVITY * ((self.X_des - X_0)/(v_0 * np.cos(self.theta_des)))**2
#         return J

#     def getJacobianOfFun(self,z):
#         return Jacobian(lambda z: self.getFun(z))(z).ravel()
#     def getHessOfFun(self,z):
#         return Hessian(lambda z: self.getFun(z))(z)

#     def getConstraint(self,z):
#         # v_0 = z[0]
#         # X_0 = z[1]
#         # T = z[2]
#         v_0 = z[0]
#         X_0 = z[1]
#         Z_0 = z[2]
#         theta_0 = self.theta_des

#         b = 2 * v_0 * np.sin(theta_0)/(self.GRAVITY)
#         delta = np.sqrt(b**2 - 8 * (self.Z_des - Z_0)/self.GRAVITY)
#         T = (b + delta)/2
#         # in inequality changes - into c
#         self.toll = 0 # regularizzation

#         c_1 = self.X_des - X_0 - v_0 * np.cos(theta_0) * T - self.toll
#         c_2 = self.Z_des - Z_0 - v_0 * np.sin(theta_0) * T + 0.5 * self.GRAVITY * (T) ** 2 - self.toll
#         # c_3 = (self.Z_des - Z_0) - (self.X_des - X_0) * np.tan(theta_0) +\
#         #      0.5 * self.GRAVITY * ((self.X_des - X_0)/(v_0*np.cos(theta_0)))**2 - self.toll
#         if self.toll > 0:
#             self.inequalities = True
#             # print('Inequality constraints')
#             # c = ([-c_1, -c_2, -c_3])
#             c = ([-c_1, -c_2])
#         else:
#             self.inequalities = False
#             # c = ([c_1, c_2, c_3])
#             c = ([c_1, c_2])
#         return c
#         # return c_1

#     def getXZvt(self):
#         if self.inequalities:
#             cons_tot = {'type': 'ineq', 'fun': self.getConstraint}
#         else:
#             cons_tot = {'type': 'eq', 'fun': self.getConstraint}

#         solution = minimize(self.getFun,
#                     self.z0,
#                     method='SLSQP',
#                     bounds=self.bounds,
#                     # jac=self.getJacobianOfFun,
#                     # hess=self.getHessOfFun,
#                     constraints=cons_tot,
#                     tol= 1e0,
#                     options={'disp': True, 'maxiter': 1e5})
#         print(solution)
#         return solution

# # X_des = 10
# # Z_des = 0
# # test_motion = GetProjectileMotion(X_des, Z_des)
# # try_res = test_motion.getXZvt()
# # state_des = try_res.x
# # print('===================================')
# # b = 2 * state_des[0] * np.sin(test_motion.theta_des)/(test_motion.GRAVITY)
# # delta = np.sqrt(b**2 - 8 * (test_motion.Z_des - state_des[2])/test_motion.GRAVITY)
# # time_f = (b + delta)/2
# # print('v_0: {}\ntheta_0: {}\nX_0: {}\nZ_0: {}\nT:{}'\
# #     .format(state_des[0],test_motion.theta_des,state_des[1], state_des[2], time_f))
# # print('===================================')
# # if try_res.success:
# #     print('Optimization ... SUCCESS')
# # else:
# #     print('Optimization ... FAILED')

# # suppose_trow_X = state_des[1] + state_des[0] * np.cos(test_motion.theta_des) * time_f
# # suppose_trow_Z = state_des[2] + state_des[0] * np.sin(test_motion.theta_des) * time_f \
# #     - 0.5 * test_motion.GRAVITY * time_f**2
# # print('Final Pos Motion\nX:{}\nZ:{}\nerr X:{}\nerr Z:{}'.format(suppose_trow_X,suppose_trow_Z,\
# #     abs(X_des - suppose_trow_X),abs(Z_des - suppose_trow_Z)))


# directory = 'data_opt' 
# if os.path.exists(directory):
#     shutil.rmtree(directory)
# os.makedirs(directory)


# Z_des = 0
# conto = 0
# max_trial = 250
# for X_des in range(1,max_trial):
#     print(X_des)
#     test_motion = GetProjectileMotion(X_des, Z_des)
#     try_res = test_motion.getXZvt()
#     if try_res.success :
#         conto = conto + 1
#         b = 2 * try_res.x[0] * np.sin(test_motion.theta_des)/(test_motion.GRAVITY)
#         delta = np.sqrt(b**2 - 8 * (test_motion.Z_des - try_res.x[2])/test_motion.GRAVITY)
#         time_f = (b + delta)/2
#         suppose_trow_X = try_res.x[1] + try_res.x[0] * np.cos(test_motion.theta_des) * time_f
#         suppose_trow_Z = try_res.x[2] + try_res.x[0] * np.sin(test_motion.theta_des) * time_f \
#             - 0.5 * test_motion.GRAVITY * time_f**2
#         print('Final Pos Motion\nX:{}\nZ:{}\nerr X:{}\nerr Z:{}'.format(suppose_trow_X,suppose_trow_Z,\
#             abs(test_motion.X_des - suppose_trow_X),abs(Z_des - suppose_trow_Z)))
#         print('======================================================================')
#         print('SUCCESS')
#         print('state: ', try_res.x)
#         print('X_des:', X_des)
#         time.sleep(1)
#         with open(str(pathlib.Path().resolve()) +'/'+ directory +\
#             '/test_v0XZ_reg_' + str(conto) +'.csv', 'w') as f:
#             writer = csv.writer(f)
#             writer.writerow(['X_des', 'try_res'])
#             writer.writerow([X_des, try_res.x])
#             # writer.writerow(try_res.x)
#             f.close()
# print('======================================================================')
# print('SUCCESS tests are {}/{}'.format(conto,max_trial))



class GetProjectileMotion():
    # https://apmonitor.com/pdc/index.php/Main/NonlinearProgramming
    # https://het.as.utexas.edu/HET/Software/Scipy/generated/scipy.optimize.minimize.html
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    def __init__(self, X_des, Z_des):
        self.GRAVITY = 9.81
        self.gravity = [0, 0, -self.GRAVITY]
        self.X_des = X_des
        self.Z_des = Z_des
        self.inequalities = False
        self.z0 = [0.01, 2, 0, 0.2] # theta is NOT a state variables
        self.bounds = ((0.2,inf),(1,2.5),(0.4,2.1),(0,np.pi)) # theta is NOT a state variables
        # self.v_0 = 0       # z_0
        # self.X_0 = 0       # z_1
        # self.Z_0 = 0       # z_2
        # self.theta_0 = 0   # z_3
       
    def getFun(self, z):
        # v_0 = z[0]
        # X_0 = z[1]
        # Z_0 = z[2]
        # T = z[3]
        v_0 = z[0]
        X_0 = z[1]
        Z_0 = z[2]
        theta_0 = z[3]
        # return 1e5 * (self.X_des - X_0)**2 + 1e5 * (self.Z_des - Z_0)**2 + 1e1 * v_0**2
        J = (self.Z_des - Z_0) -  (self.X_des - X_0) * np.tan(theta_0) \
            + 0.5 * self.GRAVITY * ((self.X_des - X_0)/(v_0 * np.cos(theta_0)))**2
        return J

    def getJacobianOfFun(self,z):
        return Jacobian(lambda z: self.getFun(z))(z).ravel()
    def getHessOfFun(self,z):
        return Hessian(lambda z: self.getFun(z))(z)

    def getConstraint(self,z):
        v_0 = z[0]
        X_0 = z[1]
        Z_0 = z[2]
        theta_0 = z[3]

        b = 2 * v_0 * np.sin(theta_0)/(self.GRAVITY)
        delta = np.sqrt(b**2 - 8 * (self.Z_des - Z_0)/self.GRAVITY)
        T = (b + delta)/2
        # in inequality changes - into c
        self.toll = 0 # regularizzation

        c_1 = self.X_des - X_0 - v_0 * np.cos(theta_0) * T 
        c_2 = self.Z_des - Z_0 - v_0 * np.sin(theta_0) * T + 0.5 * self.GRAVITY * (T) ** 2
        # c_3 = (self.Z_des - Z_0) - (self.X_des - X_0) * np.tan(theta_0) +\
        #      0.5 * self.GRAVITY * ((self.X_des - X_0)/(v_0*np.cos(theta_0)))**2 
        if self.toll > 0:
            self.inequalities = True
            # print('Inequality constraints')
            # c = ([-c_1, -c_2, -c_3])
            c = ([-c_1 + self.toll, -c_2 + self.toll])
        else:
            self.inequalities = False
            # c = ([c_1, c_2, c_3])
            c = ([c_1, c_2])
        return c
        # return c_1

    def getXZvt(self):
        if self.inequalities:
            cons_tot = {'type': 'ineq', 'fun': self.getConstraint}
        else:
            cons_tot = {'type': 'eq', 'fun': self.getConstraint}

        solution = minimize(self.getFun,
                    self.z0,
                    # method= 'trust-constr',
                    # method='TNC',
                    method='SLSQP',
                    # method='cobyla',
                    bounds=self.bounds,
                    jac=self.getJacobianOfFun,
                    hess=self.getHessOfFun,
                    constraints=cons_tot,
                    tol= 1e0,
                    options={'disp': True, 'maxiter': 1e5})
        print(solution)
        return solution


directory = 'data_opt_with_theta_reg' 
if os.path.exists(directory):
    shutil.rmtree(directory)
os.makedirs(directory)


Z_des = 0
conto = 0
max_trial = 250
for X_des in range(1,max_trial):
    print(X_des)
    test_motion = GetProjectileMotion(X_des, Z_des)
    try_res = test_motion.getXZvt()
    if try_res.success :
        conto = conto + 1
        b = 2 * try_res.x[0] * np.sin(try_res.x[3])/(test_motion.GRAVITY)
        delta = np.sqrt(b**2 - 8 * (test_motion.Z_des - try_res.x[2])/test_motion.GRAVITY)
        time_f = (b + delta)/2
        suppose_trow_X = try_res.x[1] + try_res.x[0] * np.cos(try_res.x[3]) * time_f
        suppose_trow_Z = try_res.x[2] + try_res.x[0] * np.sin(try_res.x[3]) * time_f \
            - 0.5 * test_motion.GRAVITY * time_f**2
        print('Final Pos Motion\nX:{}\nZ:{}\nerr X:{}\nerr Z:{}'.format(suppose_trow_X,suppose_trow_Z,\
            abs(test_motion.X_des - suppose_trow_X),abs(Z_des - suppose_trow_Z)))
        print('======================================================================')
        print('SUCCESS')
        print('state: ', try_res.x)
        print('X_des:', X_des)
        time.sleep(1)
        with open(str(pathlib.Path().resolve()) +'/'+ directory +\
            '/test_v0XZ_reg_' + str(conto) +'.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['X_des', 'try_res: [v0,X0,Z0,theta0]'])
            writer.writerow([X_des, try_res.x])
            # writer.writerow(try_res.x)
            f.close()
print('======================================================================')
print('SUCCESS tests are {}/{}'.format(conto,max_trial))


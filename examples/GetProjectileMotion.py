from cgi import test
import numpy as np
from scipy.optimize import minimize
from cmath import inf
from numdifftools import Jacobian, Hessian
from termcolor import colored


WANT_DISP = False
class GetProjectileMotion():
    def __init__(self, X_des, Z_des, v_des, T):
        self.GRAVITY = 9.81
        self.gravity = [0, 0, -self.GRAVITY]
        self.X_des = X_des
        self.Z_des = Z_des
        self.v_des = v_des
        self.T = T  # [sec] time of flight 
        self.L_bar = 3.0 # [m] length of the fishing rod
        self.inequalities = True #False
        self.z0 = [0.02, 0.1, 3, 0.02] # theta is NOT a state variables
        self.bounds = ((0.1, inf),(0.1, 3.0),(0.1, 3.0),(0.1, inf)) # theta is NOT a state variables
        # self.v_0x = 0         # z_0
        # self.X_0 = 0          # z_1
        # self.Z_0 = 0          # z_2
        # self.v_0z = 0         # z_3
        # Note that # self.v_0x**2 + # self.v_0z**2 = self.v_des        

    def getFun(self, z):
        
        v_0x = z[0]
        X_0 = z[1]
        Z_0 = z[2]
        v_0z = z[3]
        
        J = 1e2 * (self.X_des - X_0)**2 \
            + 1e2 * (self.Z_des - Z_0)**2 \
            + 1e1 * (-np.sqrt(v_0x**2 + v_0z**2) + self.v_des)**2
        return J

    def getJacobianOfFun(self,z):
        return Jacobian(lambda z: self.getFun(z))(z).ravel()
    def getHessOfFun(self,z):
        return Hessian(lambda z: self.getFun(z))(z)

    def getConstraint(self,z):
        v_0x = z[0]
        X_0 = z[1]
        Z_0 = z[2]
        v_0z = z[3]
        
        # try if this works even with this computation
        # b = 2 * v_0x/(self.GRAVITY)
        # delta = np.sqrt(b**2 - 8 * (self.Z_des - Z_0)/self.GRAVITY)
        # T = (b + delta)/2
        self.toll = 0.1 # regularizzation

        c_1 = self.X_des - X_0 - v_0x * self.T 
        c_2 = self.Z_des - Z_0 - v_0z * self.T + 0.5 * self.GRAVITY * (self.T) ** 2
        c_3 = Z_0**2 + X_0**2 - self.L_bar**2
        if self.toll > 0:
            self.inequalities = True
            # print('Inequality constraints')
            c = ([-c_1 + self.toll, -c_2 + self.toll, -c_3 + self.toll])
        else:
            self.inequalities = False
            c = ([c_1, c_2, c_3])
        return c

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
                    tol= 1e-2,
                    options={'disp': WANT_DISP, 'maxiter': 1e4})
        print(solution)
        return solution

# conta_succ = 0
# n_X_des = 20
# n_v_des = 10
# n_T = 10
# for i in range(1, n_X_des):
#     for j in range(1, n_v_des):
#         for k in range(1, n_T):
#             X_des = i
#             Z_des = 0
#             v_des = j
#             T = k
#             test_motion = GetProjectileMotion(X_des, Z_des, v_des, T)
#             try_res = test_motion.getXZvt()
#             state_des = try_res.x
#             v_0x, X_0, Z_0, v_0z = try_res.x
#             # b = 2 * test_motion.v_0 * np.sin(test_motion.theta_0)/(test_motion.GRAVITY)
#             # delta = np.sqrt(b**2 - 8 * (Z_des - Z_0)/test_motion.GRAVITY)
#             # T = (b + delta)/2
            
#             print('=============================================================================================')
#             print('v_0x: {}\nX_0: {}\t\t X_des: {}\nZ_0: {}\t\t Z_des: {}\nv_0z: {}\ntheta_0: {}\nT: {}'\
#                 .format(v_0x, state_des[0], X_des, state_des[1], Z_des, v_0z, np.arctan(v_0x/v_0z), test_motion.T))
#             err_X = abs(X_des - X_0 - v_0x * test_motion.T)
#             err_Z = abs(Z_des - Z_0 - v_0z * test_motion.T + 0.5 * test_motion.GRAVITY * (test_motion.T) ** 2)
#             print('=============================================================================================')
#             print('err_X: {}\nerr_Z: {}'.format(err_X, err_Z))
            
#             if try_res.success:
#                 conta_succ = conta_succ + 1
#                 print(colored('Success','green'))
#                 print('X_des: {}\nv_des: {}'.format(X_des, v_des))
            
# print('Number of Success: {} over {} tests'.format(conta_succ, n_v_des * n_X_des * n_T))
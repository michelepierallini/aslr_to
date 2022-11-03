from __future__ import print_function
from matplotlib.ft2font import LOAD_IGNORE_GLOBAL_ADVANCE_WIDTH
import crocoddyl
import pinocchio
import numpy as np
import example_robot_data
import aslr_to
import time
from numpy import linalg as LA
import matplotlib.pyplot as plt
from cmath import inf
from GetProjectileMotion import GetProjectileMotion
from ILC_OnlyPinocchio import ILC_OnlyPinocchioFishing
import math
import shutil
import os
from termcolor import colored


ILC_TRUE = True
# WITHDISPLAY = 'display' in sys.argv or 'CROCODDYL_DISPLAY' in os.environ
WITHDISPLAY = True
FISHING_STRING = 'fishing_rod_two'

X_des = 20
Z_des = 0
v_des = 2
T_f_des = 10
test_motion = GetProjectileMotion(X_des, Z_des, v_des, T_f_des)
try_res = test_motion.getXZvt()
state_des = try_res.x
v_0x, X_0, Z_0, v_0z = try_res.x
# b = 2 * test_motion.v_0 * np.sin(test_motion.theta_0)/(test_motion.GRAVITY)
# delta = np.sqrt(b**2 - 8 * (Z_des - Z_0)/test_motion.GRAVITY)
 # T = (b + delta)/2

print('=============================================================================================')
print('v_0x: {}\nX_0: {}\t\t X_des: {}\nZ_0: {}\t\t Z_des: {}\nv_0z: {}\ntheta_0: {}\nT: {}'\
    .format(v_0x, state_des[0], X_des, state_des[1], Z_des, v_0z, np.arctan(v_0x/v_0z), test_motion.T))
err_X = abs(X_des - X_0 - v_0x * test_motion.T)
err_Z = abs(Z_des - Z_0 - v_0z * test_motion.T + 0.5 * test_motion.GRAVITY * (test_motion.T) ** 2)
print('=============================================================================================')
print('err_X: {}\nerr_Z: {}'.format(err_X, err_Z))
if try_res.success:
    print(colored('Success','green'))
    print('X_des: {}\nv_des: {}'.format(X_des, v_des))

print('=============================================================================================')
print('==================== WE START WITH DDP TO CREATE THE REFERENCE ==============================')
print('=============================================================================================')

# fishing_rod = example_robot_data.load('fishing_rod')
fishing_rod = example_robot_data.load(FISHING_STRING)
robot_model = fishing_rod.model
robot_model.gravity.linear = np.array([0, 0, -9.81]) # w.r.t. global frame
state = crocoddyl.StateMultibody(robot_model)
actuation = aslr_to.ASRFishing(state)
nu = actuation.nu

runningCostModel = crocoddyl.CostModelSum(state,nu)
terminalCostModel = crocoddyl.CostModelSum(state,nu)
xResidual = crocoddyl.ResidualModelState(state, state.zero(), nu)
uResidual = crocoddyl.ResidualModelControl(state, nu)
uRegCost = crocoddyl.CostModelResidual(state, uResidual)
fixed_length = 1.1

# L0 = 3.3666667 - 0.15 #ok
# alpha_des = np.pi/8 #ok
# vel_cos = 1 #1 ok

# L0 = 3.3666667 - 0.15 #ok
# alpha_des = np.pi/8 #ok
# vel_cos = 0 # 1 # ok

L0 = 3.3666667 - 0.05 #ok 0.1
alpha_des = np.pi/8 #ok
vel_cos = 0 # 6 # 1 5_2 # ok
target_pos = np.array([L0*np.cos(alpha_des), 0, L0*np.sin(alpha_des)])
target_vel = np.array([vel_cos*np.cos(alpha_des), 0, vel_cos*np.cos(alpha_des)])

framePlacementResidual = crocoddyl.ResidualModelFramePlacement(state, robot_model.getFrameId("Link_EE"),
                                                               pinocchio.SE3(np.eye(3), target_pos),nu)
framePlacementVelocity = crocoddyl.ResidualModelFrameVelocity(state, robot_model.getFrameId("Link_EE"),
                                                               pinocchio.Motion(target_vel,np.zeros(3)),
                                                               pinocchio.ReferenceFrame(state.nv*2 + 2).WORLD, nu)
                                                               # pinocchio.ReferenceFrame(45).WORLD, nu)

xActivation = crocoddyl.ActivationModelWeightedQuad(np.array([1e1] * state.nv + [1e0] * state.nv)) # 1e1
xResidual = crocoddyl.ResidualModelState(state, state.zero(), nu)
xRegCost = crocoddyl.CostModelResidual(state, xActivation, xResidual)

goalTrackingCost = crocoddyl.CostModelResidual(state, framePlacementResidual)
goalVelCost = crocoddyl.CostModelResidual(state, framePlacementVelocity)
xRegCost = crocoddyl.CostModelResidual(state, xResidual)

# Then let's added the running and terminal cost functions
# runningCostModel.addCost("gripperPose", goalTrackingCost, 1e2) #1e3
# runningCostModel.addCost("gripperVel", goalVelCost, 1e2) # 1e1 # 1e0
# runningCostModel.addCost("xReg", xRegCost, 1e1) # 1e1
# # runningCostModel.addCost("uReg", uRegCost, 1e-4) # 1e-3 # 1e-1 # increase to decrease the cost of the control
# runningCostModel.addCost("uReg", uRegCost, 1e1) # increase to decrease the cost of the control
# terminalCostModel.addCost("gripperPose", goalTrackingCost, 1e3) # 1e2
# terminalCostModel.addCost("gripperVel", goalVelCost, 1e3) #1e2

# runningCostModel.addCost("gripperPose", goalTrackingCost, 1e1)
# runningCostModel.addCost("gripperVel", goalVelCost, 1e1)
# runningCostModel.addCost("xReg", xRegCost, 1e1)
# runningCostModel.addCost("uReg", uRegCost, 1e1)
# terminalCostModel.addCost("gripperPose", goalTrackingCost, 1e3)
# terminalCostModel.addCost("gripperVel", goalVelCost, 1e3)

runningCostModel.addCost("gripperPose", goalTrackingCost, 1e1)
runningCostModel.addCost("gripperVel", goalVelCost, 1e-1)
runningCostModel.addCost("xReg", xRegCost, 1e-2)
runningCostModel.addCost("uReg", uRegCost, 1e1) # increase to decrease the cost of the control
terminalCostModel.addCost("gripperPose", goalTrackingCost, 1e1) 
terminalCostModel.addCost("gripperVel", goalVelCost, 1e-1) 

dt = 1e-4
T = 10000 # 20000
max_step = T
T_f = T * dt
# Up to ILC_DATA_2
# k_11 = 40.1503 # 2 *
k_ii = 1.3 * np.array([0,34.6099,26.8363,17.2029,11.8908,8.9892,\
        12.6067,8.8801,4.0401,3.6465,3.0458,6.3951,3.6361,2.6307,\
        3.0194,1.1336,1.5959,1.3670,0.7066,0.5041,0.412])

d_ii = 3e1 * np.array([0.0191,0.0164,0.0127,0.0082,0.0056,0.0043,\
        0.0060,0.0042,0.0019,0.0017,0.0014,0.0030, 0.0017,0.0012,0.0014,\
        0.0005, 0.0008, 0.0006,0.0003,0.0002,0.0000])

D = np.diag(d_ii)
K = np.diag(k_ii)

runningModel = crocoddyl.IntegratedActionModelEuler(
  aslr_to.DAM2(state, actuation, runningCostModel, K, D), dt)
terminalModel = crocoddyl.IntegratedActionModelEuler(
  aslr_to.DAM2(state, actuation, terminalCostModel, K, D), 0) # dt) # need to rescale the problem

u_max = 40
runningModel.u_lb = np.array([-u_max])
runningModel.u_ub = np.array([u_max])

# q0 = fishing_rod.q0
q0 = np.zeros(state.nv)
q0[0] = np.pi/2
x0 = np.concatenate([q0,pinocchio.utils.zero(state.nv)])
toll = 1
problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)

# Creating the DDP solver for this OC problem, defining a logger
# solver = crocoddyl.SolverFDDP(problem)
solver = crocoddyl.SolverBoxFDDP(problem)
# cameraTF = [2., 2.68, 0.54, 0.2, 0.62, 0.72, 0.22]
cameraTF = [5.2, 5.2, 1.8, 0.2, 0.62, 0.72, 0.22]

if WITHDISPLAY:
    display = crocoddyl.GepettoDisplay(fishing_rod)
    display.robot.viewer.gui.addSphere('world/point', .05, [1., 0., 0., 1.])  # radius = .1, RGBA=1001
    display.robot.viewer.gui.applyConfiguration('world/point',
                                            target_pos.tolist() + [0., 0., 0., 1.])  # xyz+quaternion
    display.robot.viewer.gui.refresh()
    solver.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose(), crocoddyl.CallbackDisplay(display)])
else:
    solver.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])

xs = [x0] * (solver.problem.T + 1)
u = np.zeros(1)
us = [u] * (solver.problem.T)
solver.th_stop = 1e-5
# Solving it with the DDP algorithm
# feasibility ( last column ) must be 0
solver.solve(xs, us, 100) # 50
print('=============================================================================================')
print('Reached Pos: {}\nReached Vel: {}'.format(solver.problem.terminalData.differential.multibody.pinocchio.oMf[robot_model.getFrameId(
    "Link_EE")].translation.T, \
        pinocchio.getFrameVelocity(solver.problem.terminalModel.differential.state.pinocchio,\
            solver.problem.terminalData.differential.multibody.pinocchio, robot_model.getFrameId("Link_EE")).linear))
print('Desired Pos: {}\nDesired Vel: {}'.format(target_pos,target_vel))
print('=============================================================================================')
final_pos = solver.problem.terminalData.differential.multibody.pinocchio.oMf[robot_model.getFrameId("Link_EE")].translation.T
err = LA.norm(final_pos - target_pos, 2)
print('Final Position error = ', err)
print('=============================================================================================')
# print('Initial position = ', solver.problem.runningDatas.tolist()[0].differential.multibody.pinocchio.oMf[robot_model.getFrameId(
#     "Link_EE")].translation.T)
# print('=============================================================================================')
if err < toll:
    print(colored("Error is OK", "green"))
    ILC_TRUE = True 
else:
    print(colored("Error is not OK.\nDo NOT start the ILC", 'red'))
    ILC_TRUE = False
    # sys.exit()
# print('=============================================================================================')

log = solver.getCallbacks()[0]
# solver.K
u_ddp_out = np.array([]) # only the first joint is acutated
for i in range(len(log.us)):
    u_ddp_out = np.append(u_ddp_out,log.us[i][0])
data_q = np.zeros([len(log.xs),state.nq])
data_q_dot = np.zeros([len(log.xs),state.nq])
for i in range(len(log.xs)):
    for j in range(state.nq):
        data_q[i][j] = log.xs[i][j]
    for j1 in range(state.nq):
        data_q_dot[i][j1] = log.xs[i][j1 + state.nq]
print('=============================================================================================')
MSE = np.square(u_ddp_out).mean()
u_rms = math.sqrt(MSE)
print('DDP: {}\t Max: {} \t Min: {}'.format(u_rms, np.max(u_ddp_out), np.min(u_ddp_out)))
# Visualizing the solution in gepetto-viewer
if ILC_TRUE:
    cont_plot = 0
    n_time_display = 2
    if WITHDISPLAY:
        while cont_plot < n_time_display:
            display.displayFromSolver(solver)
            display.robot.viewer.gui.addSphere('world/point', .05, [1., 0., 0., 1.])  # radius = .1, RGBA=1001
            display.robot.viewer.gui.applyConfiguration('world/point',
                                                        target_pos.tolist() + [0., 0., 0., 1.])  # xyz+quaternion
            display.robot.viewer.gui.refresh()
            time.sleep(dt)
            cont_plot+=1
else:
    if WITHDISPLAY:
        while True:
            display.displayFromSolver(solver)
            display.robot.viewer.gui.addSphere('world/point', .05, [1., 0., 0., 1.])  # radius = .1, RGBA=1001
            display.robot.viewer.gui.applyConfiguration('world/point',
                                                    target_pos.tolist() + [0., 0., 0., 1.])  # xyz+quaternion
            display.robot.viewer.gui.refresh()
            time.sleep(dt)

print('=============================================================================================')
print('====================== WE HAD CREATED THE REFERENCE FOR THE ILC =============================')
print('=============================================================================================')

t = np.arange(0,T*dt,dt)
if ILC_TRUE:
    pos_x = np.zeros(T)
    vel_x = np.zeros(T)
    acc_x = np.zeros(T)
    pos_y = np.zeros(T)
    pos_z = np.zeros(T)
    for t, (m, d) in enumerate(zip(solver.problem.runningModels, solver.problem.runningDatas)):
        q = d.xnext[:state.nv]
        q_dot = d.xnext[-state.nv:]
        pinocchio.computeAllTerms(m.differential.state.pinocchio,d.differential.multibody.pinocchio,q,q_dot)
        pinocchio.forwardKinematics(m.differential.state.pinocchio, d.differential.multibody.pinocchio, q, q_dot)
        pos = d.differential.multibody.pinocchio.oMf[robot_model.getFrameId("Link_EE")].translation.T
        pos_x[t] = pos[0]
        pos_y[t] = pos[1]
        pos_z[t] = pos[2]
        vel = pinocchio.getFrameVelocity(m.differential.state.pinocchio,d.differential.multibody.pinocchio, robot_model.getFrameId("Link_EE")).linear
        vel_x[t] = vel[0]
        acc = pinocchio.getFrameAcceleration(m.differential.state.pinocchio,d.differential.multibody.pinocchio, robot_model.getFrameId("Link_EE")).linear
        acc_x[t] = acc[0]
    
    # traj_des = [np.zeros(1)]*T
    # traj_des_dot = [np.zeros(1)]*T
    # traj_des_ddot = [np.zeros(1)]*T
    # u_0 = [np.zeros(1)]*T
    t = np.arange(0,T*dt,dt)

    K_fb = solver.K.tolist()
    K_temp_fb = []
    for i in range(len(K_fb)):
        K_temp_fb.append(np.linalg.norm(K_fb[i]))

    # plt.figure(clear=True)
    # plt.ylabel(r'Feedback Gain')
    # plt.xlabel(r'Time $[s]$')
    # plt.plot(t, K_temp_fb, color='red',label='pos', linestyle='-',linewidth=2)
    # plt.grid()
    # plt.show()

    # plt.figure(clear=True)
    # plt.ylabel(r'References')
    # plt.xlabel(r'Time $[s]$')
    # plt.plot(t, pos_x, color='red',label='pos', linestyle='-',linewidth=2)
    # # plt.plot(t, vel_x, color='green',label='vel', linestyle='-',linewidth=2)
    # # plt.plot(t, acc_x, label='acc')
    # # plt.legend(loc='best')
    # plt.grid()
    # plt.show()

    # plt.figure(clear=True)
    # plt.ylabel(r'References Vel')
    # plt.xlabel(r'Time $[s]$')
    # plt.plot(t, vel_x, color='green',label='vel', linestyle='-',linewidth=2)
    # plt.grid()
    # plt.show()
    # plt.figure(clear=True)
    # plt.ylabel(r'Control Action $[Nm]$')
    # plt.xlabel(r'Time $[s]$')
    # plt.plot(t, log.us, color='green', linestyle='-',linewidth=2)
    # plt.grid()
    # plt.show()

    target_pos = np.asarray(final_pos)
    # y_des = pos_x
    # y_dot_des = vel_x
    # y_ddot_des = acc_x
    y_des = data_q[:-1,0]
    y_dot_des = data_q_dot[:-1,0]
    y_ddot_des = np.gradient(data_q_dot[:-1,0], dt)
    # print('Time: {}\nInput:\n{}\n{}\n{}'.format(np.shape(t),np.shape(y_des), np.shape(y_dot_des), np.shape(y_ddot_des)))
    
    u0 = u_ddp_out
    x_des = log.xs
    r = 2
    directory = 'ILC_DATA_{}_{}_q1_out_2'.format(str(T),str(dt))
    
    dir_plot_ddp = './' + directory + '_init' + '/DDP_init/'
    dir_plot_ddp_first = dir_plot_ddp
    if os.path.exists(dir_plot_ddp_first):
        shutil.rmtree(dir_plot_ddp_first)
    os.makedirs(dir_plot_ddp_first)
        
    np.savetxt(dir_plot_ddp_first + 'q_ddp.csv',data_q)
    np.savetxt(dir_plot_ddp_first + 'q_dot_ddp.csv',data_q_dot)
    np.savetxt(dir_plot_ddp_first + 'K_ddp_fb.csv', K_temp_fb)
    np.savetxt(dir_plot_ddp_first + 'u_ddp.csv',u_ddp_out)
    np.savetxt(dir_plot_ddp_first + 'pos_ddp_xyz.csv', pos)
    np.savetxt(dir_plot_ddp_first + 'vel_ddp_xyz.csv', vel)
    np.savetxt(dir_plot_ddp_first + 'acc_ddp_xyz.csv', acc)

    ilc = ILC_OnlyPinocchioFishing(q0,
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
                            pos_x,
                            pos_y,
                            pos_z,
                            directory)

    ilc.main()

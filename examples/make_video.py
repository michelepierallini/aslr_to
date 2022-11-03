from __future__ import print_function
import os
import sys
from matplotlib.ft2font import LOAD_IGNORE_GLOBAL_ADVANCE_WIDTH
import crocoddyl
import pinocchio
import numpy as np
import example_robot_data
import aslr_to
import time
from numpy import linalg as LA
import matplotlib.pyplot as plt
from ILC_OnlyPinocchio import ILC_OnlyPinocchio

import cv2
import numpy as np
import pyautogui

# PATH_DOT = '/ILC_DATA_10000_0.0001_5_3_DONE/'
# PATH_DOT = '/ILC_DATA_10000_0.0001_5_1_DONE/'
PATH_DOT = '/ILC_DATA_10000_0.0001_4_DONE/'
def video(action, y_des, y_dot_des,y_ddot_des, pos_y, pos_z, directory):

    dt = 1e-4
    T = 10000
    max_step = T
    t = np.arange(0,T*dt,dt)
    FISHING_STRING = 'fishing_rod_two'
    q0 = np.zeros(21)
    q0[0] = np.pi/2

    # k_ii = 1 * np.array([40.1503,34.6099,26.8363,17.2029,11.8908,8.9892,\
    #         12.6067,8.8801,4.0401,3.6465,3.0458,6.3951,3.6361,2.6307,\
    #         3.0194,1.1336,1.5959,1.3670,0.7066,0.5041,0.0412])

    # d_ii = 8 * np.array([0.0191,0.0164,0.0127,0.0082,0.0056,0.0043,\
    #         0.0060,0.0042,0.0019,0.0017,0.0014,0.0030, 0.0017,0.0012,0.0014,\
    #         0.0005, 0.0008, 0.0006,0.0003,0.0002,0.0000])

    k_ii = 1.3 * np.array([0,34.6099,26.8363,17.2029,11.8908,8.9892,\
        12.6067,8.8801,4.0401,3.6465,3.0458,6.3951,3.6361,2.6307,\
        3.0194,1.1336,1.5959,1.3670,0.7066,0.5041,0.412])

    d_ii = 3e1 * np.array([0.0191,0.0164,0.0127,0.0082,0.0056,0.0043,\
        0.0060,0.0042,0.0019,0.0017,0.0014,0.0030, 0.0017,0.0012,0.0014,\
        0.0005, 0.0008, 0.0006,0.0003,0.0002,0.0000])

    L0 = 3.3666667 - 0.05 #ok
    alpha_des = 0 #ok
    target_pos = np.array([L0*np.cos(alpha_des), 0, L0*np.sin(alpha_des)])
    # target_pos = np.asarray([3.34363755, 0.0, -0.14461393])

    ilc_video = ILC_OnlyPinocchio(q0,
                            k_ii, 
                            d_ii, 
                            1, 
                            2, 
                            action, 
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

    viz = crocoddyl.GepettoDisplay(ilc_video.robot_display, 4, 4, ilc_video.cameraTF)
    action = action
    q = list(ilc_video.q0)
    q_dot = list(ilc_video.dq0)
    ilc_video.computeWhatINeed(ilc_video.q0, ilc_video.dq0, action[0])
    y = ilc_video.data.pos[0]
    y_app = []
    y_dot = ilc_video.data.vel[0]
    y_ddot = ilc_video.data.acc[0]
    # time.sleep(10) 
    conta = 0
    salta_step = 100 #10


    for i in range(0, ilc_video.steps): # time 
        action_plug = action[i]
        u_fb = ilc_video.K_P * (ilc_video.y_des[i] - y) + ilc_video.K_V * (ilc_video.y_dot_des[i] - y_dot) # + ilc_video.K_A * (ilc_video.y_ddot_des[i] - y_ddot)
        control_action = action_plug + u_fb

        if abs(control_action) > ilc_video.max_actuator:
            # saturation
            control_action = np.sign(ilc_video.max_actuator)

        ilc_video.computeWhatINeed(q, q_dot, control_action)
        q, q_dot = ilc_video.forwardEulerMethod(q, q_dot)
       
        viz.robot.display(np.array(q))
        # traj
        viz.robot.viewer.gui.addSphere('world/point', 0.01, [13.0, 180.0, 185.0, 1.0])  # radius = .1, RGBA=1001
        viz.robot.viewer.gui.applyConfiguration('world/point',
            np.array([ilc_video.y_des[i], ilc_video.pos_y[i], ilc_video.pos_z[i]]).tolist() + [0.0, 0.0, 0.0, 1.0])  # xyz+quaternion
        viz.robot.viewer.gui.refresh()
        y = ilc_video.data.pos[0]
        y_app.append(y_app)
        if conta > salta_step :
            # time.sleep(ilc_video.dt) 
            img = pyautogui.screenshot()
            # Convert the image into numpy array
            img = np.array(img)
            # Convert the color space from BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # _, frame = webcam.read()
            # Finding the width, height and shape of our webcam image
            # fr_height, fr_width, _ = frame.shape
            # # setting the width and height properties
            # img[0:fr_height, 0: fr_width, :] = frame[0:fr_height, 0: fr_width, :]
            # #cv2.imshow('frame', img)
            # Write the frame into the file 'output.avi'
            out.write(img)
            # Press 'q' to quit
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     print("Recording Stopped")
            #     break
            conta = 0
        conta += 1
    out.release()
    cv2.destroyAllWindows()

    plt.figure(clear=True)
    plt.ylabel(r'Tip Pos $X$ $[m]$')
    plt.xlabel(r'Time $[s]$')
    plt.plot(ilc_video.time, ilc_video.y_des,color='blue',label='Des')
    plt.plot(ilc_video.time, y_app, color='red',label='Real')
    plt.grid()
    plt.show()
    plt.legend(loc='best', shadow=True, fontsize='10')
    # plt.close()

if __name__ == '__main__':
    SCREEN_SIZE = tuple(pyautogui.size())
    resolution = (960, 540)
    path = os.getcwd() + PATH_DOT
    # action = np.loadtxt(path + 'DDP_u_ddp.csv', delimiter=',')
    # action = np.loadtxt(path + 'Iter_18/action_ff.csv', delimiter=',')
    # action = np.loadtxt(path + 'DDP_/u_ddp.csv', delimiter=',')
    action = np.loadtxt(path + 'Iter_48/action_ff.csv', delimiter=',')
    # plt.plot(action, color='red')
    # plt.show()
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter(path + 'u_ddp.avi', fourcc, 100, (SCREEN_SIZE))
    out = cv2.VideoWriter(path + 'ilc_test_saroj.avi', fourcc, 100, (SCREEN_SIZE))
    # webcam = cv2.VideoCapture(0)  

    y_des = np.loadtxt(path + 'DDP_/y_des.csv', delimiter=',')
    y_dot_des = np.loadtxt(path + 'DDP_/y_dot_des.csv', delimiter=',')
    y_ddot_des = np.loadtxt(path + 'DDP_/y_ddot_des.csv', delimiter=',')
    pos_y = np.loadtxt(path + 'DDP_/pos_y.csv', delimiter=',') 
    pos_z = np.loadtxt(path + 'DDP_/pos_z.csv', delimiter=',') 
    directory = 'fake'
 
    video(action, y_des, y_dot_des, y_ddot_des, pos_y, pos_z, directory)
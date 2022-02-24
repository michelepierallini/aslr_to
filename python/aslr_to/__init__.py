from .statemultiasr import StateMultiASR
from .contact_fwddyn_aslr import (DifferentialContactASLRFwdDynModel, DifferentialContactASLRFwdDynData)
from .residual_frame_placement import (ResidualModelFramePlacementASR,ResidualDataFramePlacementASR)
from .floating_actuation import ASRFreeFloatingActuation
from .contact_fwddyn_rigid import (DifferentialContactFwdDynModelRigid,DifferentialContactFwdDynDataRigid)
from .floating_actuation_condensed import FreeFloatingActuationCondensed
from .floating_soft_model_residual import (FloatingSoftDynamicsResidualModel, FloatingSoftDynamicsResidualData)
from .floating_vsa_model_residual import (FloatingVSADynamicsResidualModel,FloatingVSADynamicsResidualData)
from .solver import DDPASLR
import numpy as np
import crocoddyl
import pinocchio
import time
import warnings
import matplotlib.pyplot as plt

def plot_theta(log, K):
    for i in [-1]:
        for k in range(12):
            theta = []
            for j in range(len(log.residual[i])):
                theta.append(log.residual[i][j][k]/K)
            plt.xlabel('Nodes')
            plt.ylabel( "theta ")
            plt.title('')
            plt.plot(theta,label="theta_"+str(k))
            plt.legend()
        plt.show()

def u_squared(log):
    u_sqaured = np.zeros(len(log.us[0]))
    for i in range(len(log.us)):
        for j in range(len(log.us[0])):
            u_sqaured[j] += abs(log.us[i][j])
    return u_sqaured

from __future__ import print_function
import os
import sys
import crocoddyl
import pinocchio
import numpy as np
import example_robot_data
import aslr_to
import time
from numpy import linalg as LA

class DAM_Fish(crocoddyl.DifferentialActionModelAbstract):

    def __init__(self, state, actuationModel, K=None, D=None):

        crocoddyl.DifferentialActionModelAbstract.__init__(self, state, actuationModel.nu)
        
        self.actuation = actuationModel
        self.enable_force = True

        if K is None:
            self.K = 1e2*np.eye(state.nv)
        else:
            self.K = K
        if D is None:
            self.D = 1e-3*np.eye(state.nv)
        else:
            self.D = D

    def calc(self, data, x, u=None):

        if u is None:
            u = np.zeros(self.nu)

        nq = self.state.nq
        nv = self.state.nv

        q = x[:nq]
        v = x[-nv:]

        self.actuation.calc(data.actuation, x, u)
        tau = data.actuation.tau

        # Computing the fwd dynamics manually
        pinocchio.computeAllTerms(self.state.pinocchio, data.pinocchio, q, v)
        data.M = data.pinocchio.M
        data.Minv = np.linalg.inv(data.M)

        data.xout[:] = np.dot(data.Minv, (tau - data.pinocchio.nle - np.dot(self.K,q) - np.dot(self.D,v)))

        # Computing the cost value and residuals
        pinocchio.forwardKinematics(self.state.pinocchio, data.pinocchio, q, v)
        pinocchio.updateFramePlacements(self.state.pinocchio, data.pinocchio)
        data.pos = data.multibody.pinocchio.oMf[self.state.pinocchio.getFrameId("Link_EE")].translation.T
        data.vel = pinocchio.getFrameVelocity(self.state.pinocchio,data.multibody.pinocchio, \
            self.state.pinocchio.getFrameId("Link_EE")).linear
        # pinocchio.updateFramePlacements(self.state.pinocchio, data.pinocchio)

    def createData(self):
        data = DAD_Fish(self)
        return data

class DAD_Fish(crocoddyl.DifferentialActionDataAbstract):
    def __init__(self, model):
        crocoddyl.DifferentialActionDataAbstract.__init__(self, model)
        self.pinocchio = pinocchio.Model.createData(model.state.pinocchio)
        self.multibody = crocoddyl.DataCollectorMultibody(self.pinocchio)
        self.actuation = model.actuation.createData()
        self.tmp_xstatic = np.zeros(model.state.nx)
        self.tmp_ustatic = np.zeros(model.nu)
        self.tmp_kine = np.zeros(6)
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.Minv = None


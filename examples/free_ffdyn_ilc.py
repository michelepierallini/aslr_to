import numpy as np
import pinocchio
import crocoddyl


class DAMILC(crocoddyl.DifferentialActionModelAbstract):
    def __init__(self, state, actuationModel, costModel, K=None, D=None):
        crocoddyl.DifferentialActionModelAbstract.__init__(self, state, actuationModel.nu, costModel.nr)
        self.actuation = actuationModel
        self.enable_force = True

        if K is None:
            self.K = 1e-1*np.eye(state.nv)
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
        # x_out = [ddq]
        data.xout[:] = np.dot(data.Minv, (tau - data.pinocchio.nle - np.dot(self.K,q)- np.dot(self.D,v)))

        # Computing the cost value and residuals
        pinocchio.forwardKinematics(self.state.pinocchio, data.pinocchio, q, v)
        pinocchio.updateFramePlacements(self.state.pinocchio, data.pinocchio)


    def createData(self):
        data = DADILC(self)
        return data

class DADILC(crocoddyl.DifferentialActionDataAbstract):
    def __init__(self, model):
        crocoddyl.DifferentialActionDataAbstract.__init__(self, model)
        self.pinocchio = pinocchio.Model.createData(model.state.pinocchio)
        self.multibody = crocoddyl.DataCollectorMultibody(self.pinocchio)
        self.actuation = model.actuation.createData()
        self.Minv = None
        self.Binv = None
        self.tmp_xstatic = np.zeros(model.state.nx)
        self.tmp_ustatic = np.zeros(model.nu)


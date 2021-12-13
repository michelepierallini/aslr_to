import numpy as np
import pinocchio
import crocoddyl

class DifferentialContactASLRFwdDynModel(crocoddyl.DifferentialActionModelAbstract):
    def __init__(self, state, actuation, contacts, costs, constraints=None):
        nu =  actuation.nu 
        crocoddyl.DifferentialActionModelAbstract.__init__(self, state, nu, costs.nr)

        self.actuation = actuation
        self.costs = costs
        self.contacts = contacts

    def calc(self, data, x, u):
        if len(x) != self.state.nx:
            raise Exception("Invalid argument: u has wrong dimension (it should be " + str(self.state.nx))
        if len(u) != self.nu:
            raise Exception("Invalid argument: u has wrong dimension (it should be " + str(self.nu)+"it is "+ str(len(u)))

        nc = self.contacts.nc
        nq_l = self.state.nq_l
        nv_l = self.state.nv_l
        nl = nq_l + nv_l
        q_l = x[:nq_l]
        v_l = x[nq_l:nl]
        q_m = x[nl:-self.state.nv_m]

        x_l = x[:nl]
        data.tau_couple = np.dot(data.K, q_l-np.hstack([np.zeros(7),q_m]))

        pinocchio.computeAllTerms(self.state.pinocchio, data.multibody.pinocchio, q_l, v_l)
        pinocchio.computeCentroidalDynamics(self.state.pinocchio, data.multibody.pinocchio, q_l, v_l)

        self.actuation.calc(data.multibody.actuation, x, u)

        self.contacts.calc(data.multibody.contacts, x_l)
        data.Binv = np.linalg.inv(data.B)
        tau = data.multibody.actuation.tau
        JMinvJt_damping_=0
        pinocchio.forwardDynamics(self.state.pinocchio, data.multibody.pinocchio, - data.tau_couple, data.multibody.contacts.Jc[:nc,:self.state.nv_l],
                        data.multibody.contacts.a0[:nc], JMinvJt_damping_)

        data.xout[:nv_l] = data.multibody.pinocchio.ddq
        data.xout[nv_l:] =  np.dot(data.Binv, tau[-self.state.nv_m:] + data.tau_couple[-self.state.nv_m:])
        self.contacts.updateAcceleration(data.multibody.contacts, data.xout[:nv_l])

        self.contacts.updateForce(data.multibody.contacts, data.multibody.pinocchio.lambda_c)
        self.costs.calc(data.costs, x[:self.state.nx], u)
        data.cost = data.costs.cost

    def calcDiff(self, data, x, u):

        nq_l = self.state.nq_l
        nv_l = self.state.nv_l
        nv_m = self.state.nv_m
        nl = nq_l + nv_l
        nc = self.contacts.nc

        q_l = x[:nq_l]
        v_l = x[nq_l:nl]
        x_l = x[:nl]

        pinocchio.computeRNEADerivatives(self.state.pinocchio, data.multibody.pinocchio, q_l, v_l, data.xout[:nv_l],
                                         data.multibody.contacts.fext)
        data.Kinv = pinocchio.getKKTContactDynamicMatrixInverse(self.state.pinocchio, data.multibody.pinocchio, data.multibody.contacts.Jc[:nc,:])
        self.actuation.calcDiff(data.multibody.actuation, x, u)
        self.contacts.calcDiff(data.multibody.contacts, x_l)

        a_partial_dtau = data.Kinv[:nv_l,:nv_l]
        a_partial_da = data.Kinv[:nv_l,-nc:]

        f_partial_dtau = data.Kinv[-nc:,:nv_l]
        f_partial_da = data.Kinv[-nc:,-nc:]

        data.Fx[:nv_l,:nv_l] = -np.dot(a_partial_dtau,data.multibody.pinocchio.dtau_dq + data.K[:,-nv_l:])
        data.Fx[:nv_l,nv_l:2*nv_l] = -np.dot(a_partial_dtau, data.multibody.pinocchio.dtau_dv)
        data.Fx[:nv_l,2*nv_l:-nv_m] = np.dot(a_partial_dtau,data.K[:,-self.state.nv_m:])

        data.Fx[:nv_l,:2*nv_l] -=   np.dot(a_partial_da, data.multibody.contacts.da0_dx[:nc,:])

        data.Fx[nv_l:, :nv_l] = np.dot(data.Binv,data.K[-self.actuation.nu:,-nv_l:])

        data.Fx[nv_l:, 2*nv_l:-nv_m] = -np.dot(data.Binv,data.K[-self.actuation.nu:, -self.actuation.nu:])
        
        #data.Fx += np.dot(a_partial_dtau,data.multibody.actuation.dtau_dx)
        data.Fu[nv_l:, :] = np.dot(data.Binv, data.multibody.actuation.dtau_du[nv_l:, :])

        data.df_dx[:nc, :nv_l] = np.dot(f_partial_dtau, data.multibody.pinocchio.dtau_dq)
        data.df_dx[:nc, nv_l:2*nv_l] = np.dot(f_partial_dtau, data.multibody.pinocchio.dtau_dv)
        data.df_dx[:nc, 2*nv_l:-nv_m] = np.dot(f_partial_dtau,data.K[:,-self.state.nv_m:])
        data.df_dx[:nc, :nv_l] += np.dot(f_partial_da, data.multibody.contacts.da0_dx[:nc,:nv_l])
        data.df_dx[:nc, nv_l:2*nv_l] += np.dot(f_partial_da, data.multibody.contacts.da0_dx[:nc,nv_l:])

        data.df_du[:nc,: ] = -np.dot(f_partial_dtau[:,-self.state.nv_m:], data.multibody.actuation.dtau_du[nv_l:, :])
        self.contacts.updateAccelerationDiff(data.multibody.contacts, data.Fx[-nv_l:,:2*nv_l])
        self.contacts.updateForceDiff(data.multibody.contacts, data.df_dx[:nc,:2*nv_l], data.df_du[:nc,:])
        
        self.costs.calcDiff(data.costs, x[:self.state.nx], u)

    def quasistatic(self, data, x):
        if len(x) != self.state.nx:
            raise Exception("Invalid argument: u has wrong dimension (it should be " + self.state.nx)
        
        nq, nv, na, nc = self.state.nq, self.state.nv, self.actuation.nu, self.contacts.nc
        data.tmp_xstatic[:nq] = x[:nq]
        data.tmp_xstatic[nq:] *= 0.
        data.tmp_ustatic[:] *= 0.

        pinocchio.computeAllTerms(self.state.pinocchio, data.multibody.pinocchio, data.tmp_xstatic[:nq],
                                  data.tmp_xstatic[nq:])
        pinocchio.computeJointJacobians(self.state.pinocchio, data.multibody.pinocchio, data.tmp_xstatic[:nq])
        pinocchio.rnea(self.state.pinocchio, data.multibody.pinocchio, data.tmp_xstatic[:nq], data.tmp_xstatic[nq:],
                       data.tmp_xstatic[nq:])
        self.actuation.calc(data.multibody.actuation, data.tmp_xstatic, data.tmp_ustatic[:])
        self.actuation.calcDiff(data.multibody.actuation, data.tmp_xstatic, data.tmp_ustatic[:])
        if nc != 0:
            self.contacts.calc(data.multibody.contacts, data.tmp_xstatic)
            data.tmp_Jstatic[:, :na] = data.multibody.actuation.dtau_du.reshape(nv, na)
            data.tmp_Jstatic[:, na:na + nc] = data.multibody.contacts.Jc[:nc, :].T
            data.tmp_ustatic[:] = np.dot(np.linalg.pinv(data.tmp_Jstatic[:, :na + nc]),
                                                  data.multibody.pinocchio.tau)[:na]
            data.multibody.pinocchio.tau[:] *= 0
            return data.tmp_ustatic
        else:
            data.tmp_ustatic[nv:nv + na] = np.dot(np.linalg.pinv(data.multibody.actuation.dtau_du.reshape(nv, na)),
                                                  data.multibody.pinocchio.tau)
            data.multibody.pinocchio.tau[:] *= 0.
            return data.tmp_ustatic

    def createData(self):
        data = DifferentialContactASLRFwdDynData(self)
        return data

class DifferentialContactASLRFwdDynData(crocoddyl.DifferentialActionDataAbstract):
    def __init__(self, model):
        crocoddyl.DifferentialActionDataAbstract.__init__(self, model)
        pmodel = pinocchio.Model.createData(model.state.pinocchio)
        actuation = model.actuation.createData()
        contacts = model.contacts.createData(pmodel)
        self.multibody = crocoddyl.DataCollectorActMultibodyInContact(pmodel, actuation, contacts)
        self.costs = model.costs.createData(self.multibody)
        self.costs.shareMemory(self)
        self.Minv = None
        self.Kinv = None
        self.K = np.zeros([model.state.pinocchio.nv,model.state.pinocchio.nq])
        nu = model.actuation.nu
        self.K[-nu:,-nu:]= 10*np.eye(nu)
        self.B = .01*np.eye(model.state.nv_m)
        self.df_dx =np.zeros([model.contacts.nc,model.state.ndx])
        self.df_du =np.zeros([model.contacts.nc, model.actuation.nu])

        self.tmp_xstatic = np.zeros(model.state.nx)
        self.tmp_ustatic = np.zeros(model.nu)
        self.tmp_Jstatic = np.zeros([model.state.nv, model.nu + model.contacts.nc_total])
import numpy as np
import pinocchio
import crocoddyl
import scipy.linalg as scl


#   state class for soft actuated robots. x = [q_l,q_m,v_l,v_m] . 
#   q_l : configuration vector of link side
#   q_m : configuration vector of motor side
#   v_l : q_dot of link side 
#   v_m : \theta_dot of the motor side

class StateMultiASR(crocoddyl.StateMultibody):
    def __init__(self, pinocchioModel):
        crocoddyl.StateMultibody.__init__(self, pinocchioModel)
        self.nq_m = pinocchioModel.nv-6
        self.nv_m = pinocchioModel.nv -6
        self.nv_l = pinocchioModel.nv
        self.nq_l = pinocchioModel.nq
        self.ndx_ = 2 * pinocchioModel.nv+2*(pinocchioModel.nv-6)
        self.nx_ =2*(pinocchioModel.nv-6) +(pinocchioModel.nq + pinocchioModel.nv)
        self.nq_ = self.nq_m +self.nq_l
        self.nv_ = self.nv_m +self.nv_l 
    def zero(self):
        q_l = pinocchio.neutral(self.pinocchio)
        v_l = pinocchio.utils.zero(self.pinocchio.nv)
        q_m = np.zeros(self.nq_m)
        v_m = np.zeros(self.nq_m)
        return np.concatenate([q_l,q_m, v_l, v_m])

    def rand(self):
        q_l = pinocchio.randomConfiguration(self.pinocchio)
        q_m = pinocchio.utils.rand(self.nq_m)
        v_l = pinocchio.utils.rand(self.pinocchio.nv)
        v_m = pinocchio.utils.rand(self.nq_m)
        return np.concatenate([q_l, q_m, v_l, v_m])

    def diff(self, x0, x1):
        nq_l = self.pinocchio.nq
        nv_l = self.pinocchio.nv
        nq = nq_l+ self.nq_m
        nv = nv_l +self.nv_m
        q0_l = x0[:nq_l]
        q0_m = x0[nq_l:nq]
        v0_l = x0[nq:-self.nv_m]
        v0_m = x0[-self.nv_m:]

        q1_l = x1[:nq_l]
        q1_m = x1[nq_l:nq]
        v1_l = x1[nq:-self.nv_m]
        v1_m = x1[-self.nv_m:]

        dq_l = pinocchio.difference(self.pinocchio, q0_l, q1_l)
        dq_m = q1_m - q0_m

        return np.concatenate([dq_l, dq_m, v1_l - v0_l, v1_m - v0_m])

    def integrate(self, x, dx):
        nq_l = self.pinocchio.nq
        nv_l = self.pinocchio.nv
        nq = nq_l+ self.nq_m
        nv = nv_l +self.nv_m
        q_l = x[:nq_l]
        q_m = x[nq_l:nq]
        v_l = x[nq:-self.nv_m]
        v_m = x[-self.nv_m:]
        dq_l = dx[:nv_l]
        dq_m = dx[nv_l:nv]
        dv_l = dx[nv:-self.nv_m]
        dv_m = dx[-self.nv_m:]

        qn_l = pinocchio.integrate(self.pinocchio, q_l, dq_l)
        qn_m = q_m+ dq_m
        return np.concatenate([qn_l, qn_m, v_l + dv_l, v_m + dv_m])

    def Jdiff(self, x1, x2, firstsecond=crocoddyl.Jcomponent.both):
        if firstsecond == crocoddyl.Jcomponent.both:
            return [self.Jdiff(x1, x2, crocoddyl.Jcomponent.first), self.Jdiff(x1, x2, crocoddyl.Jcomponent.second)]

        if firstsecond == crocoddyl.Jcomponent.first:
            nq_l = self.pinocchio.nq
            nv_l = self.pinocchio.nv
            dx = self.diff(x2, x1)

            q_l = x2[:nq_l]
            dq_l = dx[:nv_l]

            Jdq_l = pinocchio.dIntegrate(self.pinocchio, q_l, dq_l)[1]
            return np.matrix(-scl.block_diag(np.linalg.inv(Jdq_l), np.eye(self.nv_m), np.eye(self.nv_l),np.eye(self.nv_m)))
        elif firstsecond == crocoddyl.Jcomponent.second:
            nq_l = self.pinocchio.nq
            nv_l = self.pinocchio.nv
            dx = self.diff(x1, x2)

            q_l = x1[:nq_l]
            dq_l = dx[:nv_l]

            Jdq_l = pinocchio.dIntegrate(self.pinocchio, q_l, dq_l)[1]
            return np.matrix(scl.block_diag(np.linalg.inv(Jdq_l),np.eye(self.nv_m), np.eye(self.nv_l),np.eye(self.nv_m)))

    def Jintegrate(self, x, dx, firstsecond=crocoddyl.Jcomponent.both):
        if firstsecond == crocoddyl.Jcomponent.both:
            return [
                self.Jintegrate(x, dx, crocoddyl.Jcomponent.first),
                self.Jintegrate(x, dx, crocoddyl.Jcomponent.second)
            ]
        nq_l = self.pinocchio.nq
        nv_l = self.pinocchio.nv
        
        q_l = x[:nq_l]
        dq_l = dx[:nv_l]

        Jq_l, Jdq_l = pinocchio.dIntegrate(self.pinocchio, q_l, dq_l)
        if firstsecond == crocoddyl.Jcomponent.first:
            return np.matrix(scl.block_diag(np.linalg.inv(Jq_l), np.eye(self.nv_m), np.eye(self.nv_l),np.eye(self.nv_m)))
        elif firstsecond == crocoddyl.Jcomponent.second:
            return np.matrix(scl.block_diag(np.linalg.inv(Jdq_l), np.eye(self.nv_m), np.eye(self.nv_l),np.eye(self.nv_m)))

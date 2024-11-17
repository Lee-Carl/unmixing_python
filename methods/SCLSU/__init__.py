import numpy as np
from tqdm import tqdm
from custom_types import MethodBase
import copy


class SCLSU(MethodBase):

    def __init__(self, params, init):
        super().__init__(params, init)
        self.params = params
        self.init = init

    def run(self, savepath=None, tqdm_leave=True, *args, **kwargs):
        # 初始化参数
        init = copy.deepcopy(self.init)  # 不影响原始数据
        data = init['Y']
        S0 = init['E'].copy()  # 初始化权重-端元

        params = self.params
        maxiter_ADMM = params['maxiter_ADMM']
        rho = params['rho']
        tol_phi = params['tol_phi']

        # 核心程序
        [L, N] = data.shape
        [L, P] = S0.shape
        phi = np.ones([P, N])
        U = phi  # split variable
        D = np.zeros(phi.shape)  # Lagrange mutlipliers

        S0tX = np.dot(np.transpose(S0), data)
        S0tS0 = np.dot(np.transpose(S0), S0)
        I = np.identity(P)

        progress = tqdm(np.arange(maxiter_ADMM), leave=tqdm_leave)
        for i in progress:
            phi_old = phi
            phi = np.dot(np.linalg.inv(S0tS0 + rho * I), S0tX + rho * (U - D))
            U = np.maximum(U + D, 0)
            D = D + phi - U
            rel_phi = np.abs(
                (np.linalg.norm(phi, 'fro') - np.linalg.norm(phi_old, 'fro')) / np.linalg.norm(phi_old, 'fro'))
            progress.set_postfix_str(f"iteration {i} of {maxiter_ADMM} , rel_phi = {rel_phi}")
            # if rel_phi < tol_phi:
            #     break
        psi = np.sum(phi, axis=0)
        A = phi * np.reciprocal(psi)
        S = np.zeros([L, P, N])
        for i in np.arange(N):
            S[:, :, i] = psi[i] * S0
        return {
            "A": A,
            "E": S,
            "psi": psi
        }

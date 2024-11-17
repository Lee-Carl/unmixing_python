import numpy as np
from custom_types import MethodBase
from core.func.extract import extract_edm
import copy


def proj_simplex(data):
    """
    Created on Mon Dec 11 11:07:40 2017
    @author:  Lucas

    This module performs the projection of the columns of a PxN matrix on the unit simplex (with P vertices).

    Input: data matrix whose columns need to be projected on the simplex

    Output: projected data matrix

    """

    # matlab code::
    # proj_simplex_array = @(y) max(bsxfun(@minus,y,max(bsxfun(@rdivide,cumsum(sort(y,1,'descend'),1)-1,(1:size(y,1))'),[],1)),0); % projection on simplex

    data_sorted = np.sort(data, axis=0)[::-1,
                  :]  # sort rows of data array in descending order (by going through each column backwards)
    cumulative_sum = np.cumsum(data_sorted, axis=0) - 1  # cumulative sum of each row
    vector = np.arange(np.shape(data_sorted)[0]) + 1  # define vector to be divided elementwise
    divided = cumulative_sum / vector[:, None]  # perform the termwise division
    projected = np.maximum(data - np.amax(divided, axis=0), np.zeros(divided.shape))  # projection step

    return projected


class FCLSU(MethodBase):
    def __init__(self, params, init):
        super().__init__(params, init)
        self.params = params
        self.init = init

    def run(self, savepath=None, output_display=True, *args, **kwargs):
        """
            Created on Wed Nov 22 00:36:43 2017
            @author: Lucas

            Fully Constrained Least Squares Unmixing (FCLSU): unmix a hyperspectral image

            using the positivity and sum to one constraints on the abundances

            inputs: data: LxN data matrix (L: number of spectral bands, N: number of pixels) -- Y 图像
                    S0 : LxP reference endmember matrix (P: number of endmembers) -- 端元矩阵

            outputs: A: PxN abundance matrix

        """
        # 初始化参数
        init = copy.deepcopy(self.init)
        data = init['Y']
        S0 = init['E']  # 初始化权重-端元

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

        for i in np.arange(maxiter_ADMM):
            phi_old = phi
            phi = np.dot(np.linalg.inv(S0tS0 + rho * I), S0tX + rho * (U - D))
            # U = np.maximum(U+D,0)
            U = proj_simplex(U + D)
            D = D + phi - U
            rel_phi = np.abs(
                (np.linalg.norm(phi, 'fro') - np.linalg.norm(phi_old, 'fro')) / np.linalg.norm(phi_old, 'fro'))
            # print("iteration ", i, " of ", maxiter_ADMM, ", rel_phi =", rel_phi)
            if rel_phi < tol_phi:
                break
        A = phi
        E = extract_edm(y=self.init['Y'], a=A)
        return {
            "A": A,
            "E": E
        }

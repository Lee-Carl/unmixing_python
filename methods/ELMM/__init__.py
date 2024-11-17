import numpy as np
from custom_types import MethodBase


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


class ELMM(MethodBase):

    def __init__(self, params, init):
        super().__init__(params, init)
        self.params = params
        self.init = init

    def SCLSU(self, data, S0):
        # 初始化参数
        maxiter_ADMM = 1000
        rho = 1
        tol_phi = 10 ** -5

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
            U = np.maximum(U + D, 0)
            D = D + phi - U
            rel_phi = np.abs(
                (np.linalg.norm(phi, 'fro') - np.linalg.norm(phi_old, 'fro')) / np.linalg.norm(phi_old, 'fro'))
            # print("iteration ", i, " of ", maxiter_ADMM, ", rel_phi =", rel_phi)
            if rel_phi < tol_phi:
                break
        psi = np.sum(phi, axis=0)
        A = phi * np.reciprocal(psi)
        S = np.zeros([L, P, N])
        for i in np.arange(N):
            S[:, :, i] = psi[i] * S0
        return A, S, psi

    def FCLSU(self, data, S0):
        # 初始化参数
        maxiter_ADMM = 1000
        rho = 1
        tol_phi = 10 ** -5

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
            print("iteration ", i, " of ", maxiter_ADMM, ", rel_phi =", rel_phi)
            # if rel_phi < tol_phi:
            #     break
        A = phi
        return A

    def run(self, savepath=None, output_display=True, *args, **kwargs):
        # 载入数据及参数
        init = self.init.copy()
        data = init['Y']
        S0 = init['E'].copy()  # 初始化权重-端元

        A_init = self.FCLSU(data.copy(), S0.copy())

        params = self.params
        lambda_S = params['lambda_S']

        # 核心程序
        [L, N] = data.shape
        [L, P] = S0.shape
        # A = np.zeros([P,N])
        A = np.copy(A_init)
        S = np.zeros([L, P, N])
        psi = np.ones([P, N])
        for n in np.arange(N):
            S[:, :, n] = S0

        maxiter = params['maxiter']
        U = A  # split variable
        D = np.zeros(A.shape)  # Lagrange mutlipliers
        rho = params['rho']
        maxiter_ADMM = params['maxiter_ADMM']
        tol_A_ADMM = params['tol_A_ADMM']
        tol_A = params['tol_A']
        tol_S = params['tol_S']
        tol_psi = params['tol_psi']

        I = np.identity(P)

        for i in np.arange(maxiter):

            A_old = np.copy(A)
            psi_old = np.copy(psi)
            S_old = np.copy(S)

            # A update

            for j in np.arange(maxiter_ADMM):

                A_old_ADMM = np.copy(A)

                for n in np.arange(N):
                    A[:, n] = np.dot(np.linalg.inv(np.dot(np.transpose(S[:, :, n]), S[:, :, n]) + rho * I),
                                     np.dot(np.transpose(S[:, :, n]), data[:, n]) + rho * (U[:, n] - D[:, n]))

                U = proj_simplex(A + D)

                D = D + A - U

                if j > 0:
                    rel_A_ADMM = np.abs(
                        (np.linalg.norm(A, 'fro') - np.linalg.norm(A_old_ADMM, 'fro'))) / np.linalg.norm(
                        A_old_ADMM, 'fro')

                    print("iteration ", j, " of ", maxiter_ADMM, ", rel_A_ADMM =", rel_A_ADMM)

                    if rel_A_ADMM < tol_A_ADMM:
                        break

            # psi update

            for n in np.arange(N):
                for p in np.arange(P):
                    psi[p, n] = np.dot(np.transpose(S0[:, p]), S[:, p, n]) / np.dot(np.transpose(S0[:, p]), S0[:, p])

            # S update

            for n in np.arange(N):
                S[:, :, n] = np.dot(
                    np.outer(data[:, n], np.transpose(A[:, n])) + lambda_S * np.dot(S0, np.diag(psi[:, n])),
                    np.linalg.inv(np.outer(A[:, n], np.transpose(A[:, n])) + lambda_S * I))

            # termination checks

            if i > 0:

                S_vec = np.hstack(S)

                rel_A = np.abs(np.linalg.norm(A, 'fro') - np.linalg.norm(A_old, 'fro')) / np.linalg.norm(A_old, 'fro')
                rel_psi = np.abs(np.linalg.norm(psi, 'fro') - np.linalg.norm(psi_old, 'fro')) / np.linalg.norm(psi_old,
                                                                                                               'fro')
                rel_S = np.abs(np.linalg.norm(S_vec) - np.linalg.norm(np.hstack(S_old))) / np.linalg.norm(S_vec)

                print("iteration ", i, " of ", maxiter, ", rel_A =", rel_A, ", rel_psi =", rel_psi, "rel_S =", rel_S)

                if rel_A < tol_A and rel_psi and tol_psi and rel_S < tol_S and i > 1:
                    break

        return {
            "A": A,
            "E": S,
            "psi": psi
        }

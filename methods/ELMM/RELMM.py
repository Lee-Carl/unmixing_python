import numpy as np
from pymanopt.manifolds import Oblique
from pymanopt import Problem
from pymanopt.solvers import ConjugateGradient


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


class RELMM:

    def __init__(self, data, init) -> None:
        self.data = data
        self.init_E = init['E']
        # 参数
        self.maxiter_ADMM = 1000
        self.rho = 1
        self.tol_phi = 10 ** -5
        self.lambda_S = 0.5

    def SCLSU(self, y, S0):
        # 初始化参数
        maxiter_ADMM = 1000
        rho = 1
        tol_phi = 10 ** -5
        L = self.data['L']
        N = self.data['N']
        P = self.data['P']
        # 核心程序
        phi = np.ones([P, N])
        U = phi  # split variable
        D = np.zeros(phi.shape)  # Lagrange mutlipliers

        S0tX = np.dot(np.transpose(S0), y)
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
        return A

    def run(self, lam=None, savepath=None):

        data = self.data['Y'].copy()
        S0 = self.init_E

        A_init = self.SCLSU(data.copy(), S0.copy())

        lambda_S = 2
        lambda_S0 = 5

        P,L,N = self.data['P'],self.data['L'],self.data['N']

        V = P * np.eye(P) - np.outer(np.ones(P), np.transpose(np.ones(P)))

        def cost(X):

            data_fit = np.zeros(N)

            for n in np.arange(N):
                data_fit[n] = np.linalg.norm(S[:, :, n] - np.dot(X, np.diag(psi[:, n])), 'fro') ** 2

            cost = lambda_S / 2 * np.sum(data_fit, axis=0) + lambda_S0 / 2 * np.trace(
                np.dot(np.dot(X, V), np.transpose(X)))

            return cost

        def egrad(X):

            partial_grad = np.zeros([L, P, N])

            for n in np.arange(N):
                partial_grad[:, :, n] = np.dot(X, np.diag(psi[:, n])) - np.dot(S[:, :, n], np.diag(psi[:, n]))

            egrad = lambda_S * np.sum(partial_grad, axis=2) + lambda_S0 * np.dot(X, V)

            return egrad

        A = A_init
        S = np.zeros([L, P, N])
        psi = np.ones([P, N])

        for n in np.arange(N):
            S[:, :, n] = S0

        maxiter = 200

        U = A  # split variable
        D = np.zeros(A.shape)  # Lagrange mutlipliers

        rho = 1

        maxiter_ADMM = 100
        tol_A_ADMM = 10 ** -3
        tol_A = 10 ** -3
        tol_S = 10 ** -3
        tol_psi = 10 ** -3
        tol_S0 = 10 ** -3

        I = np.identity(P)

        for i in np.arange(maxiter):

            A_old = np.copy(A)
            psi_old = np.copy(psi)
            S_old = np.copy(S)
            S0_old = np.copy(S0)

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
                        (np.linalg.norm(A, 'fro') - np.linalg.norm(A_old_ADMM, 'fro'))) / np.linalg.norm(A_old_ADMM,
                                                                                                         'fro')

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

            # S0 update

            manifold = Oblique(L, P)
            solver = ConjugateGradient()
            problem = Problem(manifold=manifold, cost=cost, egrad=egrad)
            S0 = solver.solve(problem)

            # termination checks

            if i > 0:

                S_vec = np.hstack(S)

                rel_A = np.abs(np.linalg.norm(A, 'fro') - np.linalg.norm(A_old, 'fro')) / np.linalg.norm(A_old, 'fro')
                rel_psi = np.abs(np.linalg.norm(psi, 'fro') - np.linalg.norm(psi_old, 'fro')) / np.linalg.norm(psi_old,
                                                                                                               'fro')
                rel_S = np.abs(np.linalg.norm(S_vec) - np.linalg.norm(np.hstack(S_old))) / np.linalg.norm(S_old)
                rel_S0 = np.abs(np.linalg.norm(S0, 'fro') - np.linalg.norm(S0_old, 'fro')) / np.linalg.norm(S0_old,
                                                                                                            'fro')

                print("iteration ", i, " of ", maxiter, ", rel_A =", rel_A, ", rel_psi =", rel_psi, "rel_S =", rel_S,
                      "rel_S0 =", rel_S0)

                if rel_A < tol_A and rel_psi and tol_psi and rel_S < tol_S and rel_S0 < tol_S0 and i > 1:
                    break

        return {
            "A": A,
            "E": S,
            "psi": psi
        }

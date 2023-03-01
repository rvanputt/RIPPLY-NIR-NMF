# -*- coding: utf-8 -*-
"""
semi_nmf_v1.py

Semi-NMF based on multiplicative update rules (C. Ding, T. Li, and M.I. Jordan, "Convex and semi-nonnegative matrix factorizations", 
IEEE Transations on Pattern Analysis and Machine Intelligence, vol. 32, no. 1, pp. 45-55, 2010).

Code ported from Nicolas Sauwen's hNMF R package (https://cran.r-project.org/web/packages/hNMF/hNMF.pdf?X-OpenDNS-Session=_ed26518106144044050b564024e2e60a60549270f44b_9eYKEVim_)

Required input: 
    1.  X (data matrix, n*m. Example: 100 rows (n, wavenumbers) by 5000 columns (m, pre-processed spectra in time))
    2.  W0 (initial W-matrix, n*p. Example: 100 rows (n, wavenumbers) by 2 columns (p, components in NMF analysis)) 
    3.  H0 (initial H-matrix, p*m. Example: 2 rows (p, components in NMF analysis) by 5000 columns (m, pre-processed spectra in time))

Optional input:
    1.  maxiter (maximum number of iterations)
    2.  check_divergence (Boolean, checks for divergence of H matrix)

Output:
    1.  W (final W-matrix)
    2.  H (final H-matrix)

R. van Putten <rvanputt@its.jnj.com>
February 2023
"""

#%% Import modules
import numpy as np

#%% semi-NMF

def semi_nmf(X, W0, H0, maxiter=2000, check_divergence=False):

    W = W0
    H = H0
    W_old = W
    H_old = H

    epsilon = 1e-9
    has_diverged = False
    err = np.zeros(maxiter+1)
    err[1] = np.linalg.norm(X-np.matmul(W,H)) 
    rel_tol = 1e-6
    rel_error = np.inf
    iter = 1

    while(iter < maxiter and rel_error > rel_tol and has_diverged == False):

        H[np.sum(H, axis=1)==0,:] = epsilon
        W_new = np.matmul(X,np.linalg.pinv(H))

        # Artificial intervention to slow down potential W divergence
        for i in range(np.shape(W_new)[1]):
            W_new[:,i] = W_new[:,i]*np.amax(abs(W[:,i]))/np.amax(abs(W_new[:,i]))

        W_diff = (W - W_new)/(W + epsilon)
        W_diff[W==0] = 0
        W_diff_log = np.log(4*abs(W_diff)+1)/4*np.sign(W_diff)
        W = W*(1-W_diff_log)
        A = np.matmul(np.transpose(X),W)
        Ap = (abs(A)+A)/2
        An = (abs(A)-A)/2
        B = np.matmul(np.transpose(W),W)
        Bp = (abs(B)+B)/2
        Bn = (abs(B)-B)/2
        PP = np.transpose(An) + np.matmul(np.transpose(Bp),H)

        if len(np.where(PP == 0)) > 0:
            PP[PP==0] = np.min(PP[PP != 0])

        H = H*np.sqrt((np.transpose(Ap) + np.matmul(np.transpose(Bn),H))/PP)

        err[iter+1] = np.linalg.norm(X-np.matmul(W,H))
        rel_error = abs(err[iter+1]-err[iter])/(err[iter])

        # Prevent initialization being used as result
        if iter == 1 or iter == 11:
            W_old = W
            H_old = H

        if iter == 1 or iter % 10 == 0:
            # print('Iteration:',iter,'Relative error:',relError)

            if iter > 1 and check_divergence == True:
                has_diverged = divergence_check(W,W0)

                # If diverged, use result from 10 iterations ago
                if has_diverged == True:
                    W = W_old
                    H = H_old

        iter += 1

    return W, H

def divergence_check(W, W0):

    has_diverged = False

    # Normalize columns of W and W0:
    N0 = np.sqrt(np.diag(np.matmul(np.transpose(W0),W0)))
    N_W0 = np.tile(N0,(len(W0),1)) 
    W0 = W0/N_W0
    N = np.sqrt(np.diag(np.matmul(np.transpose(W),W)))
    N_W = np.tile(N,(len(W),1))
    W = W/N_W

    S_W0W0 = np.matmul(np.transpose(W0),W0)
    S_WW = np.matmul(np.transpose(W),W)
    S_WW0 = np.matmul(np.transpose(W),W0)

    # Divergence check 1: non-diagonal correlation values should not exceed treshold

    thr_check1 = 0.97
    non_diags0 = S_W0W0[np.triu_indices(S_W0W0.shape[0], k=1)]
    non_diags = S_WW[np.triu_indices(S_WW.shape[0], k=1)]
    check1_vect = np.full((len(non_diags)), thr_check1)
    check1_vect[non_diags0 > thr_check1] = non_diags0[non_diags0 > thr_check1] + (1-non_diags0[non_diags0 > thr_check1])/2
    check1 = np.where(non_diags > check1_vect)

    if len(check1[0]) > 0:
        has_diverged = True
        print('Triggered divergence check 1: non-diagonal correlation exceed treshold')
        return has_diverged

    # Divergence check 2: correlation values should (row-wise) be highest on diagonal
    
    max_per_row = np.argmax(S_WW0, axis=1)
    check2_vect = np.arange(1, np.shape(S_WW0)[0]+1)

    if max_per_row.all() != check2_vect.all():
        has_diverged = True
        print('Triggered divergence check 2: correlation not highest on diagonal')
        return has_diverged

    # Divergence check 3: H-matrix should not diverge too much from initialization

    thr_check3 = 0.90
    check3 = np.where(np.diag(S_WW0) < thr_check3)

    if len(check3[0]) > 0:
        has_diverged = True
        print('Triggered divergence check 3: solution diverged too far from initialization')
        return has_diverged
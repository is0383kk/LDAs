#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys,getopt
from numpy import array,matrix,diag
from scipy import sum,log,exp,mean,dot,ones,zeros
from scipy.special import polygamma
from scipy.linalg import norm
from random import random

def usage():
    print("usage: python lda.py [-Nclasses] [-Iemmax] [-Ddemmax] [-Eepsilon] train model")
    sys.exit(0)

def main():
    # Get params
    try:
        opts, args = getopt.getopt(sys.argv[1:], "N:I:D:E:h")
        train = sys.argv[len(opts) + 1]
        model = sys.argv[len(opts) + 2]
    except:
        usage()
        sys.exit(0)

    # Set params
    k = 10
    #k = 2
    #emmax = 100
    emmax = 100
    demmax = 20
    epsilon = 0.0001

    for o, a in opts:
        if o == '-N': k = int(a)
        if o == '-I': emmax = int(a)
        if o == '-D': demmax = int(a)
        if o == '-E': epsilon = float(a)
        if o == '-h': usage()

    # Train
    print("train=>{}".format(train))
    train = open(train,'r').read()
    alpha,beta = ldamain(train, k, emmax, demmax, epsilon)

    # Write
    print('writing data..')
    # alpha
    print('alpha -> %s.alpha' % model)
    writer = open(model + '.alpha','w')
    writer.write(str(alpha.tolist()))
    writer.close()
    # beta
    print('beta  -> %s.beta' % model)
    writer = open(model + '.beta','w')
    writer.write(str(beta.tolist()))
    writer.close()

def ldamain(train, k, emmax=100, demmax=20, epsilon=1.0e-4):
    '''
    Latent Dirichlet Allocation, standard model.
    [alpha,beta] = ldamain(train,k,[emmax,demmax])
    train  : group of data ASCII text.
             Each line represents a document and consists of pairs of <feature_id>:<count>.
             e.g.

             1:1 2:4 5:2
             1:2 3:3 5:1 6:1 7:1
             2:4 5:1 7:1

    k      : # of classes to assume
    emmax  : # of maximum VB-EM iteration (default 100)
    demmax : # of maximum VB-EM iteration for a document (default 20)
    epsilon: A threshold to determine the whole convergence of the estimation (default 0.0001)
    '''
    """
    d->[[(10, 37, 40, 43, 62, 72, 75), (7, 1, 1, 1, 3, 1, 1)], [(10, 37, 40, 43, 62, 72, 75), (7, 1, 1, 1, 3, 1, 1)]]
    """
    """
    d = []
    for L in train.split('\n'):
        print("L->{}".format(L))
        for w in L.split():
            print("w->{}".format(w))
            for x in w.split(':'):
                print("x->{}".format(int(x)))
                d.append(int(x))
    """
    d = [ zip(*[ [int(x) for x in w.split(':')] for w in L.split()]) for L in train.split('\n') if L ]
    print("d->{}".format(d))

    return lda.train(d,k,emmax,demmax)


class lda():
    '''
    Latent Dirichlet Allocation, standard model.
    [alpha,beta] = lda.train(d,k,[emmax,demmax])
    d      : data of documents
    k      : # of classes to assume
    emmax  : # of maximum VB-EM iteration (default 100)
    demmax : # of maximum VB-EM iteration for a document (default 20)
    '''

    @staticmethod
    def train(d, k, emmax=100, demmax=20, epsilon=1.0e-4):
        '''
        Latent Dirichlet Allocation, standard model.
        [alpha,beta] = lda.train(d,k,[emmax,demmax])
        d      : data of documents
        k      : # of classes to assume
        emmax  : # of maximum VB-EM iteration (default 100)
        demmax : # of maximum VB-EM iteration for a document (default 20)
        '''

        # # of documents
        n = len(d)
        # # of words
        L = max(map(lambda x: max(x[0]), d)) + 1

        # initialize
        beta = ones((L, k)) / L
        alpha = lda.normalize(sorted([random() for i in range(k)], reverse=True))
        gammas = zeros((n, k))
        lik = 0
        plik = lik

        print('number of documents (n)      = %d' % n)
        print('number of words (l)          = %d' % L)
        print('number of latent classes (k) = %d' % k)

        for j in range(emmax):
            print('iteration %d/%d..\t' % (j+1, emmax))
            #vb-esstep
            betas = zeros((L, k))
            for i in range(n):
                gamma,q = lda.vbem(d[i], beta, alpha, demmax)
                gammas[i,:] = gamma
                betas = lda.accum_beta(betas,q,d[i])
            #vb-mstep
            alpha = lda.newton_alpha(gammas)
            beta = lda.mnormalize(betas,0)
            #converge?
            lik = lda.lda_lik(d, beta, gammas)
            print('log-likelihoood =', lik)
            if j > 1 and abs((lik - plik) / lik) < epsilon:
                if j < 5:
                    #print
                    return lda.train(d, k, emmax, demmax) # try again
                    return
                print('converged')
                return alpha, beta
            plik = lik


    @staticmethod
    def vbem(d, beta, alpha0, emmax=20):
        '''
        [alpha,q] = vbem(d,beta,alpha0,[emmax])
        calculates a document and words posterior for a document d.
        alpha  : Dirichlet posterior for a document d
        q      : (L * K) matrix of word posterior over latent classes
        d      : document data
        beta   :
        alpha0 : Dirichlet prior of alpha
        emmax  : maximum # of VB-EM iteration.
        '''
        digamma = lambda x: polygamma(0,x)

        L = len(d[0])
        k = len(alpha0)
        q = zeros((L, k))
        nt = ones((1, k)) * L / k
        pnt = nt

        for j in range(emmax):
            #vb-estep
            q = lda.mnormalize(matrix(beta[d[0],:]) * matrix(diag(exp(digamma(alpha0 + nt))[0])), 1)
            #vb-mstep
            nt = matrix(d[1]) * q
            #converge?
            if j > 1 and lda.converged(nt, pnt, 1.0e-2):
                break
            pnt = nt.copy()
        alpha = alpha0 + nt
        return alpha, q

    @staticmethod
    def accum_beta(betas, q, t):
        '''
        betas = accum_beta(betas,q,t)
        accumulates word posteriors to latent classes.
        betas : (V * K) matrix of summand
        q     : (L * K) matrix of word posteriors
        t     : document of struct array
        '''
        betas[t[0],:] += matrix(diag(t[1])) * q
        return betas

    @staticmethod
    def lda_lik(d, beta, gammas):
        '''
        lik = lda_lik(d, beta, gammas)
        returns the likelihood of d, given LDA model of (beta, gammas).
        '''
        egamma = matrix(lda.mnormalize(gammas, 1))
        lik = 0
        n = len(d)
        for i in range(n):
            t = d[i]
            lik += (matrix(t[1]) * log(matrix(beta[t[0],:]) * egamma[i,:].T))[0,0]
        return lik

    @staticmethod
    def newton_alpha(gammas,maxiter=20,ini_alpha=[]):
        '''
        alpha = newton_alpha (gammas,[maxiter])
        Newton-Raphson iteration of LDA Dirichlet prior.
        gammas  : matrix of Dirichlet posteriors (M * k)
        maxiter : # of maximum iteration of Newton-Raphson
        '''
        digamma = lambda x: polygamma(0, x)
        trigamma = lambda x: polygamma(1, x)

        M,K = gammas.shape

        if not M > 1:
            return gammas[1,:]
        if not len(ini_alpha) > 0:
            ini_alpha = mean(gammas, 0) / K

        L = 0
        g = zeros((1,K))
        pg = sum(digamma(gammas), 0) - sum(digamma(sum(gammas, 1)))
        alpha = ini_alpha.copy()
        palpha = zeros((1, K))

        for t in range(maxiter):
            L += 1
            alpha0 = sum(alpha)
            g = M * (digamma(alpha0) - digamma(alpha)) + pg
            h = -1.0 / trigamma(alpha)
            hgz = dot(h,g) / (1.0 / trigamma(alpha0) + sum(h))

            for i in range(K):
                alpha[i] = alpha[i] - h[i] * (g[i] - hgz) / M
                if alpha[i] < 0:
                    return lda.newton_alpha(gammas, maxiter, ini_alpha / 10.0)

            #converge?
            if L > 1 and lda.converged(alpha, palpha, 1.0e-4):
                break

            palpha = alpha.copy()

        return alpha

    @staticmethod
    def normalize(v):
        return v / sum(v)

    @staticmethod
    def mnormalize(m, d=0):
        '''
        x = mnormalize(m, d)
        normalizes a 2-D matrix m along the dimension d.
        m : matrix
        d : dimension to normalize (default 0)
        '''
        m = array(m)
        v = sum(m, d)
        if d == 0:
            return m * matrix(diag(1.0 / v))
        else:
            return matrix(diag(1.0 / v)) * m

    @staticmethod
    def converged(u, udash, threshold=1.0e-3):
        '''
        converged(u,udash,threshold)
        Returns 1 if u and udash are not different by the ratio threshold
        '''
        return norm(u - udash) / norm(u) < threshold


if __name__ == '__main__':
    main()

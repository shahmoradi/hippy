'''
Created on Aug 18, 2017

@author: uvilla
'''
import dolfin as dl
import math
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append( "../hiipylib" )
from hippylib import *


def computeGammaDelta(corr_len, marginal, alpha, ndim):
    """
    A covariance operator C from a Matern distribution can be written as:
    C = (\delta I - \gamma \Delta)^{-\alpha}
    
    This function computes the parameters gamma and delta of the PDE representation, given:
    - corr_len: the desidered correlation lenght of the field.
    - marginal: the desidered marginal variance of the field.
    - alpha: 1 -> Laplacian Prior (1D only), 2-> Bilaplacian Prior (2D and 3D).
    - ndim: the number of space dimensions
    """
    nu = alpha - 0.5*float(ndim)
    assert alpha > 0., "Alpha must be larger than ndim/2"
    kappa = math.sqrt(8.*nu)/corr_len
    gamma = math.sqrt(math.gamma(nu)/math.gamma(alpha))/(math.pow( 4.0*math.pi, 0.25*float(ndim) )*math.pow(kappa, nu))
    delta = kappa*kappa*gamma
    
    return gamma, delta
    

if __name__ == "__main__":
    dl.set_log_active(False)
    # 1. set up the mesh and finite element space
    ndim = 2
    nx = 256
    ny = 256
    h = 1./float(nx)
    mesh = dl.UnitSquareMesh(nx, ny)
        
    Vh = dl.FunctionSpace(mesh, "CG", 1)
    
    # 2. set up the prior distribution for m ~ N(0, C_pr)
    corr_len = 0.02  # Correlation lenght
    marginal = 1.0   # Marginal (pointwise) variance of the field
    alpha = 2.0      # Use Bilaplacian prior
    
    gamma, delta = computeGammaDelta(corr_len, marginal, alpha, ndim)
    print "Gamma, Delta = ", gamma, delta
    prior = BiLaplacianPrior(Vh, gamma, delta)
    
    # 3. generate realization of the medium with inclusion.
    # For each realization of the GRF m, the inclusions are defined by the tau-isocontour of m.
    # tau controls the density of the inclusions (the bigger tau, the fewer inclusions)
    # the correlation length of m controls the diameter of the inclusion
    tau = 1.5
    
    fid_prior = dl.File("samples/sample_prior.pvd")
    nsamples = 50
    noise = dl.Vector()
    s     = dl.Vector()
    prior.init_vector(noise,"noise")
    prior.init_vector(s,0)
    s_prior = dl.Function(Vh, name="sample_prior")
    for i in range(nsamples):
        parRandom.normal(1., noise)
        prior.sample(noise, s)
        s.set_local(np.tanh((s.array()- tau)/h)) #comment out this line if you want to see realization of m instead
        s_prior.vector().zero()
        s_prior.vector().axpy(1., s)
        fid_prior << s_prior

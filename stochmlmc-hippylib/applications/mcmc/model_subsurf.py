# Copyright (c) 2016, The University of Texas at Austin & University of
# California, Merced.
#
# All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the hIPPYlib library. For more information and source code
# availability see https://hippylib.github.io.
#
# hIPPYlib is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License (as published by the Free
# Software Foundation) version 2.0 dated June 1991.

import dolfin as dl
import math
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append( "../../" )
from hippylib import *

class FluxQOI(object):
    def __init__(self, Vh, dsGamma):
        self.Vh = Vh
        self.dsGamma = dsGamma
        self.n = dl.Constant((0.,1.))#dl.FacetNormal(Vh[STATE].mesh())
        
        self.u = None
        self.m = None
        self.L = {}
        
    def form(self, x):
        #return dl.avg(dl.exp(x[PARAMETER])*dl.dot( dl.grad(x[STATE]), self.n) )*self.dsGamma
        return dl.exp(x[PARAMETER])*dl.dot( dl.grad(x[STATE]), self.n)*self.dsGamma
    
    def eval(self, x):
        """
        Given x evaluate the cost functional.
        Only the state u and (possibly) the parameter a are accessed.
        """
        u = vector2Function(x[STATE], self.Vh[STATE])
        m = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        return dl.assemble(self.form([u,m]))

class GammaBottom(dl.SubDomain):
    def inside(self, x, on_boundary):
        return ( abs(x[1]) < dl.DOLFIN_EPS )
       
def u_boundary(x, on_boundary):
    return on_boundary and ( x[1] < dl.DOLFIN_EPS or x[1] > 1.0 - dl.DOLFIN_EPS)

def v_boundary(x, on_boundary):
    return on_boundary and ( x[0] < dl.DOLFIN_EPS or x[0] > 1.0 - dl.DOLFIN_EPS)

def true_model(Vh, gamma, delta, anis_diff):
    prior = BiLaplacianPrior(Vh, gamma, delta, anis_diff )
    noise = dl.Vector()
    prior.init_vector(noise,"noise")
    parRandom.normal(1., noise)
    atrue = dl.Vector()
    prior.init_vector(atrue, 0)
    prior.sample(noise,atrue)
    return atrue
            
if __name__ == "__main__":
    dl.set_log_active(False)
    sep = "\n"+"#"*80+"\n"
    ndim = 2
    nx = 64  
    ny = 64
    mesh = dl.UnitSquareMesh(nx, ny)
    
    rank = dl.MPI.rank(mesh.mpi_comm())
    nproc = dl.MPI.size(mesh.mpi_comm())
        
    Vh2 = dl.FunctionSpace(mesh, 'Lagrange', 2)
    Vh1 = dl.FunctionSpace(mesh, 'Lagrange', 1)
    Vh = [Vh2, Vh1, Vh2]
    
    ndofs = [Vh[STATE].dim(), Vh[PARAMETER].dim(), Vh[ADJOINT].dim()]
    if rank == 0:
        print sep, "Set up the mesh and finite element spaces", sep
        print "Number of dofs: STATE={0}, PARAMETER={1}, ADJOINT={2}".format(*ndofs)
        
    # Initialize Expressions
    f = dl.Constant(0.0)
        
    u_bdr = dl.Expression("x[1]", degree = 1)
    u_bdr0 = dl.Constant(0.0)
    bc = dl.DirichletBC(Vh[STATE], u_bdr, u_boundary)
    bc0 = dl.DirichletBC(Vh[STATE], u_bdr0, u_boundary)
    
    def pde_varf(u,a,p):
        return dl.exp(a)*dl.inner(dl.nabla_grad(u), dl.nabla_grad(p))*dl.dx - f*p*dl.dx
    
    pde = PDEVariationalProblem(Vh, pde_varf, bc, bc0, is_fwd_linear=True)
    if dlversion() <= (1,6,0):
        pde.solver = dl.PETScKrylovSolver("cg", amg_method())
        pde.solver_fwd_inc = dl.PETScKrylovSolver("cg", amg_method())
        pde.solver_adj_inc = dl.PETScKrylovSolver("cg", amg_method())
    else:
        pde.solver = dl.PETScKrylovSolver(mesh.mpi_comm(), "cg", amg_method())
        pde.solver_fwd_inc = dl.PETScKrylovSolver(mesh.mpi_comm(), "cg", amg_method())
        pde.solver_adj_inc = dl.PETScKrylovSolver(mesh.mpi_comm(), "cg", amg_method())
    pde.solver.parameters["relative_tolerance"] = 1e-15
    pde.solver.parameters["absolute_tolerance"] = 1e-20
    pde.solver_fwd_inc.parameters = pde.solver.parameters
    pde.solver_adj_inc.parameters = pde.solver.parameters
 
    ntargets = 300
    np.random.seed(seed=1)
    targets = np.random.uniform(0.1,0.9, [ntargets, ndim] )
    if rank == 0:
        print "Number of observation points: {0}".format(ntargets)
    misfit = PointwiseStateObservation(Vh[STATE], targets)
    
    
    gamma = .1
    delta = .5
    
    anis_diff = dl.Expression(code_AnisTensor2D, degree=1)
    anis_diff.theta0 = 2.
    anis_diff.theta1 = .5
    anis_diff.alpha = math.pi/4
    atrue = true_model(Vh[PARAMETER], gamma, delta,anis_diff)
        
    locations = np.array([[0.1, 0.1], [0.1, 0.9], [.5,.5], [.9, .1], [.9, .9]])

    pen = 1e1
    prior = MollifiedBiLaplacianPrior(Vh[PARAMETER], gamma, delta, locations, atrue, anis_diff, pen)
    
    if rank == 0:    
        print "Prior regularization: (delta_x - gamma*Laplacian)^order: delta={0}, gamma={1}, order={2}".format(delta, gamma,2)    
                
    #Generate synthetic observations
    utrue = pde.generate_state()
    x = [utrue, atrue, None]
    pde.solveFwd(x[STATE], x, 1e-9)
    misfit.B.mult(x[STATE], misfit.d)
    rel_noise = 0.01
    MAX = misfit.d.norm("linf")
    noise_std_dev = rel_noise * MAX
    parRandom.normal_perturb(noise_std_dev, misfit.d)
    misfit.noise_variance = noise_std_dev*noise_std_dev
    
    model = Model(pde,prior, misfit)
    
    if rank == 0:       
        print sep, "Find the MAP point", sep
    a = prior.mean.copy()
    parameters = ReducedSpaceNewtonCG_ParameterList()
    parameters["rel_tolerance"] = 1e-8
    parameters["abs_tolerance"] = 1e-12
    parameters["max_iter"]      = 25
    parameters["inner_rel_tolerance"] = 1e-15
    parameters["globalization"] = "LS"
    parameters["GN_iter"] = 5
    if rank != 0:
        parameters["print_level"] = -1
    solver = ReducedSpaceNewtonCG(model, parameters)
    
    x = solver.solve([None, a, None])
    
    if rank == 0:
        if solver.converged:
            print "\nConverged in ", solver.it, " iterations."
        else:
            print "\nNot Converged"

        print "Termination reason: ", solver.termination_reasons[solver.reason]
        print "Final gradient norm: ", solver.final_grad_norm
        print "Final cost: ", solver.final_cost
    
    ## Build the low rank approximation of the posterior
    model.setPointForHessianEvaluations(x, gauss_newton_approx=True)
    Hmisfit = ReducedHessian(model, solver.parameters["inner_rel_tolerance"], misfit_only=True)
    k = 50
    p = 20
    if rank == 0:
        print "Double Pass Algorithm. Requested eigenvectors: {0}; Oversampling {1}.".format(k,p)
    
    Omega = MultiVector(x[PARAMETER], k+p)
    parRandom.normal(1., Omega)
        
    d, U = doublePassG(Hmisfit, prior.R, prior.Rsolver, Omega, k, s=2, check = False)
    d[d < 0.] = 0.
    nu = GaussianLRPosterior(prior, d, U)
    nu.mean = x[PARAMETER]

    ## Define the QOI
    GC = GammaBottom()
    marker = dl.FacetFunction("size_t", mesh)
    marker.set_all(0)
    GC.mark(marker, 1)
    dss = dl.Measure("ds", subdomain_data=marker)
    qoi = FluxQOI(Vh,dss(1))
    
    kSN = SNmapKernel(model, nu)
    
    kMALA = MALAKernel(model)
    kMALA.parameters["delta_t"] = 2e-4
    
    kpCN = pCNKernel(model)
    kpCN.parameters["s"] = 0.025
    
    kgpCN = gpCNKernel(model,nu)
    kgpCN.parameters["s"] = 0.25
    
    kIS = ISKernel(model,nu)
    
    noise = dl.Vector()
    nu.init_vector(noise, "noise")
    parRandom.normal(1., noise)
    pr_s = model.generate_vector(PARAMETER)
    post_s = model.generate_vector(PARAMETER)
    
    nu.sample(noise, pr_s, post_s, add_mean=True)
        
    for kernel in [kSN, kMALA, kpCN, kgpCN, kIS]:
        
        if rank == 0:
            print kernel.name()
        
        fid_m = dl.File(kernel.name()+"/parameter.pvd")
        fid_u = dl.File(kernel.name()+"/state.pvd")
        chain = MCMC(kernel)
        chain.parameters["burn_in"] = 0
        chain.parameters["number_of_samples"] = 10
        chain.parameters["print_progress"] = 10            
        tracer = FullTracer(chain.parameters["number_of_samples"], Vh, fid_m, fid_u)
        if rank != 0:
            chain.parameters["print_level"] = -1
        
        n_accept = chain.run(post_s, qoi, tracer)
        if rank == 0:
            np.savetxt(kernel.name()+".txt", tracer.data)
            print "Number accepted = {0}".format(n_accept)
            print "E[q] = {0}".format(chain.sum_q/float(chain.parameters["number_of_samples"]))
        
            plt.figure()
            plt.plot(tracer.data[:,0], '*b')
    
    if rank == 0:
        plt.show()
        
    

'''
Created on Jun 21, 2017

@author: uvilla
'''
import dolfin as dl
import math
import numpy as np

import sys
sys.path.append( "../../" )
from hippylib import *


def u_boundary(x, on_boundary):
    return on_boundary and ( x[1] < dl.DOLFIN_EPS or x[1] > 1.0 - dl.DOLFIN_EPS)

def v_boundary(x, on_boundary):
    return on_boundary and ( x[0] < dl.DOLFIN_EPS or x[0] > 1.0 - dl.DOLFIN_EPS)
            
if __name__ == "__main__":
    dl.set_log_active(False)
    
    assert dlversion() >= (2016,2,0)
    
    world_comm = dl.mpi_comm_world()
    self_comm  = dl.mpi_comm_self()
    
    ndim = 2
    nx = 16
    ny = 16
    mesh = dl.UnitSquareMesh(self_comm, nx, ny)
    
    rank = dl.MPI.rank(world_comm)
    nproc = dl.MPI.size(world_comm)
            
    Vh2 = dl.FunctionSpace(mesh, 'Lagrange', 2)
    Vh1 = dl.FunctionSpace(mesh, 'Lagrange', 1)
    Vh = [Vh2, Vh1, Vh2]
    
    ndofs = [Vh[STATE].dim(), Vh[PARAMETER].dim(), Vh[ADJOINT].dim()]
    if rank == 0:
        print "Set up the mesh and finite element spaces"
        print "Number of dofs: STATE={0}, PARAMETER={1}, ADJOINT={2}".format(*ndofs)
    
    # Initialize Expressions
    f = dl.Constant(0.0)
        
    u_bdr = dl.Expression("x[1]", element = Vh[STATE].ufl_element() )
    u_bdr0 = dl.Constant(0.0)
    bc = dl.DirichletBC(Vh[STATE], u_bdr, u_boundary)
    bc0 = dl.DirichletBC(Vh[STATE], u_bdr0, u_boundary)
    
    def pde_varf(u,a,p):
        return dl.exp(a)*dl.inner(dl.nabla_grad(u), dl.nabla_grad(p))*dl.dx - f*p*dl.dx
    
    pde = PDEVariationalProblem(Vh, pde_varf, bc, bc0, is_fwd_linear=True)
    pde.solver = dl.PETScKrylovSolver(self_comm, "cg", amg_method())
    pde.solver.parameters["relative_tolerance"] = 1e-15
    pde.solver.parameters["absolute_tolerance"] = 1e-20
    pde.solver_fwd_inc = dl.PETScKrylovSolver(self_comm, "cg", amg_method())
    pde.solver_fwd_inc.parameters = pde.solver.parameters
    pde.solver_adj_inc = dl.PETScKrylovSolver(self_comm, "cg", amg_method())
    pde.solver_adj_inc.parameters = pde.solver.parameters
 
    ntargets = 300
    np.random.seed(seed=1)
    targets = np.random.uniform(0.1,0.9, [ntargets, ndim] )
    if rank == 0:
        print "Number of observation points: {0}".format(ntargets)
    misfit = PointwiseStateObservation(Vh[STATE], targets)
    misfit.noise_variance = 1e-4
    
    gamma = .1
    delta = .5
    
    anis_diff = dl.Expression(code_AnisTensor2D, degree = 1)
    anis_diff.theta0 = 2.
    anis_diff.theta1 = .5
    anis_diff.alpha = math.pi/4
    
    prior = BiLaplacianPrior(Vh[PARAMETER], gamma, delta, anis_diff)
    model = Model(pde,prior, misfit)
    
    ns = 2
    m_MC = model.generate_vector(PARAMETER)
    u_MC = model.generate_vector(STATE)
    noise_m = dl.Vector()
    prior.init_vector(noise_m, "noise")
    noise_obs = dl.Vector()
    misfit.B.init_vector(noise_obs, 0)
    
    data = np.zeros( (ns,5) )
    
    k = 50
    p = 20
    
    Omega = MultiVector(m_MC, k+p)
    parRandom.normal(1., Omega)
    
    for iMC in np.arange(ns):
        print "Rank: ", rank, "Sample: ", iMC
        parRandom.normal(1., noise_m)
        parRandom.normal(1., noise_obs)
        prior.sample(noise_m, m_MC)
        pde.solveFwd(u_MC, [u_MC, m_MC, None], 1e-9)
        misfit.B.mult(u_MC, misfit.d)
        misfit.d.axpy(np.sqrt(misfit.noise_variance), noise_obs)
    
        a = m_MC.copy()
        parameters = ReducedSpaceNewtonCG_ParameterList()
        parameters["rel_tolerance"] = 1e-9
        parameters["abs_tolerance"] = 1e-12
        parameters["max_iter"]      = 25
        parameters["inner_rel_tolerance"] = 1e-15
        parameters["globalization"] = "LS"
        parameters["GN_iter"] = 5
        parameters["print_level"] = -1
        
        solver = ReducedSpaceNewtonCG(model, parameters)
    
        x = solver.solve([None, a, None])
                
        model.setPointForHessianEvaluations(x, gauss_newton_approx=False)
        Hmisfit = ReducedHessian(model, solver.parameters["inner_rel_tolerance"], misfit_only=True)
        d, U = doublePassG(Hmisfit, prior.R, prior.Rsolver, Omega, k, s=1, check=False)
        posterior = GaussianLRPosterior(prior, d, U)
        posterior.mean = x[PARAMETER]
    
        kl_dist, c_logdet, c_tr, cr = posterior.klDistanceFromPrior(sub_comp=True)
        
        cm = model.misfit.cost(x)
        data[iMC, 0] = kl_dist
        data[iMC, 1] = cm
        data[iMC, 2] = cr
        data[iMC, 3] = c_logdet
        data[iMC, 4] = c_tr
                
    
    np.savetxt('kldist_p{0}.txt'.format(rank), data, header='kldist, c_misfit, c_reg, c_logdet, C_tr', comments='% ')
        

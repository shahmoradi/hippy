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
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append( "../../" )
from hippylib import *


def u_boundary(x, on_boundary):
    return on_boundary

class Poisson:
    def __init__(self, mesh, Vh, atrue, targets, prior, rel_noise_level):
        """
        Construct a model by proving
        - the mesh
        - the finite element spaces for the STATE/ADJOINT variable and the PARAMETER variable
        - the Prior information
        """
        self.mesh = mesh
        self.Vh = Vh
        
        # Initialize Expressions
        self.atrue = atrue
        self.f = dl.Constant(1.0)
        self.u_o = dl.Vector()
        
        self.u_bdr = dl.Constant(0.0)
        self.u_bdr0 = dl.Constant(0.0)
        self.bc = dl.DirichletBC(self.Vh[STATE], self.u_bdr, u_boundary)
        self.bc0 = dl.DirichletBC(self.Vh[STATE], self.u_bdr0, u_boundary)
                
        # Assemble constant matrices      
        self.prior = prior
        self.B = assemblePointwiseObservation(Vh[STATE],targets)
                
        self.noise_variance = self.computeObservation(self.u_o, rel_noise_level)
        rank = dl.MPI.rank(mesh.mpi_comm())
        if rank == 0:
            print "Noise variance:", self.noise_variance
                
        self.A = None
        self.At = None
        self.C = None
        self.Raa = None
        self.Wau = None
        
        self.gauss_newton_approx=False
        
    def generate_vector(self, component="ALL"):
        """
        Return the list x=[u,a,p] where:
        - u is any object that describes the state variable
        - a is a Vector object that describes the parameter.
          (Need to support linear algebra operations)
        - p is any object that describes the adjoint variable
        
        If component is STATE, PARAMETER, or ADJOINT return x[component]
        """
        if component == "ALL":
            x = [dl.Vector(), dl.Vector(), dl.Vector()]
            self.B.init_vector(x[STATE],1)
            self.prior.init_vector(x[PARAMETER],0)
            self.B.init_vector(x[ADJOINT], 1)
        elif component == STATE:
            x = dl.Vector()
            self.B.init_vector(x,1)
        elif component == PARAMETER:
            x = dl.Vector()
            self.prior.init_vector(x,0)
        elif component == ADJOINT:
            x = dl.Vector()
            self.B.init_vector(x,1)
            
        return x
    
    def init_parameter(self, a):
        """
        Reshape a so that it is compatible with the parameter variable
        """
        self.prior.init_vector(a,0)
        
    def assembleA(self,x, assemble_adjoint = False, assemble_rhs = False):
        """
        Assemble the matrices and rhs for the forward/adjoint problems
        """
        trial = dl.TrialFunction(self.Vh[STATE])
        test = dl.TestFunction(self.Vh[STATE])
        c = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        Avarf = dl.inner(dl.exp(c)*dl.nabla_grad(trial), dl.nabla_grad(test))*dl.dx
        if not assemble_adjoint:
            bform = dl.inner(self.f, test)*dl.dx
            Matrix, rhs = dl.assemble_system(Avarf, bform, self.bc)
        else:
            # Assemble the adjoint of A (i.e. the transpose of A)
            s = vector2Function(x[STATE], self.Vh[STATE])
            bform = dl.inner(dl.Constant(0.), test)*dl.dx
            Matrix, _ = dl.assemble_system(dl.adjoint(Avarf), bform, self.bc0)
            Bu = -(self.B*x[STATE])
            Bu += self.u_o
            rhs = dl.Vector()
            self.B.init_vector(rhs, 1)
            self.B.transpmult(Bu,rhs)
            rhs *= 1.0/self.noise_variance
            
        if assemble_rhs:
            return Matrix, rhs
        else:
            return Matrix
    
    def assembleC(self, x):
        """
        Assemble the derivative of the forward problem with respect to the parameter
        """
        trial = dl.TrialFunction(self.Vh[PARAMETER])
        test = dl.TestFunction(self.Vh[STATE])
        s = vector2Function(x[STATE], Vh[STATE])
        c = vector2Function(x[PARAMETER], Vh[PARAMETER])
        Cvarf = dl.inner(dl.exp(c) * trial * dl.nabla_grad(s), dl.nabla_grad(test)) * dl.dx
        C = dl.assemble(Cvarf)
#        print "||c||", x[PARAMETER].norm("l2"), "||s||", x[STATE].norm("l2"), "||C||", C.norm("linf")
        self.bc0.zero(C)
        return C
                
    def assembleWau(self, x):
        """
        Assemble the derivative of the parameter equation with respect to the state
        """
        trial = dl.TrialFunction(self.Vh[STATE])
        test  = dl.TestFunction(self.Vh[PARAMETER])
        a = vector2Function(x[ADJOINT], Vh[ADJOINT])
        c = vector2Function(x[PARAMETER], Vh[PARAMETER])
        varf = dl.inner(dl.exp(c)*dl.nabla_grad(trial),dl.nabla_grad(a))*test*dl.dx
        Wau = dl.assemble(varf)
        Wau_t = Transpose(Wau)
        self.bc0.zero(Wau_t)
        Wau = Transpose(Wau_t)
        return Wau
    
    def assembleRaa(self, x):
        """
        Assemble the derivative of the parameter equation with respect to the parameter (Newton method)
        """
        trial = dl.TrialFunction(self.Vh[PARAMETER])
        test  = dl.TestFunction(self.Vh[PARAMETER])
        s = vector2Function(x[STATE], Vh[STATE])
        c = vector2Function(x[PARAMETER], Vh[PARAMETER])
        a = vector2Function(x[ADJOINT], Vh[ADJOINT])
        varf = dl.inner(dl.nabla_grad(a),dl.exp(c)*dl.nabla_grad(s))*trial*test*dl.dx
        return dl.assemble(varf)

        
    def computeObservation(self, u_o, rel_noise_level):
        """
        Compute the syntetic observation
        """
        x = [self.generate_vector(STATE), self.atrue, None]
        self.solveFwd(x[STATE], x, tol=1e-9)
        
        # Create noisy data, ud
        MAX = x[STATE].norm("linf")
        std_dev = rel_noise_level * MAX
        parRandom.normal_perturb(std_dev, x[STATE])
        
        self.B.init_vector(u_o,0)
        self.B.mult(x[STATE], u_o)
        
        return std_dev*std_dev
        
    
    def cost(self, x):
        """
        Given the list x = [u,a,p] which describes the state, parameter, and
        adjoint variable compute the cost functional as the sum of 
        the misfit functional and the regularization functional.
        
        Return the list [cost functional, regularization functional, misfit functional]
        
        Note: p is not needed to compute the cost functional
        """        
        assert x[STATE] != None
                       
        diff = self.B*x[STATE]
        diff -= self.u_o
        misfit = (.5/self.noise_variance) * diff.inner(diff)
        
        Rdiff_x = dl.Vector()
        self.prior.init_vector(Rdiff_x,0)
        diff_x = x[PARAMETER] - self.prior.mean
        self.prior.R.mult(diff_x, Rdiff_x)
        reg = .5 * diff_x.inner(Rdiff_x)
        
        c = misfit + reg
        
        return c, reg, misfit
    
    def solveFwd(self, out, x, tol=1e-9):
        """
        Solve the forward problem.
        """
        A, b = self.assembleA(x, assemble_rhs = True)
        A.init_vector(out, 1)
        if dlversion() <= (1,6,0):
            solver = dl.PETScKrylovSolver("cg", amg_method())
        else:
            solver = dl.PETScKrylovSolver(self.mesh.mpi_comm(), "cg", amg_method())
        solver.parameters["relative_tolerance"] = tol
        solver.set_operator(A)
        nit = solver.solve(out,b)
        
#        print "FWD", (self.A*out - b).norm("l2")/b.norm("l2"), nit

    
    def solveAdj(self, out, x, tol=1e-9):
        """
        Solve the adjoint problem.
        """
        At, badj = self.assembleA(x, assemble_adjoint = True,assemble_rhs = True)
        At.init_vector(out, 1)
        
        if dlversion() <= (1,6,0):
            solver = dl.PETScKrylovSolver("cg", amg_method())
        else:
            solver = dl.PETScKrylovSolver(self.mesh.mpi_comm(), "cg", amg_method())
        solver.parameters["relative_tolerance"] = tol
        solver.set_operator(At)
        nit = solver.solve(out,badj)
        
#        print "ADJ", (self.At*out - badj).norm("l2")/badj.norm("l2"), nit
    
    def evalGradientParameter(self,x, mg, misfit_only = False):
        """
        Evaluate the gradient for the variation parameter equation at the point x=[u,a,p].
        Parameters:
        - x = [u,a,p] the point at which to evaluate the gradient.
        - mg the variational gradient (g, atest) being atest a test function in the parameter space
          (Output parameter)
        
        Returns the norm of the gradient in the correct inner product g_norm = sqrt(g,g)
        """ 
        C = self.assembleC(x)

        self.prior.init_vector(mg,0)
        C.transpmult(x[ADJOINT], mg)
        
        if misfit_only == False:
            Rdx = dl.Vector()
            self.prior.init_vector(Rdx,0)
            dx = x[PARAMETER] - self.prior.mean
            self.prior.R.mult(dx, Rdx)   
            mg.axpy(1., Rdx)
        
        g = dl.Vector()
        self.prior.init_vector(g,1)
        
        self.prior.Msolver.solve(g, mg)
        g_norm = dl.sqrt( g.inner(mg) )
        
        return g_norm
        
    
    def setPointForHessianEvaluations(self, x, gauss_newton_approx=False):  
        """
        Specify the point x = [u,a,p] at which the Hessian operator (or the Gauss-Newton approximation)
        need to be evaluated.
        """      

        self.gauss_newton_approx = gauss_newton_approx

        self.A  = self.assembleA(x)
        self.At = self.assembleA(x, assemble_adjoint=True )
        self.C  = self.assembleC(x)
        if gauss_newton_approx:
            self.Wau = None
            self.Raa = None
        else:
            self.Wau = self.assembleWau(x)
            self.Raa = self.assembleRaa(x)

        
    def solveFwdIncremental(self, sol, rhs, tol):
        """
        Solve the incremental forward problem for a given rhs
        """
        if dlversion() <= (1,6,0):
            solver = dl.PETScKrylovSolver("cg", amg_method())
        else:
            solver = dl.PETScKrylovSolver(self.mesh.mpi_comm(), "cg", amg_method())
        solver.set_operator(self.A)
        solver.parameters["relative_tolerance"] = tol
        self.A.init_vector(sol,1)
        nit = solver.solve(sol,rhs)
#        print "FwdInc", (self.A*sol-rhs).norm("l2")/rhs.norm("l2"), nit
        
    def solveAdjIncremental(self, sol, rhs, tol):
        """
        Solve the incremental adjoint problem for a given rhs
        """
        if dlversion() <= (1,6,0):
            solver = dl.PETScKrylovSolver("cg", amg_method())
        else:
            solver = dl.PETScKrylovSolver(self.mesh.mpi_comm(), "cg", amg_method())
        solver.set_operator(self.At)
        solver.parameters["relative_tolerance"] = tol
        self.At.init_vector(sol,1)
        nit = solver.solve(sol, rhs)
#        print "AdjInc", (self.At*sol-rhs).norm("l2")/rhs.norm("l2"), nit
    
    def applyC(self, da, out):
        self.C.mult(da,out)
    
    def applyCt(self, dp, out):
        self.C.transpmult(dp,out)
    
    def applyWuu(self, du, out):
        help = dl.Vector()
        self.B.init_vector(help, 0)
        self.B.mult(du, help)
        self.B.transpmult(help, out)
        out *= 1./self.noise_variance
    
    def applyWua(self, da, out):
        if self.gauss_newton_approx:
            out.zero()
        else:
            self.Wau.transpmult(da,out)

    
    def applyWau(self, du, out):
        if self.gauss_newton_approx:
            out.zero()
        else:
            self.Wau.mult(du, out)
    
    def applyR(self, da, out):
        self.prior.R.mult(da, out)
        
    def Rsolver(self):        
        return self.prior.Rsolver
    
    def applyRaa(self, da, out):
        if self.gauss_newton_approx:
            out.zero()
        else:
            self.Raa.mult(da, out)
            
if __name__ == "__main__":
    dl.set_log_active(False)
    sep = "\n"+"#"*80+"\n"
    
    ndim = 2
    nx = 64
    ny = 64
    mesh = dl.UnitSquareMesh(nx, ny)
    
    rank = dl.MPI.rank(mesh.mpi_comm())
    nproc = dl.MPI.size(mesh.mpi_comm())
        
    if rank == 0:  
        print sep, "Set up the mesh and finite element spaces", sep
    
    Vh2 = dl.FunctionSpace(mesh, 'Lagrange', 2)
    Vh1 = dl.FunctionSpace(mesh, 'Lagrange', 1)
    Vh = [Vh2, Vh1, Vh2]
    ndofs = [Vh[ii].dim() for ii in [STATE, PARAMETER, ADJOINT]]
    if rank == 0:
        print "Number of dofs: STATE={0}, PARAMETER={1}, ADJOINT={2}".format(*ndofs)
    
    if rank == 0:
        print sep, "Set up the location of observation, Prior Information, and model", sep
    ntargets = 300
    np.random.seed(seed=1)
    targets = np.random.uniform(0.1,0.9, [ntargets, ndim] )
    if rank == 0:
        print "Number of observation points: {0}".format(ntargets)

    orderPrior = 2
        
    if orderPrior == 1:
        gamma = 30
        delta = 30
        prior = LaplacianPrior(Vh[PARAMETER], gamma, delta)
    elif orderPrior == 2:
        gamma = 2
        delta = 5
        prior = BiLaplacianPrior(Vh[PARAMETER], gamma, delta)
    
    if rank == 0:   
        print "Prior regularization: (delta - gamma*Laplacian)^order: delta={0}, gamma={1}, order={2}".format(delta, gamma,orderPrior)    
    
    
    atrue_expression = dl.Expression('log(2+7*(pow(pow(x[0] - 0.5,2) + pow(x[1] - 0.5,2),0.5) > 0.2)) - log(10)', element=Vh[PARAMETER].ufl_element())
    prior_mean_expression = dl.Expression('log(9) - log(10)', element=Vh[PARAMETER].ufl_element())
    
    atrue = dl.interpolate(atrue_expression, Vh[PARAMETER]).vector()
    prior.mean = dl.interpolate(prior_mean_expression, Vh[PARAMETER]).vector()
    
    rel_noise = 0.01
    model = Poisson(mesh, Vh, atrue, targets, prior, rel_noise )

    if rank == 0: 
        print sep, "Test the gradient and the Hessian of the model", sep
    a0 = dl.interpolate(dl.Expression("sin(x[0])", element=Vh[PARAMETER].ufl_element()), Vh[PARAMETER])
    modelVerify(model, a0.vector(), 1e-12, is_quadratic = False, verbose = (rank == 0))
    
    if rank == 0:
        print sep, "Find the MAP point", sep
    a = prior.mean.copy()

    SOLVER = 'Newton'
    
    if SOLVER == 'Newton':
        parameters = ReducedSpaceNewtonCG_ParameterList()
        parameters["rel_tolerance"] = 1e-9
        parameters["abs_tolerance"] = 1e-12
        parameters["max_iter"]      = 25
        parameters["inner_rel_tolerance"] = 1e-15
        parameters["globalization"] = "LS"
        parameters["GN_iter"] = 5
        if rank != 0:
            parameters["print_level"] = -1
            
        solver = ReducedSpaceNewtonCG(model, parameters)
        x = solver.solve([None, a, None])
        
    elif SOLVER == 'BFGS':
        parameters = BFGS_ParameterList()
        parameters["rel_tolerance"] = 1e-9
        parameters["abs_tolerance"] = 1e-12
        parameters["max_iter"]      = 100
        parameters["inner_rel_tolerance"] = 1e-15
        parameters["globalization"] = "LS"
        parameters["GN_iter"] = 5
        if rank != 0:
            parameters["print_level"] = -1
            
        solver = BFGS(model, parameters)            
        x = solver.solve([None, a, None], [-10.0, 10.0])
        
    elif SOLVER == 'SD':
        parameters = SteepestDescent_ParameterList()
        parameters["alpha"] = 1e-2
        if rank != 0:
            parameters["print_level"] = -1
        solver =SteepestDescent(model, parameters)            
        x = solver.solve([None, a, None])

    if rank == 0:
        if solver.converged:
            print "\nConverged in ", solver.it, " iterations."
        else:
            print "\nNot Converged"

        print "Termination reason: ", solver.termination_reasons[solver.reason]
        print "Final gradient norm: ", solver.final_grad_norm
        print "Final cost: ", solver.final_cost
        
    if rank == 0:
        print sep, "Compute the low rank Gaussian Approximation of the posterior", sep
    
    model.setPointForHessianEvaluations(x, False)
    Hmisfit = ReducedHessian(model, solver.parameters["inner_rel_tolerance"], misfit_only=True)
    k = 50
    p = 20
    if rank == 0:
        print "Double Pass Algorithm. Requested eigenvectors: {0}; Oversampling {1}.".format(k,p)

    Omega = MultiVector(x[PARAMETER], k+p)
    parRandom.normal(1., Omega)

    d, U = doublePassG(Hmisfit, prior.R, prior.Rsolver, Omega, k, s=1, check=False)
    posterior = GaussianLRPosterior(prior, d, U)
    posterior.mean = x[PARAMETER]
        
    post_tr, prior_tr, corr_tr = posterior.trace(method="Estimator", tol=1e-1, min_iter=20, max_iter=100)
    if rank == 0:
        print "Posterior trace {0:5e}; Prior trace {1:5e}; Correction trace {2:5e}".format(post_tr, prior_tr, corr_tr)
    
    post_pw_variance, pr_pw_variance, corr_pw_variance = posterior.pointwise_variance("Exact")
    fid = dl.File("results/pointwise_variance.pvd")
    fid << vector2Function(post_pw_variance, Vh[PARAMETER], name="Posterior")
    fid << vector2Function(pr_pw_variance, Vh[PARAMETER], name="Prior")
    fid << vector2Function(corr_pw_variance, Vh[PARAMETER], name="Correction")
        
    if rank == 0:
        print sep, "Save State, Parameter, Adjoint, and observation in paraview", sep
    xxname = ["State", "exp(Parameter)", "Adjoint"]
    xx = [vector2Function(x[i], Vh[i], name=xxname[i]) for i in range(len(Vh))]
    dl.File("results/poisson_state.pvd") << xx[STATE]
    expc = dl.project( dl.exp( xx[PARAMETER] ), Vh[PARAMETER] )
    expc.rename("exp(Parameter)", "ignore_this")
    dl.File("results/poisson_parameter.pvd") << expc
    expc = dl.project( dl.exp( vector2Function(model.atrue, Vh[PARAMETER]) ), Vh[PARAMETER])
    expc.rename("exp(Parameter)", "ignore_this")
    dl.File("results/poisson_parameter_true.pvd") << expc
    dl.File("results/poisson_adjoint.pvd") << xx[ADJOINT]
    
    exportPointwiseObservation(Vh[STATE], model.B, model.u_o, "results/poisson_observation")
    
    
    if rank == 0:
        print sep, "Generate samples from Prior and Posterior\n","Export generalized Eigenpairs", sep
    
    fid_prior = dl.File("samples/sample_prior.pvd")
    fid_post  = dl.File("samples/sample_post.pvd")
    nsamples = 500
    noise = dl.Vector()
    posterior.init_vector(noise,"noise")
    s_prior = dl.Function(Vh[PARAMETER], name="sample_prior")
    s_post = dl.Function(Vh[PARAMETER], name="sample_post")
    for i in range(nsamples):
        parRandom.normal(1., noise)
        posterior.sample(noise, s_prior.vector(), s_post.vector())
        fid_prior << s_prior
        fid_post << s_post
        
    #Save eigenvalues for printing:
    U.export(Vh[PARAMETER], "hmisfit/evect.pvd", varname = "gen_evects", normalize = True)
    if rank == 0:
        np.savetxt("hmisfit/eigevalues.dat", d)
        
    if nproc == 1:
        print sep, "Visualize results", sep
        dl.plot(xx[STATE], title = xxname[STATE])
        dl.plot(dl.exp(xx[PARAMETER]), title = xxname[PARAMETER])
        dl.plot(xx[ADJOINT], title = xxname[ADJOINT])
        dl.interactive()
    
    if rank == 0:
        plt.figure()
        plt.plot(range(0,k), d, 'b*', range(0,k), np.ones(k), '-r')
        plt.yscale('log')
        plt.show()  
    

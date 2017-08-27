# Copyright (c) 2016, The University of Texas at Austin & University.
#
# All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the hIPPYlib library. For more information and source code
# availability see https://hippylib.github.io.
#
# hIPPYlib is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.

import numpy as np
import dolfin as dl
from variables import STATE, PARAMETER, ADJOINT
from random import parRandom


class ReducedHessianQOI:
    """
    This class implements matrix free application of the reduced hessian operator.
    The constructor takes the following parameters:
    - reduced_qoi:         the object that describes the parameter-to-qoi map
    - innerTol:            the relative tolerance for the solution of the incremental
                           forward and adjoint problems.
    """
    def __init__(self, reduced_qoi, innerTol):
        """
        Construct the Hessian Operator of the parameter-to-qoi map
        """
        self.reduced_qoi = reduced_qoi
        self.tol = innerTol
        self.ncalls = 0
        
        self.rhs_fwd = reduced_qoi.generate_vector(STATE)
        self.rhs_adj = reduced_qoi.generate_vector(ADJOINT)
        self.rhs_adj2 = reduced_qoi.generate_vector(ADJOINT)
        self.uhat    = reduced_qoi.generate_vector(STATE)
        self.phat    = reduced_qoi.generate_vector(ADJOINT)
        self.yhelp = reduced_qoi.generate_vector(PARAMETER)
    
    def init_vector(self, x, dim):
        """
        Reshape the Vector x so that it is compatible with the reduced Hessian
        operator.
        Parameters:
        - x: the vector to reshape
        - dim: if 0 then x will be reshaped to be compatible with the range of
               the reduced Hessian
               if 1 then x will be reshaped to be compatible with the domain of
               the reduced Hessian
               
        Note: Since the reduced Hessian is a self adjoint operator, the range and
              the domain is the same. Either way, we choosed to add the parameter
              dim for consistency with the interface of Matrix in dolfin.
        """
        self.reduced_qoi.init_parameter(x)
        
    def mult(self,x,y):
        """
        Apply the Hessian of the parameter-to-qoi map to the vector x
        Return the result in y.
        """
        self.reduced_qoi.applyC(x, self.rhs_fwd)
        self.reduced_qoi.solveFwdIncremental(self.uhat, self.rhs_fwd, self.tol)
        self.reduced_qoi.applyWuu(self.uhat, self.rhs_adj)
        self.reduced_qoi.applyWum(x, self.rhs_adj2)
        self.rhs_adj.axpy(-1., self.rhs_adj2)
        self.reduced_qoi.solveAdjIncremental(self.phat, self.rhs_adj, self.tol)
        self.reduced_qoi.applyWmm(x, y)
        self.reduced_qoi.applyCt(self.phat, self.yhelp)
        y.axpy(1., self.yhelp)
        self.reduced_qoi.applyWmu(self.uhat, self.yhelp)
        y.axpy(-1., self.yhelp)

        
        self.ncalls += 1
    
    def inner(self,x,y):
        """
        Perform the inner product between x and y in the norm induced by the Hessian H.
        (x, y)_H = x' H y
        """
        Ay = self.reduced_qoi.generate_vector(PARAMETER)
        Ay.zero()
        self.mult(y,Ay)
        return x.inner(Ay)


class ReducedQOI:
    def __init__(self, problem, qoi):
        """
        Create a parameter-to-qoi map given:
        - problem: the description of the forward/adjoint problem and all the sensitivities
        - qoi: the quantity of interest as a function of the state and parameter
        """
        self.problem = problem
        self.qoi = qoi
                
    def generate_vector(self, component = "ALL"):
        """
        By default, return the list [u,m,p] where:
        - u is any object that describes the state variable
        - m is a Vector object that describes the parameter variable.
          (Need to support linear algebra operations)
        - p is any object that describes the adjoint variable
        
        If component = STATE return only u
           component = PARAMETER return only a
           component = ADJOINT return only p
        """ 
        if component == "ALL":
            x = [self.problem.generate_state(), self.problem.generate_parameter(), self.problem.generate_state()]
        elif component == STATE:
            x = self.problem.generate_state()
        elif component == PARAMETER:
            x = self.problem.generate_parameter()
        elif component == ADJOINT:
            x = self.problem.generate_state()
            
        return x
    
    def init_parameter(self, m):
        """
        Reshape m so that it is compatible with the parameter variable
        """
        self.problem.init_parameter(m)
            
    def eval(self, x):
        """
        Given the list x = [u,m,p] which describes the state, parameter, and
        adjoint variable compute the QOI.
        
        Note: p is not needed to compute the QOI
        """
        return self.qoi.eval(x)
        
    def solveFwd(self, out, x, tol=1e-9):
        """
        Solve the (possibly non-linear) forward problem.
        Parameters:
        - out: is the solution of the forward problem (i.e. the state) (Output parameters)
        - x = [u,m,p] provides
              1) the parameter variable m for the solution of the forward problem
              2) the initial guess u if the forward problem is non-linear
          Note: p is not accessed
        - tol is the relative tolerance for the solution of the forward problem.
              [Default 1e-9].
        """
        self.problem.solveFwd(out, x, tol)

    
    def solveAdj(self, out, x, tol=1e-9):
        """
        Solve the linear adjoint problem.
        Parameters:
        - out: is the solution of the adjoint problem (i.e. the adjoint p) (Output parameter)
        - x = [u,m,p] provides
              1) the parameter variable m for assembling the adjoint operator
              2) the state variable u for assembling the adjoint right hand side
          Note: p is not accessed
        - tol is the relative tolerance for the solution of the adjoint problem.
              [Default 1e-9].
        """
        rhs = self.problem.generate_state()
        self.qoi.grad(STATE, x, rhs)
        rhs *= -1.
        self.problem.solveAdj(out, x, rhs, tol)
    
    def evalGradientParameter(self,x, mg):
        """
        Evaluate the gradient for the variational parameter equation at the point x=[u,m,p].
        Parameters:
        - x = [u,m,p] the point at which to evaluate the gradient.
        - mg the variational gradient (g, mtest) being mtest a test function in the parameter space
          (Output parameter)
        """ 
        self.problem.eval_da(x, mg)
        tmp = self.problem.generate_parameter()
        self.qoi.grad(PARAMETER, x, tmp)
        mg.axpy(1., tmp)

        
    
    def setLinearizationPoint(self, x):
        """
        Specify the point x = [u,m,p] at which the Hessian operator needs to be evaluated.
        Parameters:
        - x = [u,m,p]: the point at which the Hessian needs to be evaluated.
        """
        self.problem.setLinearizationPoint(x, gauss_newton_approx=False)
        self.qoi.setLinearizationPoint(x)

        
    def solveFwdIncremental(self, sol, rhs, tol):
        """
        Solve the linearized (incremental) forward problem for a given rhs
        Parameters:
        - sol the solution of the linearized forward problem (Output)
        - rhs the right hand side of the linear system
        - tol the relative tolerance for the linear system
        """
        self.problem.solveIncremental(sol,rhs, False, tol)
        
    def solveAdjIncremental(self, sol, rhs, tol):
        """
        Solve the incremental adjoint problem for a given rhs
        Parameters:
        - sol the solution of the incremental adjoint problem (Output)
        - rhs the right hand side of the linear system
        - tol the relative tolerance for the linear system
        """
        self.problem.solveIncremental(sol,rhs, True, tol)
    
    def applyC(self, dm, out):
        """
        Apply the C block of the Hessian to a (incremental) parameter variable.
        out = C dm
        Parameters:
        - dm the (incremental) parameter variable
        - out the action of the C block on dm
        
        Note: this routine assumes that out has the correct shape.
        """
        self.problem.apply_ij(ADJOINT,PARAMETER, dm, out)
    
    def applyCt(self, dp, out):
        """
        Apply the transpose of the C block of the Hessian to a (incremental) adjoint variable.
        out = C^t dp
        Parameters:
        - dp the (incremental) adjoint variable
        - out the action of the C^T block on dp
        
        Note: this routine assumes that out has the correct shape.
        """
        self.problem.apply_ij(PARAMETER,ADJOINT, dp, out)

    
    def applyWuu(self, du, out):
        """
        Apply the Wuu block of the Hessian to a (incremental) state variable.
        out = Wuu du
        Parameters:
        - du the (incremental) state variable
        - out the action of the Wuu block on du
        
        Note: this routine assumes that out has the correct shape.
        """
        self.qoi.apply_ij(STATE,STATE, du, out)
        tmp = self.generate_vector(STATE)
        self.problem.apply_ij(STATE,STATE, du, tmp)
        out.axpy(1., tmp)
    
    def applyWum(self, dm, out):
        """
        Apply the Wum block of the Hessian to a (incremental) parameter variable.
        out = Wum dm
        Parameters:
        - dm the (incremental) parameter variable
        - out the action of the Wua block on du
        
        Note: this routine assumes that out has the correct shape.
        """
        self.problem.apply_ij(STATE,PARAMETER, dm, out)
        tmp = self.generate_vector(STATE)
        self.qoi.apply_ij(STATE,PARAMETER, dm, tmp)
        out.axpy(1., tmp)

    
    def applyWmu(self, du, out):
        """
        Apply the Wmu block of the Hessian to a (incremental) state variable.
        out = Wmu du
        Parameters:
        - du the (incremental) state variable
        - out the action of the Wau block on du
        
        Note: this routine assumes that out has the correct shape.
        """
        self.problem.apply_ij(PARAMETER, STATE, du, out)
        tmp = self.generate_vector(PARAMETER)
        self.qoi.apply_ij(PARAMETER, STATE, du, tmp)
        out.axpy(1., tmp)
        
    def applyWmm(self, dm, out):
        """
        Apply the Wmm block of the Hessian to a (incremental) parameter variable.
        out = Wmm dm
        Parameters:
        - dm the (incremental) parameter variable
        - out the action of Wmm on dm
        
        Note: this routine assumes that out has the correct shape.
        """
        self.problem.apply_ij(PARAMETER,PARAMETER, dm, out)
        tmp = self.generate_vector(PARAMETER)
        self.qoi.apply_ij(PARAMETER,PARAMETER, dm, tmp)
        out.axpy(1., tmp)
        
    def reduced_eval(self, m):
        """
        Evaluate the parameter-to-qoi map at a given realization m
        Note: This evaluation requires the solution of a forward solve
        """
        u = self.problem.generate_state()
        if hasattr(self.problem, "initial_guess"):
            u.axpy(1., self.problem.initial_guess)
        self.problem.solveFwd(u, [u, m], tol=1e-6)
        return self.qoi.eval([u,m])
    
    def reduced_gradient(self, m, g):
        """
        Evaluate the gradient of parameter-to-qoi map at a given realization m
        Note: This evaluation requires the solution of a forward and adjoint solve
        """
        u = self.problem.generate_state()
        p = self.problem.generate_state()
        self.solveFwd(u, [u, m, p], self.tol)
        self.solveAdj(p, [u, m, p], self.tol)
        self.evalGradientParameter(self, [u, m, p], g)
        return [u,m,p]
    
    def reduced_hessian(self, m=None, x=None, innerTol = 1e-10):
        """
        Evaluate the Hessian of parameter-to-qoi map.
        If a relization of the parameter m is given, this function will automatically
        compute the state u and adjoint p.
        As an alternative on can provide directly x = [u, m, p]
        
        It returns an object of type ReducedHessianQOI which provides the Hessian-apply functionality
        """
        if m is not None:
            assert x is None
            u = self.problem.generate_state()
            p = self.problem.generate_state()
            self.solveFwd(u, [u, m, p], self.tol)
            self.solveAdj(p, [u, m, p], self.tol)
            x = [u, m, p]
        else:
            assert x is not None
            
        self.setLinearizationPoint(x)
        return ReducedHessianQOI(self, innerTol)
        
def reducedQOIVerify(rQOI, m0, h=None, innerTol=1e-9,eps=None, plotting = True):
    """
    Verify the gradient and the Hessian of a parameter-to-qoi map.
    It will produce two loglog plots of the finite difference checks
    for the gradient and for the Hessian.
    It will also check for symmetry of the Hessian.
    """
    rank = dl.MPI.rank(m0.mpi_comm())
    
    if h is None:
        h = rQOI.generate_vector(PARAMETER)
        parRandom.normal(1., h)

    
    x = rQOI.generate_vector()
    
    if hasattr(rQOI.problem, "initial_guess"):
        x[STATE].axpy(1., rQOI.problem.initial_guess)
    x[PARAMETER] = m0
    rQOI.solveFwd(x[STATE], x, innerTol)
    rQOI.solveAdj(x[ADJOINT], x, innerTol)
    qoi_x = rQOI.eval(x)
    
    grad_x = rQOI.generate_vector(PARAMETER)
    rQOI.evalGradientParameter(x, grad_x)
    grad_xh = grad_x.inner( h )
    
    H = rQOI.reduced_hessian(x=x, innerTol=innerTol)
    Hh = rQOI.generate_vector(PARAMETER)
    H.mult(h, Hh)
    
    if eps is None:
        n_eps = 32
        eps = np.power(.5, np.arange(n_eps-5,-5,-1))
    else:
        n_eps = eps.shape[0]
        
    err_grad = np.zeros(n_eps)
    err_H = np.zeros(n_eps)
    qois = np.zeros(n_eps)
    
    x_plus = rQOI.generate_vector()
    x_plus[STATE].axpy(1., x[STATE])
    
    for i in range(n_eps):
        my_eps = eps[i]
        
        x_plus[PARAMETER].zero()
        x_plus[PARAMETER].axpy(1., m0)
        x_plus[PARAMETER].axpy(my_eps, h)
        rQOI.solveFwd(x_plus[STATE],   x_plus, innerTol)
        rQOI.solveAdj(x_plus[ADJOINT], x_plus,innerTol)
        
        qoi_plus = rQOI.eval(x_plus)
        qois[i] = qoi_plus
        dQOI = qoi_plus - qoi_x
        err_grad[i] = abs(dQOI/my_eps - grad_xh)
        
        #Check the Hessian
        grad_xplus = rQOI.generate_vector(PARAMETER)
        rQOI.evalGradientParameter(x_plus, grad_xplus)
        
        err  = grad_xplus - grad_x
        err *= 1./my_eps
        err -= Hh
        
        err_H[i] = err.norm('linf')

        if rank == 0:
            print "{0:1.7e} {1:1.7e} {2:1.7e} {3:1.7e}".format(eps[i], qois[i], err_grad[i], err_H[i])
    
    if plotting and (rank == 0):
        reducedQOIVerifyVerifyPlotErrors(eps, err_grad, err_H) 

    out = np.zeros((eps.shape[0], 4))
    out[:,0] = eps
    out[:,1] = qois
    out[:,2] = err_grad
    out[:,3] = err_H
    
    if rank == 0:
        np.savetxt('fd_check.txt', out)
      
    xx = rQOI.generate_vector(PARAMETER)
    parRandom.normal(1., xx)
    yy = rQOI.generate_vector(PARAMETER)
    parRandom.normal(1., yy)
    
    ytHx = H.inner(yy,xx)
    xtHy = H.inner(xx,yy)
    rel_symm_error = 2*abs(ytHx - xtHy)/(ytHx + xtHy)
    if rank == 0:
        print "(yy, H xx) - (xx, H yy) = ", rel_symm_error
        if(rel_symm_error > 1e-10):
            print "HESSIAN IS NOT SYMMETRIC!!"
        
    return out

def reducedQOIVerifyVerifyPlotErrors(eps, err_grad, err_H):
    try:
        import matplotlib.pyplot as plt
    except:
        print "Matplotlib is not installed."
        return
    
    plt.figure()
    plt.subplot(121)
    plt.loglog(eps, err_grad, "-ob", eps, eps*(err_grad[0]/eps[0]), "-.k")
    plt.title("FD Gradient Check")
    plt.subplot(122)
    plt.loglog(eps, err_H, "-ob", eps, eps*(err_H[0]/eps[0]), "-.k")
    plt.title("FD Hessian Check")

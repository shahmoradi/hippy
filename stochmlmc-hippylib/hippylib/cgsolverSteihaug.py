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

from dolfin import Vector
from parameterList import ParameterList
import math

def CGSolverSteihaug_ParameterList():
    """
    Generate a ParameterList for CGSolverSteihaug.
    Type: CGSolverSteihaug_ParameterList().showMe() for default values and their descriptions
    """
    parameters = {}
    parameters["rel_tolerance"] = [1e-9, "the relative tolerance for the stopping criterion"]
    parameters["abs_tolerance"] = [1e-12, "the absolute tolerance for the stopping criterion"]
    parameters["max_iter"]      = [1000, "the maximum number of iterations"]
    parameters["zero_initial_guess"] = [True, "if True we start with a 0 initial guess; if False we use the x as initial guess."]
    parameters["print_level"] = [0, "verbosity level: -1 --> no output on screen; 0 --> only final residual at convergence or reason for not not convergence"]
    return ParameterList(parameters)

class CGSolverSteihaug:
    """
    Solve the linear system A x = b using preconditioned conjugate gradient ( B preconditioner)
    and the Steihaug stopping criterion:
    - reason of termination 0: we reached the maximum number of iterations (no convergence)
    - reason of termination 1: we reduced the residual up to the given tolerance (convergence)
    - reason of termination 2: we reached a negative direction (premature termination due to not spd matrix)
    - reason of termination 3: we reached the boundary of the trust region
    
    The stopping criterion is based on either
    - the absolute preconditioned residual norm check: || r^* ||_{B^{-1}} < atol
    - the relative preconditioned residual norm check: || r^* ||_{B^{-1}}/|| r^0 ||_{B^{-1}} < rtol
    where r^* = b - Ax^* is the residual at convergence and r^0 = b - Ax^0 is the initial residual.
    
    The operator A is set using the method set_operator(A).
    A must provide the following two methods:
    - A.mult(x,y): y = A*x
    - A.init_vector(x, dim): initialize the vector x so that it is compatible with the range (dim = 0) or
      the domain (dim = 1) of A.
      
    The preconditioner B is set using the method set_preconditioner(B).
    B must provide the following method:
    - B.solve(z,r): z is the action of the preconditioner B on the vector r
    
    To solve the linear system A*x = b call self.solve(x,b).
    Here x and b are assumed to be FEniCS::Vector objects.
    
    Type: CGSolverSteihaug_ParameterList().showMe() for default parameters and their descriptions
    """

    reason = ["Maximum Number of Iterations Reached",
              "Relative/Absolute residual less than tol",
              "Reached a negative direction",
              "Reached trust region boundary"
              ]
    def __init__(self, parameters=CGSolverSteihaug_ParameterList()):
        
        self.parameters = parameters
        
        self.A = None
        self.B_solver = None
        self.B_op = None
        self.converged = False
        self.iter = 0
        self.reasonid = 0
        self.final_norm = 0

        self.TR_radius_2 = None

        self.update_x = self.update_x_without_TR
        
        self.r = Vector()
        self.z = Vector()
        self.d = Vector()
        self.Bx = Vector()
                
    def set_operator(self, A):
        """
        Set the operator A.
        """
        self.A = A
        self.A.init_vector(self.r,0)
        self.A.init_vector(self.z,0)
        self.A.init_vector(self.d,0)
        
        
    def set_preconditioner(self, B_solver):
        """
        Set the preconditioner B.
        """
        self.B_solver = B_solver

    def set_TR(self,radius,B_op):
        assert self.parameters["zero_initial_guess"]
        self.TR_radius_2 = radius*radius
        self.update_x = self.update_x_with_TR
        self.B_op = B_op
        self.B_op.init_vector(self.Bx,0)

    def update_x_without_TR(self,x,alpha,d):
        x.axpy(alpha,d)
        return False

    def update_x_with_TR(self,x,alpha,d):
        x_bk = x.copy()
        x.axpy(alpha,d)
        self.Bx.zero()
        self.B_op.mult(x, self.Bx)
        x_Bnorm2 = self.Bx.inner(x)

        if x_Bnorm2 < self.TR_radius_2:
            return  False
        else:
            # Move point to boundary of trust region
            self.Bx.zero()
            self.B_op.mult(x_bk, self.Bx)
            x_Bnorm2 = self.Bx.inner(x_bk)
            Bd = Vector()
            self.B_op.init_vector(Bd,0)
            Bd.zero()
            self.B_op.mult(self.d,Bd)
            d_Bnorm2 = Bd.inner(d)
            d_Bx = Bd.inner(x_bk)
            a_tau = alpha*alpha*d_Bnorm2
            b_tau_half = alpha* d_Bx
            c_tau = x_Bnorm2- self.TR_radius_2
            # Solve quadratic for tau
            tau = (-b_tau_half + math.sqrt(b_tau_half*b_tau_half - a_tau*c_tau))/a_tau
            x.zero()
            x.axpy(1,x_bk)
            x.axpy(tau*alpha, d)

            return  True
        
    def solve(self,x,b):
        """
        Solve the linear system Ax = b
        """
        self.iter = 0
        self.converged = False
        self.reasonid  = 0
        
        betanom = 0.0
        alpha = 0.0 
        beta = 0.0
                
        if self.parameters["zero_initial_guess"]:
            self.r.zero()
            self.r.axpy(1.0, b)
            x.zero()
        else:
            assert self.TR_radius_2==None
            self.A.mult(x,self.r)
            self.r *= -1.0
            self.r.axpy(1.0, b)
        
        self.z.zero()
        self.B_solver.solve(self.z,self.r) #z = B^-1 r  
              
        self.d.zero()
        self.d.axpy(1.,self.z); #d = z
        
        nom0 = self.d.inner(self.r)
        nom = nom0
        
        if self.parameters["print_level"] == 1:
            print " Iterartion : ", 0, " (B r, r) = ", nom
            
        rtol2 = nom * self.parameters["rel_tolerance"] * self.parameters["rel_tolerance"]
        atol2 = self.parameters["abs_tolerance"] * self.parameters["abs_tolerance"]
        r0 = max(rtol2, atol2)
        
        if nom <= r0:
            self.converged  = True
            self.reasonid   = 1
            self.final_norm = math.sqrt(nom)
            if(self.parameters["print_level"] >= 0):
                print self.reason[self.reasonid]
                print "Converged in ", self.iter, " iterations with final norm ", self.final_norm
            return
        
        self.A.mult(self.d, self.z)  #z = A d
        den = self.z.inner(self.d)
        
        if den <= 0.0:
            self.converged = True
            self.reasonid = 2
            x.axpy(1., self.d)
            self.r.axpy(-1., self.z)
            self.B_solver.solve(self.z, self.r)
            nom = self.r.inner(self.z)
            self.final_norm = math.sqrt(nom)
            if(self.parameters["print_level"] >= 0):
                print self.reason[self.reasonid]
                print "Converged in ", self.iter, " iterations with final norm ", self.final_norm
            return
        
        # start iteration
        self.iter = 1
        while True:
            alpha = nom/den
            TrustBool = self.update_x(x,alpha,self.d)   # x = x + alpha d
            if TrustBool == True:
                self.converged = True
                self.reasonid = 3
                self.final_norm = math.sqrt(betanom)
                if(self.parameters["print_level"] >= 0):
                    print self.reason[self.reasonid]
                    print "Converged in ", self.iter, " iterations with final norm ", self.final_norm
                break

            self.r.axpy(-alpha, self.z) # r = r - alpha A d
            
            self.B_solver.solve(self.z, self.r)     # z = B^-1 r
            betanom = self.r.inner(self.z)
            
            if self.parameters["print_level"] == 1:
                print " Iteration : ", self.iter, " (B r, r) = ", betanom
                
            if betanom < r0:
                self.converged = True
                self.reasonid = 1
                self.final_norm = math.sqrt(betanom)
                if(self.parameters["print_level"] >= 0):
                    print self.reason[self.reasonid]
                    print "Converged in ", self.iter, " iterations with final norm ", self.final_norm
                break
            
            self.iter += 1
            if self.iter > self.parameters["max_iter"]:
                self.converged = False
                self.reasonid = 0
                self.final_norm = math.sqrt(betanom)
                if(self.parameters["print_level"] >= 0):
                    print self.reason[self.reasonid]
                    print "Not Converged. Final residual norm ", self.final_norm
                break
            
            beta = betanom/nom
            self.d *= beta
            self.d.axpy(1., self.z)  #d = z + beta d
            
            self.A.mult(self.d,self.z)   # z = A d
            
            den = self.d.inner(self.z)
            
            if den <= 0.0:
                self.converged = True
                self.reasonid = 2
                self.final_norm = math.sqrt(nom)
                if(self.parameters["print_level"] >= 0):
                    print self.reason[self.reasonid]
                    print "Converged in ", self.iter, " iterations with final norm ", self.final_norm
                break
            
            nom = betanom


                
            
            
        
        

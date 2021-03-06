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

import math
from parameterList import ParameterList
from variables import STATE, PARAMETER, ADJOINT
from cgsolverSteihaug import CGSolverSteihaug
from reducedHessian import ReducedHessian

def LS_ParameterList():
    """
    Generate a ParameterList for line search globalization.
    type: LS_ParameterList().showMe() for default values and their descriptions
    """
    parameters = {}
    parameters["c_armijo"]              = [1e-4, "Armijo constant for sufficient reduction"]
    parameters["max_backtracking_iter"] = [10, "Maximum number of backtracking iterations"]
    
    return ParameterList(parameters)

def TR_ParameterList():
    """
    Generate a ParameterList for Trust Region globalization.
    type: RT_ParameterList().showMe() for default values and their descriptions
    """
    parameters = {}
    parameters["eta"] = [0.05, "Reject step if (actual reduction)/(predicted reduction) < eta"]
    
    return ParameterList(parameters)

def ReducedSpaceNewtonCG_ParameterList():
    """
    Generate a ParameterList for ReducedSpaceNewtonCG.
    type: ReducedSpaceNewtonCG_ParameterList().showMe() for default values and their descriptions
    """
    parameters = {}
    parameters["rel_tolerance"]         = [1e-6, "we converge when sqrt(g,g)/sqrt(g_0,g_0) <= rel_tolerance"]
    parameters["abs_tolerance"]         = [1e-12, "we converge when sqrt(g,g) <= abs_tolerance"]
    parameters["gda_tolerance"]         = [1e-18, "we converge when (g,da) <= gda_tolerance"]
    parameters["max_iter"]              = [20, "maximum number of iterations"]
    parameters["inner_rel_tolerance"]   = [1e-9, "relative tolerance used for the solution of the forward, adjoint, and incremental (fwd,adj) problems"]
    parameters["globalization"]         = ["LS", "Globalization technique: line search (LS)  or trust region (TR)"]
    parameters["print_level"]           = [0, "Control verbosity of printing screen"]
    parameters["GN_iter"]               = [5, "Number of Gauss Newton iterations before switching to Newton"]
    parameters["cg_coarse_tolerance"]   = [.5, "Coarsest tolerance for the CG method (Eisenstat-Walker)"]
    parameters["LS"]                    = [LS_ParameterList(), "Sublist containing LS globalization parameters"]
    parameters["TR"]                    = [TR_ParameterList(), "Sublist containing TR globalization parameters"]
    
    return ParameterList(parameters)
  
    

class ReducedSpaceNewtonCG:
    
    """
    Inexact Newton-CG method to solve constrained optimization problems in the reduced parameter space.
    The Newton system is solved inexactly by early termination of CG iterations via Eisenstat-Walker
    (to prevent oversolving) and Steihaug (to avoid negative curvature) criteria.
    Globalization is performed using one of the following methods:
    - line search (LS) based on the armijo sufficient reduction condition; or
    - trust region (TR) based on the prior preconditioned norm of the update direction.
    The stopping criterion is based on a control on the norm of the gradient and a control of the
    inner product between the gradient and the Newton direction.
       
    The user must provide a model that describes the forward problem, cost functionals, and all the
    derivatives for the gradient and the Hessian.
    
    More specifically the model object should implement following methods:
       - generate_vector() -> generate the object containing state, parameter, adjoint
       - cost(x) -> evaluate the cost functional, report regularization part and misfit separately
       - solveFwd(out, x,tol) -> solve the possibly non linear Fwd Problem up a tolerance tol
       - solveAdj(out, x,tol) -> solve the linear adj problem
       - evalGradientParameter(x, out) -> evaluate the gradient of the parameter and compute its norm
       - setPointForHessianEvaluations(x) -> set the state to perform hessian evaluations
       - solveFwdIncremental(out, rhs, tol) -> solve the linearized forward problem for a given rhs
       - solveAdjIncremental(out, rhs, tol) -> solve the linear adjoint problem for a given rhs
       - applyC(da, out)    --> Compute out = C_x * da
       - applyCt(dp, out)   --> Compute out = C_x' * dp
       - applyWuu(du,out)   --> Compute out = Wuu_x * du
       - applyWua(da, out)  --> Compute out = Wua_x * da
       - applyWau(du, out)  --> Compute out = Wau * du
       - applyR(da, out)    --> Compute out = R * da
       - applyRaa(da,out)   --> Compute out = Raa * out
       - Rsolver()          --> A solver for the regularization term
       
    Type help(Model) for additional information
    """
    termination_reasons = [
                           "Maximum number of Iteration reached",      #0
                           "Norm of the gradient less than tolerance", #1
                           "Maximum number of backtracking reached",   #2
                           "Norm of (g, da) less than tolerance"       #3
                           ]
    
    def __init__(self, model, parameters=ReducedSpaceNewtonCG_ParameterList()):
        """
        Initialize the ReducedSpaceNewtonCG.
        Type `ReducedSpaceNewtonCG_ParameterList().showMe()` for list of default parameters
        and their descriptions.
        """
        self.model = model
        self.parameters = parameters
        
        self.it = 0
        self.converged = False
        self.total_cg_iter = 0
        self.ncalls = 0
        self.reason = 0
        self.final_grad_norm = 0
        
    def solve(self, x):
        """
        INPUT: x = [u, a, p] represents the initial guess (u and p may be None).
        x will be overwritten.
        Returns x
        """
        
        if x[STATE] is None:
            x[STATE] = self.model.generate_vector(STATE)
            
        if x[ADJOINT] is None:
            x[ADJOINT] = self.model.generate_vector(ADJOINT)
            
        if self.parameters["globalization"] == "LS":
            return self._solve_ls(x)
        elif self.parameters["globalization"] == "TR":
            return self._solve_tr(x)
        else:
            raise ValueError(self.parameters["globalization"])
        
    def _solve_ls(self,x):
        """
        Solve the constrained optimization problem with initial guess x.
        """
        rel_tol = self.parameters["rel_tolerance"]
        abs_tol = self.parameters["abs_tolerance"]
        max_iter = self.parameters["max_iter"]
        innerTol = self.parameters["inner_rel_tolerance"]
        print_level = self.parameters["print_level"]
        GN_iter = self.parameters["GN_iter"]
        cg_coarse_tolerance = self.parameters["cg_coarse_tolerance"]
        
        c_armijo = self.parameters["LS"]["c_armijo"]
        max_backtracking_iter = self.parameters["LS"]["max_backtracking_iter"]
        
        u,a,p = x
        self.model.solveFwd(u, x, innerTol)
        
        self.it = 0
        self.converged = False
        self.ncalls += 1
        
        a_star = self.model.generate_vector(PARAMETER)
        ahat = self.model.generate_vector(PARAMETER)    
        mg = self.model.generate_vector(PARAMETER)
        
        u_star = self.model.generate_vector(STATE)
        
        cost_old, _, _ = self.model.cost(x)
        
        while (self.it < max_iter) and (self.converged == False):
            self.model.solveAdj(p, x, innerTol)
            
            self.model.setPointForHessianEvaluations(x, gauss_newton_approx=(self.it < GN_iter) )
            gradnorm = self.model.evalGradientParameter(x, mg)
            
            if self.it == 0:
                gradnorm_ini = gradnorm
                tol = max(abs_tol, gradnorm_ini*rel_tol)
                
            # check if solution is reached
            if (gradnorm < tol) and (self.it > 0):
                self.converged = True
                self.reason = 1
                break
            
            self.it += 1
            
            tolcg = min(cg_coarse_tolerance, math.sqrt(gradnorm/gradnorm_ini))
            
            HessApply = ReducedHessian(self.model, innerTol)
            solver = CGSolverSteihaug()
            solver.set_operator(HessApply)
            solver.set_preconditioner(self.model.Rsolver())
            solver.parameters["rel_tolerance"] = tolcg
            solver.parameters["zero_initial_guess"] = True
            solver.parameters["print_level"] = print_level-1
            
            solver.solve(ahat, -mg)
            self.total_cg_iter += HessApply.ncalls
            
            alpha = 1.0
            descent = 0
            n_backtrack = 0
            
            mg_ahat = mg.inner(ahat)
            
            while descent == 0 and n_backtrack < max_backtracking_iter:
                # update a and u
                a_star.zero()
                a_star.axpy(1., a)
                a_star.axpy(alpha, ahat)
                u_star.zero()
                u_star.axpy(1., u)
                self.model.solveFwd(u_star, [u_star, a_star, p], innerTol)
                
                cost_new, reg_new, misfit_new = self.model.cost([u_star,a_star,p])
                
                # Check if armijo conditions are satisfied
                if (cost_new < cost_old + alpha * c_armijo * mg_ahat) or (-mg_ahat <= self.parameters["gda_tolerance"]):
                    cost_old = cost_new
                    descent = 1
                    a.zero()
                    a.axpy(1., a_star)
                    u.zero()
                    u.axpy(1., u_star)
                else:
                    n_backtrack += 1
                    alpha *= 0.5
                            
            if(print_level >= 0) and (self.it == 1):
                print "\n{0:3} {1:3} {2:15} {3:15} {4:15} {5:15} {6:14} {7:14} {8:14}".format(
                      "It", "cg_it", "cost", "misfit", "reg", "(g,da)", "||g||L2", "alpha", "tolcg")
                
            if print_level >= 0:
                print "{0:3d} {1:3d} {2:15e} {3:15e} {4:15e} {5:15e} {6:14e} {7:14e} {8:14e}".format(
                        self.it, HessApply.ncalls, cost_new, misfit_new, reg_new, mg_ahat, gradnorm, alpha, tolcg)
                
            if n_backtrack == max_backtracking_iter:
                self.converged = False
                self.reason = 2
                break
            
            if -mg_ahat <= self.parameters["gda_tolerance"]:
                self.converged = True
                self.reason = 3
                break
                            
        self.final_grad_norm = gradnorm
        self.final_cost      = cost_new
        return [u,a,p]
    
    def _solve_tr(self,x):
        rel_tol = self.parameters["rel_tolerance"]
        abs_tol = self.parameters["abs_tolerance"]
        max_iter = self.parameters["max_iter"]
        innerTol = self.parameters["inner_rel_tolerance"]
        print_level = self.parameters["print_level"]
        GN_iter = self.parameters["GN_iter"]
        cg_coarse_tolerance = self.parameters["cg_coarse_tolerance"]
        
        eta_TR = self.parameters["TR"]["eta"]
        delta_TR = None
        
        
        u,a,p = x
        self.model.solveFwd(u, x, innerTol)
        
        self.it = 0
        self.converged = False
        self.ncalls += 1
        
        a_star = self.model.generate_vector(PARAMETER)
        ahat = self.model.generate_vector(PARAMETER) 
        R_ahat = self.model.generate_vector(PARAMETER)   
        mg = self.model.generate_vector(PARAMETER)
        
        u_star = self.model.generate_vector(STATE)
        
        cost_old, reg_old, misfit_old = self.model.cost(x)
        while (self.it < max_iter) and (self.converged == False):
            self.model.solveAdj(p, x, innerTol)
            
            self.model.setPointForHessianEvaluations(x, gauss_newton_approx=(self.it < GN_iter) )
            gradnorm = self.model.evalGradientParameter(x, mg)
            
            if self.it == 0:
                gradnorm_ini = gradnorm
                tol = max(abs_tol, gradnorm_ini*rel_tol)
                
            # check if solution is reached
            if (gradnorm < tol) and (self.it > 0):
                self.converged = True
                self.reason = 1
                break
            
            self.it += 1
            

            tolcg = min(cg_coarse_tolerance, math.sqrt(gradnorm/gradnorm_ini))
            
            HessApply = ReducedHessian(self.model, innerTol)
            solver = CGSolverSteihaug()
            solver.set_operator(HessApply)
            solver.set_preconditioner(self.model.Rsolver())
            if self.it > 1:
                solver.set_TR(delta_TR, self.model.prior.R)
            solver.parameters["rel_tolerance"] = tolcg
            solver.parameters["print_level"] = print_level-1
            
            solver.solve(ahat, -mg)
            self.total_cg_iter += HessApply.ncalls

            if self.it == 1:
                self.model.prior.R.mult(ahat,R_ahat)
                ahat_Rnorm = R_ahat.inner(ahat)
                delta_TR = max(math.sqrt(ahat_Rnorm),1)

            a_star.zero()
            a_star.axpy(1., a)
            a_star.axpy(1., ahat)   #a_star = a +ahat
            u_star.zero()
            u_star.axpy(1., u)      #u_star = u
            self.model.solveFwd(u_star, [u_star, a_star, p], innerTol)
            cost_star, reg_star, misfit_star = self.model.cost([u_star,a_star,p])
            ACTUAL_RED = cost_old - cost_star
            #Calculate Predicted Reduction
            H_ahat = self.model.generate_vector(PARAMETER)
            H_ahat.zero()
            HessApply.mult(ahat,H_ahat)
            mg_ahat = mg.inner(ahat)
            PRED_RED = -0.5*ahat.inner(H_ahat) - mg_ahat
            # print "PREDICTED REDUCTION", PRED_RED, "ACTUAL REDUCTION", ACTUAL_RED
            rho_TR = ACTUAL_RED/PRED_RED


            # Nocedal and Wright Trust Region conditions (page 69)
            if rho_TR < 0.25:
                delta_TR *= 0.5
            elif rho_TR > 0.75 and solver.reasonid == 3:
                delta_TR *= 2.0
            

            # print "rho_TR", rho_TR, "eta_TR", eta_TR, "rho_TR > eta_TR?", rho_TR > eta_TR , "\n"
            if rho_TR > eta_TR:
                a.zero()
                a.axpy(1.0,a_star)
                u.zero()
                u.axpy(1.0,u_star)
                cost_old = cost_star
                reg_old = reg_star
                misfit_old = misfit_star
                accept_step = True
            else:
                accept_step = False
                
                            
            if(print_level >= 0) and (self.it == 1):
                print "\n{0:3} {1:3} {2:15} {3:15} {4:15} {5:15} {6:14} {7:14} {8:14} {9:11} {10:14}".format(
                      "It", "cg_it", "cost", "misfit", "reg", "(g,da)", "||g||L2", "TR Radius", "rho_TR", "Accept Step","tolcg")
                
            if print_level >= 0:
                print "{0:3d} {1:3d} {2:15e} {3:15e} {4:15e} {5:15e} {6:14e} {7:14e} {8:14e} {9:11} {10:14e}".format(
                        self.it, HessApply.ncalls, cost_old, misfit_old, reg_old, mg_ahat, gradnorm, delta_TR, rho_TR, accept_step,tolcg)
                

            #TR radius can make this term arbitrarily small and prematurely exit.
            if -mg_ahat <= self.parameters["gda_tolerance"]:
                self.converged = True
                self.reason = 3
                break
                            
        self.final_grad_norm = gradnorm
        self.final_cost      = cost_old
        return [u,a,p]

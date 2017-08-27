import numpy as np

from variables import STATE, PARAMETER, ADJOINT
from parameterList import ParameterList
from NewtonCG import LS_ParameterList

def BFGSoperator_ParameterList():
    parameters = {}
    parameters["BFGS_damping"] = [0.2, "Damping of BFGS"]
    parameters["memory_limit"] = [np.inf, "Number of vectors to store in limited memory BFGS"]
    return ParameterList(parameters)

def BFGS_ParameterList():
    parameters = {}
    parameters["rel_tolerance"]         = [1e-6, "we converge when sqrt(g,g)/sqrt(g_0,g_0) <= rel_tolerance"]
    parameters["abs_tolerance"]         = [1e-12, "we converge when sqrt(g,g) <= abs_tolerance"]
    parameters["gda_tolerance"]         = [1e-18, "we converge when (g,da) <= gda_tolerance"]
    parameters["max_iter"]              = [500, "maximum number of iterations"]
    parameters["inner_rel_tolerance"]   = [1e-9, "relative tolerance used for the solution of the forward, adjoint, and incremental (fwd,adj) problems"]
    parameters["globalization"]         = ["LS", "Globalization technique: line search (LS)  or trust region (TR)"]
    parameters["print_level"]           = [0, "Control verbosity of printing screen"]
    parameters["GN_iter"]               = [5, "Number of Gauss Newton iterations before switching to Newton"]
    parameters["cg_coarse_tolerance"]   = [.5, "Coarsest tolerance for the CG method (Eisenstat-Walker)"]
    ls_list = LS_ParameterList()
    ls_list["max_backtracking_iter"] = 25
    parameters["LS"]                    = [ls_list, "Sublist containing LS globalization parameters"]
    parameters["H0inv"]                 = ["Rinv", "Initial BFGS operator: Rinv -> inverse of regularization, Minv -> inverse of mass matrix, I-> identity"]
    parameters["BFGS_op"]               = [BFGSoperator_ParameterList(), "BFGS operator"]
    return ParameterList(parameters)

class H0invdefault():
    """
    Default operator for H0inv
    Corresponds to applying d0*I
    """
    def __init__(self):
        self.d0 = 1.0

    def solve(self, x, b):
        x.zero()
        x.axpy(self.d0, b)


class BFGS_operator:

    def __init__(self, parameters=BFGSoperator_ParameterList()):
        self.S, self.Y, self.R = [],[],[]

        self.H0inv = H0invdefault()
        self.isH0invdefault = True
        self.updated0 = True

        self.parameters = parameters

    def set_H0inv(self, H0inv):
        """
        Set user-defined operator corresponding to H0inv
        Input:
            H0inv: Fenics operator with method 'solve'
        """
        self.H0inv = H0inv
        self.isH0invdefault = False


    def solve(self, x, b):
        """
        Solve system:           H_bfgs * x = b
        where H_bfgs is the approximation to the Hessian build by BFGS. 
        That is, we apply
                                x = (H_bfgs)^{-1} * b
                                  = Hk * b
        where Hk matrix is BFGS approximation to the inverse of the Hessian.
        Computation done via double-loop algorithm.
        Inputs:
            x = vector (Fenics) [out]; x = Hk*b
            b = vector (Fenics) [in]
        """
        A = []
        x.zero()
        x.axpy(1.0, b)

        for s, y, r in zip(reversed(self.S), reversed(self.Y), reversed(self.R)):
            a = r * s.inner(x)
            A.append(a)
            x.axpy(-a, y)

        x_copy = x.copy()
        self.H0inv.solve(x, x_copy)     # x = H0 * x_copy

        for s, y, r, a in zip(self.S, self.Y, self.R, reversed(A)):
            b = r * y.inner(x)
            x.axpy(a - b, s)


    def update(self, s, y):
        """
        Update BFGS operator with most recent gradient update
        To handle potential break from secant condition, update done via damping
        Input:
            s = Vector (Fenics) [in]; corresponds to update in medium parameters
            y = Vector (Fenics) [in]; corresponds to update in gradient
        """
        damp = self.parameters["BFGS_damping"]
        memlim = self.parameters["memory_limit"]
        Hy = y.copy()

        sy = s.inner(y)
        self.solve(Hy, y)
        yHy = y.inner(Hy)
        theta = 1.0
        if sy < damp*yHy:
            theta = (1.0-damp)*yHy/(yHy-sy)
            s *= theta
            s.axpy(1-theta, Hy)
            sy = s.inner(y)
        assert(sy > 0.)
        rho = 1./sy
        self.S.append(s.copy())
        self.Y.append(y.copy())
        self.R.append(rho)

        # if L-BFGS
        if len(self.S) > memlim:
            self.S.pop(0)
            self.Y.pop(0)
            self.R.pop(0)
            self.updated0 = True

        # re-scale H0 based on earliest secant information
        if self.isH0invdefault and self.updated0:
            s0  = self.S[0]
            y0 = self.Y[0]
            d0 = s0.inner(y0) / y0.inner(y0)
            self.H0inv.d0 = d0
            self.udpated0 = False

        return theta



class BFGS:
    """
    Implement BFGS technique with backtracking inexact line search and damped updating
    See Nocedal & Wright (06), $6.2, $7.3, $18.3

    The user must provide a model that describes the forward problem, cost functionals, and all the
    derivatives for the gradient and the Hessian.
    
    More specifically the model object should implement following methods:
       - generate_vector() -> generate the object containing state, parameter, adjoint
       - cost(x) -> evaluate the cost functional, report regularization part and misfit separately
       - solveFwd(out, x,tol) -> solve the possibly non linear Fwd Problem up a tolerance tol
       - solveAdj(out, x,tol) -> solve the linear adj problem
       - evalGradientParameter(x, out) -> evaluate the gradient of the parameter and compute its norm
       - applyR(da, out)    --> Compute out = R * da
       - Rsolver()          --> A solver for the regularization term
       
    Type help(Model) for additional information
    """
    termination_reasons = [
                           "Maximum number of Iteration reached",      #0
                           "Norm of the gradient less than tolerance", #1
                           "Maximum number of backtracking reached",   #2
                           "Norm of (g, da) less than tolerance"       #3
                           ]

    def __init__(self, model, parameters=BFGS_ParameterList()):
        """
        Initialize the BFGS solver.
        Type BFGS_ParameterList().showMe() for default parameters and their description
        """
        self.model = model
        
        self.parameters = parameters        
        self.it = 0
        self.converged = False
        self.ncalls = 0
        self.reason = 0
        self.final_grad_norm = 0

        self.BFGSop = BFGS_operator(self.parameters["BFGS_op"])


    def solve(self, x, bounds_xPARAM=None):
        """
        Solve the constrained optimization problem with initial guess x = [u, a0, p]. Note: u and p may be None.
        x will be overwritten.
        bounds_xPARAM: set bounds for parameter a in line search to avoid potential instabilities
        Return the solution [u,a,p] 
        """
        rel_tol = self.parameters["rel_tolerance"]
        abs_tol = self.parameters["abs_tolerance"]
        max_iter = self.parameters["max_iter"]
        innerTol = self.parameters["inner_rel_tolerance"]
        ls_list = self.parameters[self.parameters["globalization"]]
        c_armijo = ls_list["c_armijo"]
        max_backtracking_iter = ls_list["max_backtracking_iter"]
        print_level = self.parameters["print_level"]

        H0inv = self.parameters['H0inv']
        self.BFGSop.parameters["BFGS_damping"] = self.parameters["BFGS_op"]["BFGS_damping"]
        self.BFGSop.parameters["memory_limit"] = self.parameters["BFGS_op"]["memory_limit"]

        if x[STATE] is None:
            x[STATE] = self.model.generate_vector(STATE)
            
        if x[ADJOINT] is None:
            x[ADJOINT] = self.model.generate_vector(ADJOINT)
            
        [u,a,p] = x
        self.model.solveFwd(u, x, innerTol)
        
        self.it = 0
        self.converged = False
        self.ncalls += 1
        
        a_star = self.model.generate_vector(PARAMETER) 
        ahat = self.model.generate_vector(PARAMETER)    
        mg = self.model.generate_vector(PARAMETER)
        
        u_star = self.model.generate_vector(STATE) 
        
        cost_old, reg_old, misfit_old = self.model.cost(x)

        if(print_level >= 0):
            print "\n {:3} {:15} {:15} {:15} {:15} {:14} {:14} {:14}".format(
            "It", "cost", "misfit", "reg", "(g,da)", "||g||L2", "alpha", "theta")
            print "{:3d} {:15e} {:15e} {:15e} {:15} {:14} {:14} {:14}".format(
            self.it, cost_old, misfit_old, reg_old, "", "", "", "")
        
        while (self.it < max_iter) and (self.converged == False):
            self.model.solveAdj(p, x, innerTol)
            
            # update H0
            if H0inv == 'Rinv':
                self.model.setPointForHessianEvaluations(x)
                self.BFGSop.set_H0inv(self.model.prior.Rsolver)
            elif H0inv == 'Minv':
                self.BFGSop.set_H0inv(self.model.prior.Msolver)

            mg_old = mg.copy()
            gradnorm = self.model.evalGradientParameter(x, mg)
            # Update BFGS
            if self.it > 0:
                s = ahat * alpha
                y = mg - mg_old
                theta = self.BFGSop.update(s, y)
            else:
                gradnorm_ini = gradnorm
                tol = max(abs_tol, gradnorm_ini*rel_tol)
                theta = 1.0
                
            # check if solution is reached
            if (gradnorm < tol) and (self.it > 0):
                self.converged = True
                self.reason = 1
                break
            
            self.it += 1

            # compute search direction with BFGS:
            self.BFGSop.solve(ahat, -mg)
            
            # backtracking line-search
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
                if bounds_xPARAM is not None:
                    amin = a_star.min()
                    amax = a_star.max()
                    if amin < bounds_xPARAM[0] or amax > bounds_xPARAM[1]:
                        n_backtrack += 1
                        alpha *= 0.5
                        continue
                self.model.solveFwd(u_star, [u_star, a_star, p], innerTol)
                cost_new, reg_new, misfit_new = self.model.cost([u_star, a_star, p])
                
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

            if print_level >= 0:
                print "{:3d} {:15e} {:15e} {:15e} {:15e} {:14e} {:14e} {:14e}".format(
                self.it, cost_new, misfit_new, reg_new, mg_ahat, gradnorm, alpha, theta)
                
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
        return x

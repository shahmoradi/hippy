import math
from variables import STATE, PARAMETER, ADJOINT
from posterior import LowRankHessianMisfit
from random import parRandom
from vector2function import vector2Function
import numpy as np
import dolfin as dl


class NullQoi(object):
    def __init__(self):
        pass
    def eval(self,x):
        return 0.

class NullTracer(object):
    def __init__(self):
        pass
    def append(self,current, q):
        pass
    
class QoiTracer(object):
    def __init__(self, n):
        self.data = np.zeros(n)
        self.i = 0
        
    def append(self,current, q):
        self.data[self.i] = q
        self.i+=1
        
class FullTracer(object):
    def __init__(self, n, Vh, par_fid = None, state_fid = None, ):
        self.data = np.zeros((n,2))
        self.i = 0
        self.Vh = Vh
        self.par_fid = par_fid
        self.state_fid = state_fid
        
    def append(self,current, q):
        self.data[self.i, 0] = q
        self.data[self.i, 1] = current.cost
        if self.par_fid is not None:
            self.par_fid << vector2Function(current.m, self.Vh[PARAMETER], name="parameter")
        if self.state_fid is not None:
            self.state_fid << vector2Function(current.u, self.Vh[STATE], name = "state")
        self.i+=1

class SampleStruct:
    def __init__(self, kernel):
        self.derivative_info = kernel.derivativeInfo()
        self.u = kernel.model.generate_vector(STATE)
        self.m = kernel.model.generate_vector(PARAMETER)
        self.cost = 0
        
        if self.derivative_info >= 1:
            self.p  = kernel.model.generate_vector(STATE)
            self.g  = kernel.model.generate_vector(PARAMETER)
            self.Cg = kernel.model.generate_vector(PARAMETER)
        else:
            self.p = None
            self.g = None
        
    def assign(self, other):
        assert self.derivative_info == other.derivative_info
        self.cost = other.cost
        
        self.m = other.m.copy()
        self.u = other.u.copy()
        
        if self.derivative_info >= 1:
            self.g = other.g.copy()
            self.p = other.p.copy()
            self.Cg = other.Cg.copy()


class MCMC(object):
    def __init__(self, kernel):
        self.kernel = kernel
        self.parameters = {}
        self.parameters["number_of_samples"]     = 2000
        self.parameters["burn_in"]               = 1000
        self.parameters["print_progress"]        = 20
        self.parameters["print_level"]           = 1
        
        self.sum_q = 0.
        self.sum_q2 = 0.
        
    def run(self, m0, qoi=None, tracer = None):
        if qoi is None:
            qoi = NullQoi()
        if tracer is None:
            tracer = NullTracer()
        number_of_samples = self.parameters["number_of_samples"]
        burn_in = self.parameters["burn_in"]
        
        current = SampleStruct(self.kernel)
        proposed = SampleStruct(self.kernel)
        
        current.m.zero()
        current.m.axpy(1., m0)
        self.kernel.init_sample(current)
        
        if self.parameters["print_level"] > 0:
            print "Burn {0} samples".format(burn_in)
        sample_count = 0
        naccept = 0
        n_check = burn_in // self.parameters["print_progress"]
        while (sample_count < burn_in):
            naccept +=self.kernel.sample(current, proposed)
            sample_count += 1
            if sample_count % n_check == 0 and self.parameters["print_level"] > 0:
                print "{0:2.1f} % completed, Acceptance ratio {1:2.1f} %".format(float(sample_count)/float(burn_in)*100,
                                                                         float(naccept)/float(sample_count)*100 )
        if self.parameters["print_level"] > 0:
            print "Generate {0} samples".format(number_of_samples)
        sample_count = 0
        naccept = 0
        n_check = number_of_samples // self.parameters["print_progress"]
        while (sample_count < number_of_samples):
            naccept +=self.kernel.sample(current, proposed)
            q = qoi.eval([current.u, current.m])
            self.sum_q += q
            self.sum_q2 += q*q
            tracer.append(current, q)
            sample_count += 1
            if sample_count % n_check == 0 and self.parameters["print_level"] > 0:
                print "{0:2.1f} % completed, Acceptance ratio {1:2.1f} %".format(float(sample_count)/float(number_of_samples)*100,
                                                                         float(naccept)/float(sample_count)*100 )        
        return naccept
    
    def consume_random(self):
        number_of_samples = self.parameters["number_of_samples"]
        burn_in = self.parameters["burn_in"]
        
        for ii in xrange(number_of_samples+burn_in):
            self.kernel.consume_random()


class MALAKernel:
    def __init__(self, model):
        self.model = model
        self.pr_mean = model.prior.mean
        self.parameters = {}
        self.parameters["inner_rel_tolerance"]   = 1e-9
        self.parameters["delta_t"]               = 0.25*1e-4
        
        self.noise = dl.Vector()
        self.model.prior.init_vector(self.noise, "noise")
        
    def name(self):
        return "inf-MALA"
        
    def derivativeInfo(self):
        return 1
    
    def init_sample(self, s):
        inner_tol = self.parameters["inner_rel_tolerance"]
        self.model.solveFwd(s.u, [s.u,s.m,s.p], inner_tol)
        s.cost = self.model.cost([s.u,s.m,s.p])[2]
        self.model.solveAdj(s.p, [s.u,s.m,s.p], inner_tol)
        self.model.evalGradientParameter([s.u,s.m,s.p], s.g, misfit_only=True)
        self.model.prior.Rsolver.solve(s.Cg, s.g)
        
    def sample(self, current, proposed): 
        proposed.m = self.proposal(current)
        self.init_sample(proposed)
        rho_mp = self.acceptance_ratio(current, proposed)
        rho_pm = self.acceptance_ratio(proposed, current)
        al = rho_mp - rho_pm
        if(al > math.log(np.random.rand())):
            current.assign(proposed)
            return 1
        else:
            return 0

    def proposal(self, current):
        delta_t = self.parameters["delta_t"]
        parRandom.normal(1., self.noise)
        w = dl.Vector()
        self.model.prior.init_vector(w, 0)
        self.model.prior.sample(self.noise,w, add_mean=False)
        delta_tp2 = 2 + delta_t
        d_gam = self.pr_mean + (2-delta_t)/(2+delta_t) * (current.m -self.pr_mean) - (2*delta_t)/(delta_tp2)*current.Cg + math.sqrt(8*delta_t)/delta_tp2 * w
        return d_gam

    def acceptance_ratio(self, origin, destination):
        delta_t = self.parameters["delta_t"]
        m_m = destination.m - origin.m
        p_m = destination.m + origin.m - 2.*self.pr_mean
        temp = origin.Cg.inner(origin.g)
        rho_uv = origin.cost + 0.5*origin.g.inner(m_m) + \
                0.25*delta_t*origin.g.inner(p_m) + \
                0.25*delta_t*temp
        return rho_uv
    
    def consume_random(self):
        parRandom.normal(1., self.noise)
        np.random.rand()
        
        

class pCNKernel:
    def __init__(self, model):
        self.model = model
        self.parameters = {}
        self.parameters["inner_rel_tolerance"]   = 1e-9
        self.parameters["s"]                     = 0.1
        
        self.noise = dl.Vector()
        self.model.prior.init_vector(self.noise, "noise")
        
    def name(self):
        return "pCN"

    def derivativeInfo(self):
        return 0

    def init_sample(self, current):
        inner_tol = self.parameters["inner_rel_tolerance"]
        self.model.solveFwd(current.u, [current.u,current.m,None], inner_tol)
        current.cost = self.model.cost([current.u,current.m,None])[2]
        
    def sample(self, current, proposed): 
        proposed.m = self.proposal(current)
        self.init_sample(proposed)
        al = -proposed.cost + current.cost
        if(al > math.log(np.random.rand())):
            current.assign(proposed)
            return 1
        else:
            return 0

    def proposal(self, current):
        #Generate sample from the prior
        parRandom.normal(1., self.noise)
        w = dl.Vector()
        self.model.prior.init_vector(w, 0)
        self.model.prior.sample(self.noise,w, add_mean=False)
        # do pCN linear combination with current sample
        s = self.parameters["s"]
        w *= s
        w.axpy(1., self.model.prior.mean)
        w.axpy(np.sqrt(1. - s*s), current.m - self.model.prior.mean)
        
        return w
    
    def consume_random(self):
        parRandom.normal(1., self.noise)
        np.random.rand() 
    
class gpCNKernel:
    """
    F. J. PINSKI, G. SIMPOSN, A. STUART, H. WEBER
    Algorithms for Kullback-Leibler Approximation of Probability Measures in Infinite Dimensions
    http://arxiv.org/pdf/1408.1920v1.pdf
    Alg. 5.2
    """
    def __init__(self, model, nu):
        self.model = model
        self.nu = nu
        self.prior = model.prior
        self.Hm = LowRankHessianMisfit(self.prior, nu.d, nu.U)
        self.parameters = {}
        self.parameters["inner_rel_tolerance"]   = 1e-9
        self.parameters["s"]                     = 0.1
        
        self.noise = dl.Vector()
        self.nu.init_vector(self.noise, "noise")
        
    def name(self):
        return "gpCN"

    def derivativeInfo(self):
        return 0

    def init_sample(self, current):
        inner_tol = self.parameters["inner_rel_tolerance"]
        self.model.solveFwd(current.u, [current.u,current.m,None], inner_tol)
        current.cost = self.model.cost([current.u,current.m,None])[2]
        
    def sample(self, current, proposed): 
        proposed.m = self.proposal(current)
        self.init_sample(proposed)
        al = self.delta(current) - self.delta(proposed)
        if(al > math.log(np.random.rand())):
            current.assign(proposed)
            return 1
        else:
            return 0
        
    def delta(self,sample):
        dm = sample.m - self.nu.mean
        d_mean = self.nu.prior.mean - self.nu.mean
        phi_mu = sample.cost
        phi_nu = - self.prior.R.inner(dm, d_mean) + .5*self.Hm.inner(dm, dm)
        return phi_mu - phi_nu
        

    def proposal(self, current):
        #Generate sample from the prior
        parRandom.normal(1., self.noise)
        w_prior = dl.Vector()
        self.nu.init_vector(w_prior, 0)
        w = dl.Vector()
        self.nu.init_vector(w, 0)
        self.nu.sample(self.noise, w_prior, w, add_mean=False)
        # do pCN linear combination with current sample
        s = self.parameters["s"]
        w *= s
        w.axpy(1., self.nu.mean)
        w.axpy(np.sqrt(1. - s*s), current.m - self.nu.mean)
        
        return w
    
    def consume_random(self):
        parRandom.normal(1., self.noise)
        np.random.rand() 
    
    
class ISKernel:
    def __init__(self, model, nu):
        self.model = model
        self.nu = nu
        self.prior = model.prior
        self.parameters = {}
        self.parameters["inner_rel_tolerance"]   = 1e-9
        
        self.noise = dl.Vector()
        self.nu.init_vector(self.noise, "noise")
        
    def name(self):
        return "IS"

    def derivativeInfo(self):
        return 0

    def init_sample(self, current):
        inner_tol = self.parameters["inner_rel_tolerance"]
        self.model.solveFwd(current.u, [current.u,current.m,None], inner_tol)
        current.cost = self.model.cost([current.u,current.m,None])[2]
        
    def sample(self, current, proposed): 
        proposed.m = self.proposal(current)
        self.init_sample(proposed)
        al = self.delta(current) - self.delta(proposed)
        if(al > math.log(np.random.rand())):
            current.assign(proposed)
            return 1
        else:
            return 0
        
    def delta(self,sample):
        dm_nu = sample.m - self.nu.mean
        dm_pr = sample.m - self.prior.mean

        return sample.cost + .5*self.prior.R.inner(dm_pr, dm_pr) - .5*self.nu.Hlr.inner(dm_nu, dm_nu)
        

    def proposal(self, current):
        #Generate sample from the prior
        parRandom.normal(1., self.noise)
        w_prior = dl.Vector()
        self.nu.init_vector(w_prior, 0)
        w = dl.Vector()
        self.nu.init_vector(w, 0)
        self.nu.sample(self.noise, w_prior, w, add_mean=True)
        
        return w
    
    def consume_random(self):
        parRandom.normal(1., self.noise)
        np.random.rand() 

class SNmapKernel:
    """
    Stochastic Newton with MAP Hessian
    """
    def __init__(self, model, nu):
        """
        - model: an object of type Model
        - nu:    an object of type GaussianLRPosterior 
        """
        self.model = model
        self.nu = nu
        self.parameters = {}
        self.parameters["inner_rel_tolerance"]   = 1e-9
        
        self.noise = dl.Vector()
        self.model.prior.init_vector(self.noise, "noise")
        
        self.discard = self.model.generate_vector(PARAMETER)
        self.w       = self.model.generate_vector(PARAMETER)
        
    def name(self):
        return "StochasticNewton_MAP"
        
    def derivativeInfo(self):
        return 1

    def init_sample(self, s):
        inner_tol = self.parameters["inner_rel_tolerance"]
        self.model.solveFwd(s.u, [s.u,s.m,s.p], inner_tol)
        s.cost = self.model.cost([s.u,s.m,s.p])[0]
        self.model.solveAdj(s.p, [s.u,s.m,s.p], inner_tol)
        self.model.evalGradientParameter([s.u,s.m,s.p], s.g, misfit_only=False) 
        self.nu.Hlr.solve(s.Cg, s.g)
        
    def sample(self, current, proposed): 
        proposed.m = self.proposal(current)
        self.init_sample(proposed)
        c2p = self.neg_log_rho(current, proposed)
        p2c = self.neg_log_rho(proposed, current)
        al = c2p - p2c
        if(al > math.log(np.random.rand())):
            current.assign(proposed)
            return 1
        else:
            return 0

    def proposal(self, current):
        parRandom.normal(1., self.noise)
        self.nu.sample(self.noise, self.discard, self.w, add_mean=False)
        return current.m - current.Cg + self.w

    def neg_log_rho(self, origin, destination):
        w = destination.m - origin.m + origin.Cg
        return origin.cost + 0.5*self.nu.Hlr.inner(w,w)
    
    def consume_random(self):
        parRandom.normal(1., self.noise)
        np.random.rand()







'''
Created on Jun 21, 2017

@author: uvilla
'''
import dolfin as dl
import numpy as np
from random import parRandom
from variables import STATE, PARAMETER
from multivector import MultiVector
from randomizedEigensolver import doublePassG
from NewtonCG import ReducedSpaceNewtonCG, ReducedSpaceNewtonCG_ParameterList
from reducedHessian import ReducedHessian
from posterior import GaussianLRPosterior

def expectedInformationGainLaplace(model, ns, k, save_any=10, fname="kldist"):
    rank = dl.MPI.rank(dl.mpi_comm_world())
    m_MC = model.generate_vector(PARAMETER)
    u_MC = model.generate_vector(STATE)
    noise_m = dl.Vector()
    model.prior.init_vector(noise_m, "noise")
    noise_obs = dl.Vector()
    model.misfit.B.init_vector(noise_obs, 0)
    
    out = np.zeros( (ns,5) )
    header='kldist, c_misfit, c_reg, c_logdet, c_tr'
    
    p = 20
    
    Omega = MultiVector(m_MC, k+p)
    parRandom.normal(1., Omega)
    
    for iMC in np.arange(ns):
        if rank == 0:
            print "Sample: ", iMC
        parRandom.normal(1., noise_m)
        parRandom.normal(1., noise_obs)
        model.prior.sample(noise_m, m_MC)
        model.solveFwd(u_MC, [u_MC, m_MC, None], 1e-9)
        model.misfit.B.mult(u_MC, model.misfit.d)
        model.misfit.d.axpy(np.sqrt(model.misfit.noise_variance), noise_obs)
    
        a = m_MC.copy()
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
    
        if rank == 0:
            if solver.converged:
                print "\nConverged in ", solver.it, " iterations."
            else:
                print "\nNot Converged"

            print "Termination reason: ", solver.termination_reasons[solver.reason]
            print "Final gradient norm: ", solver.final_grad_norm
            print "Final cost: ", solver.final_cost
            
        model.setPointForHessianEvaluations(x, gauss_newton_approx=False)
        Hmisfit = ReducedHessian(model, solver.parameters["inner_rel_tolerance"],  misfit_only=True)
        d, U = doublePassG(Hmisfit, model.prior.R, model.prior.Rsolver, Omega, k, s=1, check=False)
        posterior = GaussianLRPosterior(model.prior, d, U)
        posterior.mean = x[PARAMETER]
    
        kl_dist, c_detlog, c_tr, cr = posterior.klDistanceFromPrior(sub_comp=True)
        if rank == 0:
            print "KL-Distance from prior: ", kl_dist
        
        cm = model.misfit.cost(x)
        out[iMC, 0] = kl_dist
        out[iMC, 1] = cm
        out[iMC, 2] = cr
        out[iMC, 3] = c_detlog
        out[iMC, 4] = c_tr
        
        if (rank == 0) and (iMC%save_any == save_any-1):
            all_kl = out[0:iMC+1, 0]
            print "I = ", np.mean(all_kl), " Var[I_MC] = ", np.var(all_kl, ddof=1)/float(iMC+1)
            if fname is not None:
                np.savetxt(fname+'_tmp.txt', out[0:iMC+1, :], header=header, comments='% ')
    
    if fname is not None:
        np.savetxt(fname+'.txt', out, header=header, comments='% ')
    
    return np.mean(out[:,0])


def expectedInformationGainMC2(model, n, fname="loglikeEv"):
    rank = dl.MPI.rank(dl.mpi_comm_world())
    # STEP 1: Generate random m and evaluate/store B*u(m)
    u = model.generate_vector(STATE)
    m = model.generate_vector(PARAMETER)
    all_p2o = [dl.Vector() for i in range(n)]
    [model.misfit.B.init_vector(p2o, 0) for p2o in all_p2o]
    noise_m   = dl.Vector()
    model.prior.init_vector(noise_m, "noise")
    
    for i in range(n):
        if rank == 0 and i%10 == 0:
            print "Compute observation", i
        parRandom.normal(1., noise_m)
        model.prior.sample(noise_m, m)
        model.solveFwd(u, [u, m, None])
        model.misfit.B.mult(u, all_p2o[i])
        
    
    # STEP 2: Compute Evidence and likelihood
    noise_obs = dl.Vector()
    model.misfit.B.init_vector(noise_obs,0)
    obs = dl.Vector()
    model.misfit.B.init_vector(obs,0)
    diff = dl.Vector()
    model.misfit.B.init_vector(diff,0)
    
    gamma_noise = model.misfit.noise_variance
    log_evidence = np.zeros(n)
    log_like = np.zeros(n)
    
    nobs = float( noise_obs.size() )
    
    for i in range(n):
        if rank == 0 and i%10 == 0:
            print "Evaluate evidence", i
        parRandom.normal(1., noise_obs)
        #Note: the normalizing costant 1/sqrt( det( 2\pi \Gamma_noise ) ) cancels out between the log_like and log_evidence
        log_like[i] = -0.5*noise_obs.inner(noise_obs) #- 0.5*np.log(2.*np.pi*gamma_noise)*nobs
        obs.zero()
        obs.axpy(1., all_p2o[i])
        obs.axpy(np.sqrt(gamma_noise), noise_obs)
        tmp = 0.0
        for j in range(n):
            diff.zero()
            diff.axpy(1., obs)
            diff.axpy(-1., all_p2o[j])
            tmp += np.exp(-0.5*diff.inner(diff)/gamma_noise)
            
        log_evidence[i] = np.log(tmp)-np.log(n) #- 0.5*np.log(2.*np.pi*gamma_noise)*nobs 
    
    if fname is not None:
        out = np.zeros( (n, 2) )
        out[:,0] = log_like
        out[:,1] = log_evidence
        np.savetxt(fname+'.txt', out, header='log_like, log_ev', comments='% ') 

    return np.mean(log_like) - np.mean(log_evidence)
        
            
        
# README #

We are interested in simulating heat diffusion in a medium with random inclusion.
Specifically, we need the capability of
1. Generate realization of medium with random inclusion
2. Solve the forward problem (heat equation) for a given realization of the medium and generate synthetic observation
3. Solve the inverse problem and compute the MAP point estimator of the field given noisy observations
4. Perform linearized (Laplace approximation) or full (MCMC) Bayesian inference
5. Multiscale/model adaptivity

We will use [hIPPYlib](hippylib.github.io) to address tasks 1 through 4.
hIPPYlib depends on [FEniCS](fenicsproject.org) for the finite element discretization and [PETSc](https://www.mcs.anl.gov/petsc/) for the linear algebra

### How do I get set up? ###

* Build FEniCS:
  * On the ICES desktops, FEniCS can be loaded using the module system `module load c7 fenics-m/2016.1`
  * On a MacOS laptop or other machines, use the fenics-install script provided by hippylib
  
* If you haven't done it, setup your SSH key on bitbucket; see [here](https://confluence.atlassian.com/bitbucket/set-up-an-ssh-key-728138079.html)
  
* Clone `hIPPYlib`: `git clone git@bitbucket.org:hippylibdev/hippylib.git` and switch to the `random_refactor-dev` branch.

* Clone this repository in the same folder where `hIPPYlib` was cloned. `git clone git@bitbucket.org:hippylibdev/multifidelity.git`

* Enjoy!

### Who do I talk to? ###

* U. Villa; uvilla@ices.utexas.edu
* D. Faghihi, danial@ices.utexas.edu
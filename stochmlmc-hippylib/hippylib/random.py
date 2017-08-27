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
import os
from multivector import MultiVector
from timeDependentVector import TimeDependentVector

abspath = os.path.dirname( os.path.abspath(__file__) )
source_directory = os.path.join(abspath,"cpp_rand")
header_file = open(os.path.join(source_directory,"PRNG.h"), "r")
code = header_file.read()
header_file.close()
cpp_sources = ["PRNG.cpp"]  

include_dirs = [".", source_directory]
for ss in ['PROFILE_INSTALL_DIR', 'PETSC_DIR', 'SLEPC_DIR']:
    if os.environ.has_key(ss):
        include_dirs.append(os.environ[ss]+'/include')
        
cpp_module = dl.compile_extension_module(
             code=code, source_directory=source_directory,
             sources=cpp_sources, include_dirs=include_dirs)


class Random(cpp_module.Random):
    """
    This class handle parallel generation of random numbers in 
    """
    def __init__(self,myid=0, nproc=1, blocksize=1000000, seed=1):
        """
        Create a parallel random number number generator.
        INPUTS:
        - myid: id of the calling process
        - nproc: number of processor in the communicator
        - blocksize: number of consecutive random number to be generated before jumping headed in the stream
        - seed: random seed to initialize the random engine
        """
        super(Random, self).__init__(seed)
        self.split(myid, nproc, blocksize)
        
    def uniform(self,a,b, out=None):
        """
        Sample from uniform distribution
        """
        if out is None:
            return super(Random, self).uniform(a,b)
        elif type(out) is MultiVector:
            for i in range(out.nvec()):
                super(Random, self).uniform(out[i], a,b)
            return None
        elif type(out) is TimeDependentVector:
            for i in range(out.nsteps):
                super(Random, self).uniform(out.data[i], a,b)
            return None
        elif type( dl.as_backend_type(out) ) is dl.PETScVector:
            super(Random, self).uniform(out[i], a,b)
            return None
    
    def normal(self, sigma, out=None):
        """
        Sample from uniform distribution with given variance
        """
        if out is None:
            return super(Random, self).normal(0, sigma)
        elif type(out) is MultiVector:
            for i in range(out.nvec()):
                super(Random, self).normal(out[i], sigma, True)
            return None
        elif type(out) is TimeDependentVector:
            for i in range(out.nsteps):
                super(Random, self).normal(out.data[i], sigma, True)
            return None
        elif type( dl.as_backend_type(out) ) is dl.PETScVector:
            super(Random, self).normal(out, sigma, True)
            
    def normal_perturb(self,sigma, out):
        """
        Add a normal perturbation to a Vector/MultiVector
        """
        if type(out) is MultiVector:
            for i in range(out.nvec()):
                super(Random, self).normal(out[i], sigma, False)
        elif type(out) is TimeDependentVector:
            for i in range(out.nsteps):
                super(Random, self).normal(out.data[i], sigma, False)
            return None
        elif type( dl.as_backend_type(out) ) is dl.PETScVector:
            super(Random, self).normal(out, sigma, False)
            
    def rademacher(self, out=None):
        """
        Sample from rademacher distribution
        """
        if out is None:
            return super(Random, self).rademacher()
        elif type(out) is MultiVector:
            for i in range(out.nvec()):
                super(Random, self).rademacher(out[i])
            return None
        elif type( dl.as_backend_type(out) ) is dl.PETScVector:
            super(Random, self).rademacher(out)
            
_world_rank = dl.MPI.rank(dl.mpi_comm_world())
_world_size = dl.MPI.rank(dl.mpi_comm_world())

parRandom = Random(_world_rank, _world_size)
            


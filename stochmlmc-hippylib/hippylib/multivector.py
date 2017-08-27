from dolfin import compile_extension_module, DoubleArray, File
from vector2function import vector2Function
import numpy as np
import os

abspath = os.path.dirname( os.path.abspath(__file__) )
source_directory = os.path.join(abspath,"cpp_multivector")
header_file = open(os.path.join(source_directory,"multivector.h"), "r")
code = header_file.read()
header_file.close()
cpp_sources = ["multivector.cpp"]  

include_dirs = [".", source_directory]
for ss in ['PROFILE_INSTALL_DIR', 'PETSC_DIR', 'SLEPC_DIR']:
    if os.environ.has_key(ss):
        include_dirs.append(os.environ[ss]+'/include')
        
cpp_module = compile_extension_module(
                code=code, source_directory=source_directory,
                sources=cpp_sources, include_dirs=include_dirs)

class MultiVector(cpp_module.MultiVector):
    def dot_v(self, v):
        m = DoubleArray(self.nvec())
        self.dot(v, m)
        return np.zeros(self.nvec()) + m.array()
    
    def dot_mv(self,mv):
        shape = (self.nvec(),mv.nvec())
        m = DoubleArray(shape[0]*shape[1])
        self.dot(mv, m)
        return np.zeros(shape) + m.array().reshape(shape, order='C')
    
    def norm(self, norm_type):
        shape = self.nvec()
        m = DoubleArray(shape)
        self.norm_all(norm_type, m)
        return np.zeros(shape) + m.array()
    
    def Borthogonalize(self,B):
        """ 
        Returns QR decomposition of self.
        Q and R satisfy the following relations in exact arithmetic
        1. QR        = Z
        2. Q^*BQ     = I
        3. Q^*BZ    = R 
        4. ZR^{-1}    = Q
        
        Returns
        Bq : MultiVector: The B^{-1}-orthogonal vectors
        r : ndarray: The r of the QR decomposition
        Note: self is overwritten by Q    
        """
        return self._mgs_stable(B)
    
    def orthogonalize(self):
        """ 
        Returns QR decomposition of self.
        Q and R satisfy the following relations in exact arithmetic
        1. QR        = Z
        2. Q^*Q     = I
        3. Q^*Z    = R 
        4. ZR^{-1}  = Q
        
        Returns
        r : ndarray: The r of the QR decomposition
        Note: self is overwritten by Q    
        """
        return self._mgs_reortho()
    
    def _mgs_stable(self, B):
        """ 
        Returns QR decomposition of self, which satisfies conditions 1--4

        Uses Modified Gram-Schmidt with re-orthogonalization (Rutishauser variant)
        for computing the B-orthogonal QR factorization
        
        References
        ----------
        .. [1] A.K. Saibaba, J. Lee and P.K. Kitanidis, Randomized algorithms for Generalized
               Hermitian Eigenvalue Problems with application to computing 
               Karhunen-Loe've expansion http://arxiv.org/abs/1307.6885
               
        .. [2] W. Gander, Algorithms for the QR decomposition. Res. Rep, 80(02), 1980
        
        https://github.com/arvindks/kle
        
        """
        n = self.nvec()
        Bq = MultiVector(self[0], n)
        r  = np.zeros((n,n), dtype = 'd')
        reorth = np.zeros((n,), dtype = 'd')
        eps = np.finfo(np.float64).eps
        
        for k in np.arange(n):
            B.mult(self[k], Bq[k])
            t = np.sqrt( Bq[k].inner(self[k]))
            
            nach = 1;    u = 0;
            while nach:
                u += 1
                for i in np.arange(k):
                    s = Bq[i].inner(self[k])
                    r[i,k] += s
                    self[k].axpy(-s, self[i])
                    
                B.mult(self[k], Bq[k])
                tt = np.sqrt(Bq[k].inner(self[k]))
                if tt > t*10.*eps and tt < t/10.:
                    nach = 1;    t = tt;
                else:
                    nach = 0;
                    if tt < 10.*eps*t:
                        tt = 0.
            

            reorth[k] = u
            r[k,k] = tt
            if np.abs(tt*eps) > 0.:
                tt = 1./tt
            else:
                tt = 0.
                
            self.scale(k, tt)
            Bq.scale(k, tt)
            
        return Bq, r 
    
    def _mgs_reortho(self):
        n = self.nvec()
        r  = np.zeros((n,n), dtype = 'd')
        reorth = np.zeros((n,), dtype = 'd')
        eps = np.finfo(np.float64).eps
        
        for k in np.arange(n):
            t = np.sqrt( self[k].inner(self[k]))
            
            nach = 1;    u = 0;
            while nach:
                u += 1
                for i in np.arange(k):
                    s = self[i].inner(self[k])
                    r[i,k] += s
                    self[k].axpy(-s, self[i])
                    
                tt = np.sqrt(self[k].inner(self[k]))
                if tt > t*10.*eps and tt < t/10.:
                    nach = 1;    t = tt;
                else:
                    nach = 0;
                    if tt < 10.*eps*t:
                        tt = 0.
            

            reorth[k] = u
            r[k,k] = tt
            if np.abs(tt*eps) > 0.:
                tt = 1./tt
            else:
                tt = 0.
                
            self.scale(k, tt)
            
        return r
    
    def export(self, Vh, filename, varname = "mv", normalize=False):
        """
        Export in paraview this multivector
        Inputs:
        - Vh:        the parameter finite element space
        - filename:  the name of the paraview output file
        - varname:   the name of the paraview variable
        - normalize: if True the vector are rescaled such that || u ||_inf = 1 
        """
        fid = File(filename)
        if not normalize:
            for i in range(self.nvec()):
                fun = vector2Function(self[i], Vh, name = varname)
                fid << fun
        else:
            tmp = self[0].copy()
            for i in range(self.nvec()):
                s = self[i].norm("linf")
                tmp.zero()
                tmp.axpy(1./s, self[i])
                fun = vector2Function(tmp, Vh, name = varname)
                fid << fun
            
    
def MatMvMult(A, x, y):
    assert x.nvec() == y.nvec(), "x and y have non-matching number of vectors"
    for i in range(x.nvec()):
        A.mult(x[i], y[i])
        
def MvDSmatMult(X, A, Y):
    assert X.nvec() == A.shape[0], "X Number of vecs incompatible with number of rows in A"
    assert Y.nvec() == A.shape[1], "Y Number of vecs incompatible with number of cols in A"
    for j in range(Y.nvec()):
        Y[j].zero()
        X.reduce(Y[j], A[:,j].flatten())

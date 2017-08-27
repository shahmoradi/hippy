import math
import dolfin as dl
import sys
sys.path.append( "../../" )
from hippylib import *
import numpy as np

try:
    import matplotlib.pyplot as plt
    has_plt = True
except:
    has_plt = False

class GammaCenter(dl.SubDomain):
    def inside(self, x, on_boundary):
        return ( abs(x[1]-.5) < dl.DOLFIN_EPS )
    
class FluxQOI(QOI):
    def __init__(self, Vh, dsGamma):
        self.Vh = Vh
        self.dsGamma = dsGamma
        self.n = dl.Constant((0.,1.))#dl.FacetNormal(Vh[STATE].mesh())
        
        self.u = None
        self.m = None
        self.L = {}
        
    def form(self, x):
        return dl.avg(dl.exp(x[PARAMETER])*dl.dot( dl.grad(x[STATE]), self.n) )*self.dsGamma
    
    def eval(self, x):
        """
        Given x evaluate the cost functional.
        Only the state u and (possibly) the parameter a are accessed.
        """
        u = vector2Function(x[STATE], self.Vh[STATE])
        m = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        return dl.assemble(self.form([u,m]))
    
    def grad(self, i, x, g):
        if i == STATE:
            self.grad_state(x, g)
        elif i==PARAMETER:
            self.grad_param(x, g)
        else:
            raise i
                
    def grad_state(self,x,g):
        """Evaluate the gradient with respect to the state.
        Only the state u and (possibly) the parameter m are accessed. """
        u = vector2Function(x[STATE], self.Vh[STATE])
        m = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        form = self.form([u,m])
        g.zero()
        dl.assemble(dl.derivative(form, u), tensor=g)
        
    def grad_param(self,x,g):
        """Evaluate the gradient with respect to the state.
        Only the state u and (possibly) the parameter m are accessed. """
        u = vector2Function(x[STATE], self.Vh[STATE])
        m = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        form = self.form([u,m])
        g.zero()
        dl.assemble(dl.derivative(form, m), tensor=g)
                
    def apply_ij(self,i,j, dir, out):
        """Apply the second variation \delta_ij (i,j = STATE,PARAMETER) of the cost in direction dir."""
        self.L[i,j].mult(dir, out)

    def setLinearizationPoint(self, x):
        self.u = vector2Function(x[STATE], self.Vh[STATE])
        self.m = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        x = [self.u,self.m]
        form = self.form(x)
        for i in range(2):
            di_form = dl.derivative(form, x[i])
            for j in range(2):
                dij_form = dl.derivative(di_form,x[j] )
                self.L[i,j] = dl.assemble(dij_form)


def u_boundary(x, on_boundary):
    return on_boundary and ( x[1] < dl.DOLFIN_EPS or x[1] > 1.0 - dl.DOLFIN_EPS)

def v_boundary(x, on_boundary):
    return on_boundary and ( x[0] < dl.DOLFIN_EPS or x[0] > 1.0 - dl.DOLFIN_EPS)

def true_model(Vh, gamma, delta, anis_diff):
    prior = BiLaplacianPrior(Vh, gamma, delta, anis_diff )
    noise = dl.Vector()
    prior.init_vector(noise,"noise")
    parRandom.normal(1., noise)
    atrue = dl.Vector()
    prior.init_vector(atrue, 0)
    prior.sample(noise,atrue)
    return atrue
            
if __name__ == "__main__":
    dl.set_log_active(False)
    sep = "\n"+"#"*80+"\n"
    ndim = 2
    nx = 64
    ny = 64
    mesh = dl.UnitSquareMesh(nx, ny)
    
    rank = dl.MPI.rank(mesh.mpi_comm())
    nproc = dl.MPI.size(mesh.mpi_comm())
        
    Vh2 = dl.FunctionSpace(mesh, 'Lagrange', 2)
    Vh1 = dl.FunctionSpace(mesh, 'Lagrange', 1)
    Vh = [Vh2, Vh1, Vh2]
    ndofs = [Vh[STATE].dim(), Vh[PARAMETER].dim(), Vh[ADJOINT].dim()]
    if rank == 0:
        print sep, "Set up the mesh and finite element spaces", sep
        print "Number of dofs: STATE={0}, PARAMETER={1}, ADJOINT={2}".format(*ndofs)
    
    # Initialize Expressions
    f = dl.Constant(0.0)
        
    u_bdr = dl.Expression("x[1]", degree=1)
    u_bdr0 = dl.Constant(0.0)
    bc = dl.DirichletBC(Vh[STATE], u_bdr, u_boundary)
    bc0 = dl.DirichletBC(Vh[STATE], u_bdr0, u_boundary)
    
    def pde_varf(u,a,p):
        return dl.exp(a)*dl.inner(dl.nabla_grad(u), dl.nabla_grad(p))*dl.dx - f*p*dl.dx
    
    pde = PDEVariationalProblem(Vh, pde_varf, bc, bc0, is_fwd_linear=True)
    if dlversion() <= (1,6,0):
        pde.solver = dl.PETScKrylovSolver("cg", amg_method())
        pde.solver_fwd_inc = dl.PETScKrylovSolver("cg", amg_method())
        pde.solver_adj_inc = dl.PETScKrylovSolver("cg", amg_method())
    else:
        pde.solver = dl.PETScKrylovSolver(mesh.mpi_comm(), "cg", amg_method())
        pde.solver_fwd_inc = dl.PETScKrylovSolver(mesh.mpi_comm(), "cg", amg_method())
        pde.solver_adj_inc = dl.PETScKrylovSolver(mesh.mpi_comm(), "cg", amg_method())
    pde.solver.parameters["relative_tolerance"] = 1e-15
    pde.solver.parameters["absolute_tolerance"] = 1e-20
    pde.solver_fwd_inc.parameters = pde.solver.parameters
    pde.solver_adj_inc.parameters = pde.solver.parameters
     
    gamma = .1
    delta = .5
    
    anis_diff = dl.Expression(code_AnisTensor2D, degree=1)
    anis_diff.theta0 = 2.
    anis_diff.theta1 = .5
    anis_diff.alpha = math.pi/4
    atrue = true_model(Vh[PARAMETER], gamma, delta,anis_diff)

    locations = np.array([[0.1, 0.1], [0.1, 0.9], [.5,.5], [.9, .1], [.9, .9]])

    pen = 1e1
    prior = MollifiedBiLaplacianPrior(Vh[PARAMETER], gamma, delta, locations, atrue, anis_diff, pen)
    
    if rank == 0:    
        print "Prior regularization: (delta_x - gamma*Laplacian)^order: delta={0}, gamma={1}, order={2}".format(delta, gamma,2)
        
    GC = GammaCenter()
    marker = dl.FacetFunction("size_t", mesh)
    marker.set_all(0)
    GC.mark(marker, 1)
    dss = dl.Measure("dS", domain=mesh, subdomain_data=marker)
    qoi = FluxQOI(Vh,dss(1)) 
    rqoi = ReducedQOI(pde, qoi)
    
    if 1:
        reducedQOIVerify(rqoi, prior.mean, eps=np.power(.5, np.arange(20,0,-1)), plotting = True )
    
    k = 100    
    Omega = MultiVector(prior.mean, k)
    parRandom.normal(1., Omega)
    
    q_taylor = TaylorApproximationQOI(rqoi, prior)
    q_taylor.computeLowRankFactorization(Omega, innerTol=1e-9)
    
    if rank == 0:
        plotEigenvalues(q_taylor.d)
    
    e_lin  = q_taylor.expectedValue(order=1)
    e_quad = q_taylor.expectedValue(order=2)
    v_lin  = q_taylor.variance(order=1)
    v_quad = q_taylor.variance(order=1)
    if rank == 0:
        print "E[Q_lin] = {0:7e}, E[Q_quad] = {1:7e}".format(e_lin, e_quad)
        print "Var[Q_lin] = {0:7e}, Var[Q_quad] = {1:7e}".format(v_lin, v_quad)
    
    varianceReductionMC(prior, rqoi, q_taylor,  nsamples=100)
    
    if rank == 0 and has_plt:
        plt.show()
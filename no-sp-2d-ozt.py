# 2d orsag-tang
# no structure preserving scheme
# BE scheme
from firedrake import *
from firedrake.petsc import PETSc
print = PETSc.Sys.Print
import csv
import numpy as np
from mpi4py import MPI
# Define two-dimensional versions of cross and curl operators
def scross(x, y):
    return x[0]*y[1] - x[1]*y[0]


def vcross(x, y):
    return as_vector([x[1]*y, -x[0]*y])


def scurl(x):
    return x[1].dx(0) - x[0].dx(1)


def vcurl(x):
    return as_vector([x.dx(1), -x.dx(0)])


def acurl(x):
    return as_vector([
                     x[2].dx(1),
                     -x[2].dx(0),
                     x[1].dx(0) - x[0].dx(1)
                     ])

baseN = 100
dp={"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 1)}

mesh = PeriodicUnitSquareMesh(baseN, baseN, distribution_parameters = dp)
mesh.coordinates.dat.data[:] *= 2 * pi

(x, y) = SpatialCoordinate(mesh)
# mesh.coordinates.dat.data[:, 0] -= L/2
# mesh.coordinates.dat.data[:, 1] -= L/2
# mesh.coordinates.dat.data[:, 2] -= L/2

Vg = VectorFunctionSpace(mesh, "CG", 2)
Vn = FunctionSpace(mesh, "DG", 0)
Vg_ = FunctionSpace(mesh, "CG", 1)
Vd = FunctionSpace(mesh, "RT", 1)
Vc = FunctionSpace(mesh, "N1curl", 1)

#u, p, B, E
Z = MixedFunctionSpace([Vg, Vn, Vd, Vg_])
z = Function(Z)
(u, p, B, E) = split(z)
(ut, pt, Bt, Et) = split(TestFunction(Z))

z_prev = Function(Z)
(up, pp, Bp, Ep) = split(z_prev)


c = Constant(1)
nu = Constant(1e-5)
eta = Constant(1e-5)

f = Function(Vg)
f.interpolate(Constant((0, 0)))

t = Constant(0)
dt = Constant(1/50)
T = 5.0

# initial condition
#u1 = -sin(2*pi * y0)
#u2 = sin(2 * pi * x0)
#B1 = -sin(2 * pi *y0) 
#B2 = sin(4 * pi * x0)

def v_grad(x):
    return as_vector([-x.dx(1), x.dx(0)])

# Biskamp-Welter-1989
phi = cos(x + 1.4) + cos(y + 0.5)
psi = cos(2 * x + 2.3) + cos(y + 4.1)

#phi = cos(x) + cos(y)
#psi = 0.5* cos(2 * x) + cos(y)

u_init = v_grad(psi)
B_init = v_grad(phi)
 
z.sub(0).interpolate(u_init)
z.sub(2).interpolate(B_init) 
z_prev.assign(z)

(u_, p_, B_, E_) = z.subfunctions
u_.rename("Velocity")
p_.rename("Pressure")
B_.rename("MagneticField")
E_.rename("ElectricField")

pvd = VTKFile("output/orzag-Tang.pvd")
pvd.write(*z.subfunctions, time=float(t))



# write out the linear form
F = (
    # equation for u
    + inner((u-up)/dt, ut) * dx
    + nu * inner(grad(u), grad(ut)) * dx
    + 1/dt * inner(div(u), div(ut)) * dx
    + inner(dot(grad(up), up), ut) * dx
    - inner(p, div(ut)) * dx
    + c * inner(vcross(Bp, E), ut) * dx
    + c * inner(scross(u, Bp), scross(ut, Bp)) * dx #nonlinear term
    #- inner(f,  ut) * dx
    
    #equation for p, the sign does matter!!!
    - inner(div(u), pt) * dx
    
    # equation for B
    + inner((B-Bp)/dt, Bt) * dx
    + inner(vcurl(E), Bt) * dx 
    
    # equation for E
    + inner(E, Et) * dx
    + inner(scross(u, Bp), Et) * dx # nonlinear term
    - eta * inner(B, vcurl(Et)) * dx
    )
    

lu = {
    "mat_type": "aij",
    "snes_type": "newtonls",
    "snes_monitor": None,
    "ksp_monitor": None,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}



def build_solver(F, u_sol, bcs, solver_parameters, options_prefix=None):
    problem = NonlinearVariationalProblem(F, u_sol, bcs=bcs)
    solver = NonlinearVariationalSolver(problem, solver_parameters=solver_parameters, options_prefix=options_prefix)
    return solver

bcs = None
time_stepper = build_solver(F, z, bcs, lu, options_prefix="primal solver")

def compute_divB(B):
    return norm(div(B), 'L2')

def compute_energy(u, B):
    return assemble(inner(u, u) * dx + c * inner(B, B) * dx)

def compute_helicity_m(B):
    return float(0)

def compute_helicity_c(u, B):
     return assemble(inner(u, B)*dx)

def norm_inf(u):
    with u.dat.vec_ro as u_v:
        u_max = u_v.norm(PETSc.NormType.INFINITY)
    return u_max

def compute_ens(w, j):
    w_max=norm_inf(w)
    j_max=norm_inf(j)
    return w_max, j_max, float(w_max) + float(j_max) 


# Time stepping
data_filename = "data.csv"
fieldnames = ["t", "divB", "energy", "helicity_m", "helicity_c", "ens_total", "w_max", "j_max"]
if mesh.comm.rank == 0:
    with open(data_filename, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

j = Function(Vg_,name="Current")
w = Function(Vg_, name="Vorticity")
j.interpolate(scurl(z.sub(2)))
w.interpolate(scurl(z.sub(0)))
pvd1 = VTKFile("output/current.pvd")
pvd1.write(w, j, time=float(t))

divB = compute_divB(z.sub(2))
energy = compute_energy(z.sub(0), z.sub(2)) # u, B
helicity_m = compute_helicity_m(z.sub(2)) # B
helicity_c = compute_helicity_c(z.sub(0), z.sub(2)) # u, B
w_max, j_max, ens_total = compute_ens(w, j) # w, j

if mesh.comm.rank == 0:
    row = {
        "t": float(t),
        "divB": float(divB),
        "energy": float(energy),
        "helicity_c": float(helicity_c),
        "helicity_m": float(helicity_m),
        "ens_total": float(ens_total),
        "w_max": float(w_max),
        "j_max": float(j_max),
    }
    with open(data_filename, "a", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)


while (float(t) < float(T-dt) + 1.0e-10):
    t.assign(t+dt)    
    if mesh.comm.rank==0:
        print(f"Solving for t = {float(t):.4f} .. ", flush=True)
    # Solve for (u, p)
    time_stepper.solve()
    # Output
    j.interpolate(scurl(z.sub(2)))
    w.interpolate(scurl(z.sub(0)))
    divB = compute_divB(z.sub(2))
    energy = compute_energy(z.sub(0), z.sub(2)) # u, B
    helicity_m = compute_helicity_m(z.sub(2)) # B
    helicity_c = compute_helicity_c(z.sub(0), z.sub(2)) # u, B
    w_max, j_max, ens_total = compute_ens(w, j) # w, j

    if mesh.comm.rank == 0:
        print(f"divB: {divB:.8f}, energy={energy}, helicity_m={helicity_m}, helicity_c={helicity_c}, ens_total={ens_total}")
        row = {
            "t": float(t),
            "divB": float(divB),
            "energy": float(energy),
            "helicity_c": float(helicity_c),
            "helicity_m": float(helicity_m),
            "ens_total": float(ens_total),
            "w_max": float(w_max),
            "j_max": float(j_max),
        }
        with open(data_filename, "a", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row)
            
    pvd.write(*z.subfunctions, time=float(t))
    pvd1.write(w, j, time=float(t))
    z_prev.assign(z)
  

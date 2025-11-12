# 2d orsag-tang

from firedrake import *
from firedrake.petsc import PETSc
print = PETSc.Sys.Print
import csv
import numpy as np
from mpi4py import MPI

baseN = 200
mesh = PeriodicUnitSquareMesh(baseN, baseN)
mesh.coordinates.dat.data[:] *= 2 * pi

(x, y) = SpatialCoordinate(mesh)
# mesh.coordinates.dat.data[:, 0] -= L/2
# mesh.coordinates.dat.data[:, 1] -= L/2
# mesh.coordinates.dat.data[:, 2] -= L/2

Vg = VectorFunctionSpace(mesh, "CG", 1)
Vn = FunctionSpace(mesh, "DG", 0)
Vg_ = FunctionSpace(mesh, "CG", 1)
Vd = FunctionSpace(mesh, "RT", 1)
Vc = FunctionSpace(mesh, "N1curl", 1)

Z1 = MixedFunctionSpace([Vc, Vg_])  # (u, p)
Z2 = MixedFunctionSpace([Vg_, Vg_, Vg_, Vc, Vd])  # (w, j, E, H, B)

z1 = Function(Z1)  # (u, p)
z2 = Function(Z2)  # (w, j, E, H, B)

(u, p) = split(z1)
(w, j, E, H, B) = split(z2)
(ut, pt) = split(TestFunction(Z1))
(wt, jt, Et, Ht, Bt) = split(TestFunction(Z2))

z1_prev = Function(Z1)
z2_prev = Function(Z2)
(up, pp) = split(z1_prev)
(wp, jp, Ep, Hp, Bp) = split(z2_prev)


c = Constant(1)
nu = Constant(0)
eta = Constant(0)

f = Function(Vg)
f.interpolate(Constant((0, 0)))

t = Constant(0)
dt = Constant(1/100)
T = 5

# initial condition
#u1 = -sin(2*pi * y0)
#u2 = sin(2 * pi * x0)
#B1 = -sin(2 * pi *y0) 
#B2 = sin(4 * pi * x0)

# Biskamp-Welter-1989
#phi = cos(x + 1.4) + cos(y + 0.5)
#psi = cos(2 * x + 2.3) + cos(y + 4.1)
def v_grad(x):
    return as_vector([-x.dx(1), x.dx(0)])

phi = cos(x) + cos(y)
psi = 0.5* cos(2 * x) + cos(y)

u_init = v_grad(psi)
B_init = v_grad(phi)
 
z1.sub(0).interpolate(u_init)
z2.sub(4).interpolate(B_init) 

z1_prev.assign(z1)
z2_prev.assign(z2)


(u_, p_) = z1.subfunctions
u_.rename("Velocity")
p_.rename("Pressure")
(w_, j_, E_, H_, B_) = z2.subfunctions
B_.rename("MagneticField")
j_.rename("Current")
pvd = VTKFile("output/orzag-Tang.pvd")
pvd.write(z1.sub(0), z1.sub(1), *z2.subfunctions, time=float(t))

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
# Variational forms
F1 = (
      inner((u - up) / dt, ut) * dx
    - inner(vcross(u, w), ut) * dx
    + nu * inner(curl(u), curl(ut)) * dx
    + inner(grad(p), ut) * dx
    + c * inner(vcross(H, j), ut) * dx
    + inner(u, grad(pt)) * dx
)

# F1 generate solution (u, p), in F2 (u, p) serve as known varibles for updating
F2 = (
    inner((B - Bp) / dt, Bt) * dx
    + inner(vcurl(E), Bt) * dx
    - eta * inner(j, Et) * dx
    + inner(E, Et) * dx
    + inner(vcross(H, Et), u) * dx
    + inner(w, wt) * dx
    - inner(u, vcurl(wt)) * dx
    + inner(j, jt) * dx
    - inner(B, vcurl(jt)) * dx
    - inner(B, Ht) * dx
    + inner(H, Ht) * dx
)

# Solvers
sp1 = {
    "mat_type": "aij",
    "snes_type": "newtonls",
    "snes_monitor": None,
    "ksp_monitor": None,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}


sp2 = {
    "mat_type": "aij",
    "snes_type": "newtonls",
    "ksp_monitor":None,
    "snes_monitor": None,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}

dirichlet_ids = ("on_boundary",)
bcs1 = [DirichletBC(Z1.sub(index), 0, subdomain) for index in range(len(Z1)) for subdomain in dirichlet_ids]
bcs2 = [DirichletBC(Z2.sub(index), 0, subdomain) for index in range(len(Z2)) for subdomain in dirichlet_ids]



def build_solver(F, u_sol, bcs, solver_parameters, options_prefix=None):
    problem = NonlinearVariationalProblem(F, u_sol, bcs=bcs)
    solver = NonlinearVariationalSolver(problem, solver_parameters=solver_parameters, options_prefix=options_prefix)
    return solver


time_stepper_u_p = build_solver(F1, z1, bcs1, sp1, options_prefix="primal solver")
time_stepper_other = build_solver(F2, z2, bcs2, sp2, options_prefix="dual solver")

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

divB = compute_divB(z2.sub(4))
energy = compute_energy(z1.sub(0), z2.sub(4)) # u, B
helicity_m = compute_helicity_m(z2.sub(4)) # B
helicity_c = compute_helicity_c(z1.sub(0), z2.sub(4)) # u, B
w_max, j_max, ens_total = compute_ens(z2.sub(0), z2.sub(1))

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
    time_stepper_u_p.solve()
    print("Primal solver is done successfully")
    # Update other variables (w, j, E, H, B)
    time_stepper_other.solve()
    print("Dual solver is done successfully")
    z1_prev.assign(z1)
    z2_prev.assign(z2)
    # Output
    (u, p) = z1.subfunctions
    (w, j, E, H, B) = z2.subfunctions
    divB = compute_divB(z2.sub(4))
    energy = compute_energy(z1.sub(0), z2.sub(4)) # u, B
    helicity_m = compute_helicity_m(z2.sub(4)) # B
    helicity_c = compute_helicity_c(z1.sub(0), z2.sub(4)) # u, B
    w_max, j_max, ens_total = compute_ens(z2.sub(0), z2.sub(1))
    print(f"divB: {divB:.8f}, energy={energy}, helicity_m={helicity_m}, helicity_c={helicity_c}, ens_total={ens_total}")
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
            
    pvd.write(u, p, w, j, E, H, B, time=float(t))

  

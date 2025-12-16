# 2d orsag-tang

from firedrake import *
from firedrake.petsc import PETSc
print = PETSc.Sys.Print
import csv
import numpy as np
from mpi4py import MPI
import os
baseN = 200
dp={}
# dp={"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 1)}

mesh = PeriodicUnitSquareMesh(baseN, baseN, distribution_parameters = dp)
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

#u, P, w, j, E, H, B
Z = MixedFunctionSpace([Vc, Vg_, Vg_, Vg_, Vg_, Vc, Vd])
z = Function(Z)
z_prev = Function(Z)
(u, P, w, j, E, H, B) = split(z)
(ut, Pt, wt, jt, Et, Ht, Bt) = split(TestFunction(Z))
(up, Pp, wp, jp, Ep, Hp, Bp) = split(z_prev)


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
z.sub(6).interpolate(B_init) 

z_prev.assign(z)
def spectrum(u, B):
    N = baseN
    x = np.linspace(0, 2*np.pi, N, endpoint = False)
    y = np.linspace(0, 2*np.pi, N, endpoint = False)
    X, Y = np.meshgrid(x, y, indexing="ij")

# uniform mesh for evaluation
    u_vals = np.zeros((N, N, 2))
    B_vals = np.zeros((N, N, 2))

    for i in range(N):
        for j in range(N):
            u_vals[i, j, :] = u.at([x[i], y[j]])
            B_vals[i, j, :] = B.at([x[i], y[j]])

    uhat_x = np.fft.fftn(u_vals[:, :, 0])
    uhat_y = np.fft.fftn(u_vals[:, :, 1])

    Bhat_x = np.fft.fftn(B_vals[:, :, 0])
    Bhat_y = np.fft.fftn(B_vals[:, :, 1])


    kx = np.fft.fftfreq(N, d=2*np.pi/N) * 2*np.pi
    ky = np.fft.fftfreq(N, d=2*np.pi/N) * 2*np.pi
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    K = np.sqrt(KX**2 + KY**2)


    E_u_k = 0.5 * (np.abs(uhat_x)**2 + np.abs(uhat_y)**2)
    E_B_k = 0.5 * (np.abs(Bhat_x)**2 + np.abs(Bhat_y)**2)


    kmax = int(np.max(K))
    E_u = np.zeros(kmax)
    E_B = np.zeros(kmax)

    for k in range(kmax):
        mask = (K >= k) & (K < k+1)
        E_u[k] = np.sum(E_u_k[mask])
        E_B[k] = np.sum(E_B_k[mask])

    import matplotlib.pyplot as plt

    k = np.arange(1, len(E_u))

    plt.figure()
    plt.loglog(k, E_u[1:], 'o-', label='Kinetic')
    plt.loglog(k, E_B[1:], 's-', label='Magnetic')

    # 参考 k^{-5/3}
    plt.loglog(k, 1e-2 * k**(-5/3), '--', label=r'$k^{-5/3}$')

    plt.xlabel(r'$k$')
    plt.ylabel(r'$E(k)$')
    plt.legend()
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig("spectrum.png", dpi=300)
    plt.close()  

(u_, P_, w_, j_, E_, H_, B_) = z.subfunctions
u_.rename("Velocity")
P_.rename("TotalPressure")
B_.rename("MagneticField")
j_.rename("Current")
pvd = VTKFile("output/orzag-Tang.pvd")
if mesh.comm.rank == 0:
    os.makedirs('output', exist_ok=True)
pvd.write(*z.subfunctions, time=float(t))

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
u_avg = (u + up)/2
B_avg = (B + Bp)/2

P_avg = P
w_avg = w
j_avg = j
E_avg = E
H_avg = H

F = (
    inner((u - up)/dt, ut) * dx
    - inner(vcross(u_avg, w), ut) * dx
    + nu * inner(curl(u_avg), curl(ut)) * dx
    + inner(grad(P_avg), ut) * dx
    + c * inner(vcross(H_avg, j_avg), ut) * dx
    + inner(u_avg, grad(Pt)) * dx
    + inner((B - Bp)/dt, Bt) * dx
    + inner(curl(E_avg), Bt) * dx
    - eta * inner(j_avg, Et) * dx
    + inner(E_avg, Et) * dx
    + inner(scross(u_avg, H_avg), Et) * dx
    + inner(w_avg, wt) * dx
    - inner(scurl(u_avg), wt) * dx
    + inner(j_avg, jt) * dx
    - inner(B_avg, vcurl(jt)) * dx
    - inner(B_avg, Ht) * dx
    + inner(H_avg, Ht) * dx
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

dirichlet_ids = ("on_boundary",)
bcs = [DirichletBC(Z.sub(index), 0, subdomain) for index in range(len(Z)) for subdomain in dirichlet_ids]



def build_solver(F, u_sol, bcs, solver_parameters, options_prefix=None):
    problem = NonlinearVariationalProblem(F, u_sol, bcs=bcs)
    solver = NonlinearVariationalSolver(problem, solver_parameters=solver_parameters, options_prefix=options_prefix)
    return solver


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

divB = compute_divB(z.sub(6))
energy = compute_energy(z.sub(0), z.sub(6)) # u, B
helicity_m = compute_helicity_m(z.sub(6)) # B
helicity_c = compute_helicity_c(z.sub(0), z.sub(6)) # u, B
w_max, j_max, ens_total = compute_ens(z.sub(2), z.sub(3)) # w, j

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

timestep = 0
while (float(t) < float(T-dt) + 1.0e-10):
    t.assign(t+dt)    
    if mesh.comm.rank==0:
        print(f"Solving for t = {float(t):.4f} .. ", flush=True)
    # Solve for (u, p)
    time_stepper.solve()
    # Output
    divB = compute_divB(z.sub(6))
    energy = compute_energy(z.sub(0), z.sub(6)) # u, B
    helicity_m = compute_helicity_m(z.sub(6)) # B
    helicity_c = compute_helicity_c(z.sub(0), z.sub(6)) # u, B
    w_max, j_max, ens_total = compute_ens(z.sub(2), z.sub(3)) # w, j
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
    if timestep == 20:
        spectrum(z.sub(0), z.sub(6))
    pvd.write(*z.subfunctions, time=float(t))
    z_prev.assign(z)
    timestep += 1

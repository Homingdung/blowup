# reproduce Miles-Rebholz discretization about NS alpha model
# ns - 2d
# compute 2D Taylor-Green vortex, expect to see less dofs used
from firedrake import *
import csv

def helicity_u(u):
    w_z = scurl(u)
    w = as_vector([0, 0, w_z])
    u = as_vector([u[0], u[1], 0])
    return assemble(inner(u, w)*dx)

def energy_u(u,u_b):
    return 0.5 * assemble(inner(u, u_b) * dx)
    
def div_u(u):
    return norm(div(u), "L2")

def scross(x, y):
    return x[0]*y[1] - x[1]*y[0]


def vcross(x, y):
    return as_vector([x[1]*y, -x[0]*y])


def scurl(x):
    return x[1].dx(0) - x[0].dx(1)


def vcurl(x):
    return as_vector([x.dx(1), -x.dx(0)])


# solver parameter
lu = {
    "mat_type": "aij",
    "snes_type": "newtonls",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}
sp = lu

# spatial parameters
dp={"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 1)}
baseN = 16
nref = 2

# temporal parameters
base = PeriodicUnitSquareMesh(baseN, baseN, distribution_parameters = dp)
mh = MeshHierarchy(base, nref, distribution_parameters = dp)
# shift from [0, 1]^2 to [0, 2pi]^2
scale = 2 * pi
shift = 0
for m in mh:
    m.coordinates.dat.data[:] = scale * m.coordinates.dat.data + shift
mesh = mh[-1]
x, y = SpatialCoordinate(mesh)

# spatial discretization
k = 2
Vg = VectorFunctionSpace(mesh, "CG", k)
Vg_ = FunctionSpace(mesh, "CG", k)
Q = FunctionSpace(mesh, "CG", k-1)

# time 
t = Constant(0) 
T = 10.0
dt = Constant(0.01)

# (u, p, u_b, lmbda, w, gamma)
Z = MixedFunctionSpace([Vg, Q, Vg, Q, Vg_])
z = Function(Z)
z_test = TestFunction(Z)
z_prev = Function(Z)

(u, p, u_b, lmbda, w) = split(z)
(ut, pt, u_bt, lmbdat, wt) = split(z_test)
(up, pp, u_bp, lmbdap, wp) = split(z_prev)

# initial condition
nu = Constant(0)
alpha = CellDiameter(mesh)

#U0 = Constant(1)
#kx= Constant(1)
#ky= Constant(1)
#
#u1 = U0 * sin(kx * x) * cos(ky * y)
#u2 = -U0 * cos(kx * x) * sin(ky * y)

# initial conditions: 2D-Taylor-Green vortex
u1 = sin(x) * cos(y) 
u2 = -cos(x) * sin(y)
u_init = as_vector([u1, u2])
rho = 1
p_init = rho/4 * (cos(2*x) + cos(2*y)) 

w_init = scurl(u_init)

z_prev.sub(0).interpolate(u_init)
z_prev.sub(4).interpolate(w_init)
z.assign(z_prev)

u_avg = (u + up)/2
u_b_avg = (u_b + u_bp)/2
w_avg = (w + wp)/2

F = (
    # u
     inner((u - up)/dt, ut) * dx
    -inner(vcross(u_b_avg, w_avg), ut) * dx
    - inner(p, div(ut)) * dx
    + nu * inner(grad(u_avg), grad(ut)) * dx
    # p
    - inner(div(u), pt) * dx
    # u_b
    + inner(u_b, u_bt) * dx
    + alpha**2 * inner(grad(u_b), grad(u_bt)) * dx
    - inner(lmbda, div(u_bt)) * dx
    - inner(u, u_bt) * dx
    # lmbda
    - inner(div(u_b), lmbdat) * dx
    # w
    - inner(scurl(u), wt) * dx
    + inner(w, wt) * dx
)

bcs = None
(u_, p_, u_b_, lmbda_, w_) = z.subfunctions
u_.rename("Velocity")
p_.rename("Pressure")
u_b_.rename("filteredVelocity")
lmbda_.rename("LM1")
w_.rename("Vorticity")

pvd = VTKFile("output/2dns-alpha.pvd")
pvd.write(*z.subfunctions, time = float(t))

pb = NonlinearVariationalProblem(F, z, bcs)
solver = NonlinearVariationalSolver(pb, solver_parameters = sp)

timestep = 0
data_filename = "output/data.csv"
fieldnames = ["t", "divu", "energy", "helicity"]

if mesh.comm.rank == 0:
    with open(data_filename, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

energy = energy_u(z.sub(0), z.sub(2))
helicity = helicity_u(z.sub(0))
divu = div_u(z.sub(0))

if mesh.comm.rank == 0:
    row = {
        "t": float(t),
        "divu": float(divu),
        "energy": float(energy),
        "helicity": float(helicity),
    }
    with open(data_filename, "a", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)

while (float(t) < float(T-dt)+1.0e-10):
    t.assign(t+dt)
    if mesh.comm.rank == 0:
        print(GREEN % f"Solving for t = {float(t):.4f}, dt = {float(dt)}, T = {T}, baseN = {baseN}, nref = {nref}, nu = {float(nu)}", flush=True)
    solver.solve()
    
    energy = energy_u(z.sub(0), z.sub(2))
    helicity = helicity_u(z.sub(0))
    divu = div_u(z.sub(0))

    if mesh.comm.rank == 0:
        row = {
            "t": float(t),
            "divu": float(divu),
            "energy": float(energy),
            "helicity": float(helicity),
        }
        with open(data_filename, "a", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row)

    print(row) 
    pvd.write(*z.subfunctions, time=float(t))
    timestep += 1
    z_prev.assign(z)


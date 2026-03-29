# 2D Taylor-Green vortex
# insipred by Grauer-Sideris-1991
from firedrake import *
import csv
from mpi4py import MPI
from irksome import GaussLegendre, Dt, MeshConstant, TimeStepper
import os
import math
def helicity_u(u):
    w_z = scurl(u)
    w = as_vector([0, 0, w_z])
    u = as_vector([u[0], u[1], 0])
    return assemble(inner(u, w)*dx)

def energy_u(u):
    return 0.5 * assemble(inner(u, u) * dx)

def ens(w):
    return 0.5 * assemble(inner(w, w) * dx)

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


# spatial parameters
dp={"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 1)}
baseN = 32
nref = 5

base = PeriodicUnitSquareMesh(baseN, baseN, distribution_parameters = dp)
mh = MeshHierarchy(base, nref, distribution_parameters = dp)

scale = 2 * pi
shift = 0
# 2pi * x
for m in mh:
    m.coordinates.dat.data[:] = scale * m.coordinates.dat.data + shift
mesh = mh[-1]

def run_simulation(nu, stage):
    # temporal parameters
    dt= Constant(1e-1)
    t = Constant(0)
    T = 30.0


    nu = Constant(nu)
    butcher_tableau=GaussLegendre(stage)
    (x, y) = SpatialCoordinate(mesh)

    Q = FunctionSpace(mesh, "CG", 1)
    Vc = FunctionSpace(mesh, "N1curl", 1)

    Z = MixedFunctionSpace([Vc, Q, Q])
    z = Function(Z)
    z_test = TestFunction(Z)

    (u, p, w) = split(z)
    (ut, pt, wt) = split(z_test)

    # initial conditions
    u1 = sin(x) * cos(y) 
    u2 = -cos(x) * sin(y)
    u_init = as_vector([u1, u2])
    rho = 1
    p_init = rho/4 * (cos(2*x) + cos(2*y)) 
    z.sub(0).interpolate(u_init)
    #z.sub(1).interpolate(p_init)
    z.sub(2).interpolate(scurl(u_init))

    F =(
        #u
          inner(Dt(u), ut) * dx
        - inner(vcross(u, w), ut) * dx
        + inner(grad(p), ut) * dx
        + nu * inner(scurl(u), scurl(ut)) * dx
        #p
        + inner(u, grad(pt)) * dx
        #w
        + inner(w, wt) * dx
        - inner(scurl(u), wt) * dx
        )
    # solver 
    lu = {
        "mat_type": "aij",
        "snes_type": "newtonls",
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }

    star = {
    "snes_type": "newtonls",
    "snes_monitor": None,
    "ksp_monitor": None,
    "snes_ksp_ew": None,
    "ksp_type": "fgmres",
    "ksp_converged_reason": None,
    "ksp_gmres_restart": 100,
    #"snes_rtol": 1.e-10,#
    #"snes_atol": 1.e-10,#
    "snes_atol": 1e-7, 
    "ksp_atol": 1.e-8,
    "pc_type": "mg",
    "mg_levels": {
        "ksp_type": "chebyshev",
        "ksp_chebyshev_esteig": "0.0,0.25,0,1.2",
        "ksp_convergence_test": "skip",
        "ksp_max_it": 2,
        "pc_type": "python",
        "pc_python_type": "firedrake.ASMStarPC",
        "pc_star_construct_dim": 0,
        "pc_star_backend_type": "tinyasm"
    },
    "mg_coarse": {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "factor_mat_solver_type": "mumps"
        }
    }
    sp = star
    stepper = TimeStepper(F, butcher_tableau, t, dt, z, solver_parameters = sp)

    A = stepper.solver.snes.getJacobian()[0]
    size = A.getSize()
    dofs = size[0]
    output_dir = f"output_db2d/baseN{baseN}_nref{nref}_stage{stage}_nu{float(nu)}_dt{float(dt)}_T{T}_dofs{dofs}"
    if mesh.comm.rank == 0:
        os.makedirs(output_dir, exist_ok=True)
    mesh.comm.barrier()

    data_filename = f"{output_dir}/data.csv"
    iter_filename = f"{output_dir}/iteration.csv"

    if mesh.comm.rank == 0:
        with open(data_filename, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["time", "helicity", "energy", "divu"])


    if mesh.comm.rank == 0:
        with open(iter_filename, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["time", "(nonlinear_its)", "avg_its"])

    # Compute initial quantities
    energy = energy_u(z.sub(0))
    helicity = helicity_u(z.sub(0))
    divu = div_u(z.sub(0))
    enstrophy = ens(z.sub(2))

    if mesh.comm.rank == 0:
        with open(data_filename, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([float(t), helicity, energy, divu])
        print(RED % f"Solved at t = {float(t): .4f}, divu = {divu}, helicity = {helicity}, energy = {energy}", flush=True)
    (u_, p_, w_) = z.subfunctions
    u_.rename("Velocity")
    p_.rename("TotalPressure")
    w_.rename("Vorticity")
    pvd = VTKFile(f"{output_dir}/db2d.pvd")
    pvd.write(*z.subfunctions, time = float(t))
    
    def should_write_this_step(t, T, dt):
        steps_left = math.ceil((float(T) - float(t)) / float(dt) + 1e-12)
        return steps_left <= 20
    
    timestep = 0
    while (float(t) < float(T-dt)+1.0e-10):
        t.assign(t+dt)
        if mesh.comm.rank == 0:
            print(GREEN % f"Solving for t = {float(t):.4f}, dt = {float(dt)}, T = {T}, baseN = {baseN}, nref = {nref}, stage = {stage}, nu = {float(nu)}, dofs = {size[0]}, ens={enstrophy}", flush=True)
        stepper.advance()
        
        energy = energy_u(z.sub(0))
        helicity = helicity_u(z.sub(0))
        divu = div_u(z.sub(0))
        enstrophy = ens(z.sub(2))

        linear_its = stepper.solver.snes.getLinearSolveIterations()
        nonlinear_its = stepper.solver.snes.getIterationNumber()
        avg_its = linear_its/max(1, float(nonlinear_its))
        
        if mesh.comm.rank == 0:
            with open(iter_filename, "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([f"{float(t):.4f}", f"({nonlinear_its})", f"{avg_its:.2f}"])
            print(RED % f"({nonlinear_its}) {avg_its}", flush = True)
            
            with open(data_filename, "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([float(t), helicity, energy, divu])
            print(RED % f"Solved at t = {float(t):.4f}, divu = {divu}, helicity = {helicity}, energy = {energy}", flush=True)
        if timestep % 10 ==0:
            pvd.write(*z.subfunctions, time=float(t))
        timestep += 1

stages = [2]
nu = 1e-4

for stage in stages:
    run_simulation(nu, stage)








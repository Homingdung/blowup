from firedrake import *
from firedrake.petsc import PETSc
print = PETSc.Sys.Print
import csv

L = 100
mesh = PeriodicRectangleMesh(L, L, 1, 1, direction="both")
(x0, y0) = SpatialCoordinate(mesh)
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
Re = Constant(100)
Rm = Constant(100)

f = Function(Vg)
f.interpolate(Constant((0, 0)))

t = 0
dt = Constant(1/200)

u1 = -sin(2*pi * y0)
u2 = sin(2 * pi * x0)
u_init = as_vector([u1, u2])
B1 = -sin(2 * pi *y0) 
B2 = sin(4 * pi * x0)
B_init = as_vector([B1, B2])
 
z1.sub(0).interpolate(u_init)
z2.sub(4).interpolate(B_init)  # B component

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
    + 1 / Re * inner(curl(u), curl(ut)) * dx
    + inner(grad(p), ut) * dx
    + c * inner(vcross(H, j), ut) * dx
    + inner(u, grad(pt)) * dx
)

# F1 generate solution (u, p), in F2 (u, p) serve as known varibles for updating
F2 = (
    inner((B - Bp) / dt, Bt) * dx
    + inner(vcurl(E), Bt) * dx
    - 1 / Rm * inner(j, Et) * dx
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


# def cross_helicity(u, B):
#     return assemble(inner(u, B)*dx)


# def helicity_solver(B):
#     A = Function(Vc)
#     v = TestFunction(Vc)
#     F_curl  = inner(curl(A), curl(v)) * dx - inner(B, curl(v)) * dx
#     sp = {  
#            "ksp_type":"gmres",
#            "pc_type": "ilu",
#     }
#     bcs = [DirichletBC(Vc, 0, "on_boundary")]
#     solver = build_solver(F_curl, A, bcs, solver_parameters = sp, options_prefix="solver_curlcurl")
#     solver.solve()
#     return assemble(inner(A, B)*dx)

time_stepper_u_p = build_solver(F1, z1, bcs1, sp1, options_prefix="primal solver")
time_stepper_other = build_solver(F2, z2, bcs2, sp2, options_prefix="dual solver")

def compute_divB(B):
    return norm(div(B), 'L2')

def compute_beta(dt, u, up, j, jp):
    eps = 1e-16
    j_max = Function(Vg_).interpolate(dot(j, j))
    j_max_ = j_max.dat.data.max()
    jp_max = Function(Vg_).interpolate(dot(jp, jp))
    jp_max_ = jp_max.dat.data.max()
    
    w_max = Function(Vg_).interpolate(dot(scurl(u), scurl(u)))
    w_max_ = w_max.dat.data.max()
    wp_max = Function(Vg_).interpolate(dot(scurl(up), scurl(up)))
    wp_max_ = w_max.dat.data.max()

    beta = j_max_ + w_max_
    #beta = (np.log(j_inf + w_inf + eps)-np.log(jp_inf + w_inf + eps))/np.log(1/float(dt))
    return beta

def ens_production(dt, u, up, j, jp):
    eps = 1e-16
    j_max = Function(Vg_).interpolate(dot(j, j))
    j_max_ = j_max.dat.data.max()
    jp_max = Function(Vg_).interpolate(dot(jp, jp))
    jp_max_ = jp_max.dat.data.max()
    
    w_max = Function(Vg_).interpolate(dot(scurl(u), scurl(u)))
    w_max_ = w_max.dat.data.max()
    wp_max = Function(Vg_).interpolate(dot(scurl(up), scurl(up)))
    wp_max_ = w_max.dat.data.max()

    Phi = j_max + w_max
    Phip = jp_max + wp_max

    return assemble(1/float(dt)*(inner(Phi, Phi)*dx - inner(Phip, Phip)*dx))

def j_max(j):
    j_max = Function(Vg_).interpolate(dot(j, j))
    j_max_ = j_max.dat.data.max()
    return j_max_




# Time stepping
T = 1
data_filename = "data.csv"
if mesh.comm.rank == 0:
    with open(data_filename, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["time", "beta", "ens_product", "j_max"])

while True:
    print(f"Solving for t = {float(t):.4f} .. ", flush=True)
    t += float(dt)

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
    divB = compute_divB(B)
    beta = compute_beta(dt, z1.sub(0), z1_prev.sub(0), z2.sub(1), z2_prev.sub(1))
    ens_product = ens_production(dt, z1.sub(0), z1_prev.sub(0), z2.sub(1), z2_prev.sub(1))
    jmax = j_max(z2.sub(1))
    print(f"divB: {divB:.8f}, beta={beta}, ens_product={ens_product}, jmax={jmax}")
    if mesh.comm.rank == 0:
        with open(data_filename, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([f"{float(t):.4f}", f"{beta}", f"{ens_product}", f"{jmax}"])
        

    #h_c = cross_helicity(u, B)
    #h_m = helicity_solver(B)
    #print(f"Cross helicity: {h_c:.8f} Magnetic helicity: {h_m: .8f}")
    pvd.write(u, p, w, j, E, H, B, time=float(t))

    if t >= T:
        break

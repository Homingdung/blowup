#problem of island: magnetic reconnection
from firedrake import *
import csv

# Reynolds Numbers and Lundquist Numbers, overwritten by command-line arguments
Re = Constant(5000) # Fluid Reynolds Number
Rem = Constant(5000) # Lundquist Number in this context

# parameters
dt = Constant(0.1)
T = 20
t = Constant(0)
timestep = 0

# Initial Condition Parameters
myk = Constant(0.2)
pert = Constant(-0.01)

# 2D versions of Cross Product and Curl
def scross(x, y):
    return x[0]*y[1] - x[1]*y[0]

def scurl(x):
    return x[1].dx(0) - x[0].dx(1)

baseN = 120
dp={"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 1)}

mesh = PeriodicUnitSquareMesh(baseN, baseN, direction="x", distribution_parameters = dp)
scale = 2
shift = -1
# 2 * x - 1
mesh.coordinates.dat.data[:] = scale * mesh.coordinates.dat.data + shift

Vg = VectorFunctionSpace(mesh, "CG", 1)
Vb = FunctionSpace(mesh, "N1curl", 1)
Q = FunctionSpace(mesh, "CG", 1)
# u, B, p, r
Z = MixedFunctionSpace([Vg, Vb, Q, Q])


# Equilibrium state of islands
def equilibrium_solution():
    (x, y) = SpatialCoordinate(mesh)
    b = (cosh(2*pi*y) + myk*cos(2*pi*x))
    u = as_vector([0, 0])
    p = ((1-myk*myk)/2)*(1 + 1/(b*b))
    B = as_vector([sinh(2*pi*y)/b, myk*sin(2*pi*x)/b])
    r = Constant(0)

    return (u, B, p, r)

    # RHS functions: designed to balance equilibrium solution at steady state
def island_rhs():
    (x,y) = SpatialCoordinate(mesh)
    u_rhs = Constant((0, 0), domain=mesh)
    denom = cosh(2*pi*y) + myk*cos(2*pi*x)
    coef = -(8*pi*pi*(myk*myk-1))/(Rem*denom*denom*denom)
    B_rhs = as_vector([coef*sinh(2*pi*y),coef*myk*sin(2*pi*x)])
    p_rhs = Constant(0, domain=mesh)
    r_rhs = Constant(0, domain=mesh)

    return (u_rhs, B_rhs, p_rhs, r_rhs)

    # Initial condition for time stepping: includes perturbation of magnetic field to induce coalescence
def initial_condition():
    (x,y) = SpatialCoordinate(mesh)
    z = Function(Z)

    (u0, B0, p0, r0) = equilibrium_solution()
    u = u0
    p = p0
    B = B0 + (pert/pi)*as_vector([-cos(pi*x)*sin(pi*y/2),cos(pi*y/2)*sin(pi*x)/2])
    r = r0

    z.sub(0).interpolate(u)
    z.sub(1).interpolate(B)
    z.sub(2).interpolate(p)
    z.sub(3).interpolate(r)
    return z



# Dirichlet BCs on u, B x n, and r, fix one point for p
(u_bcs, B_bcs, p_bcs, r_bcs) = equilibrium_solution() 

bcs = [
       DirichletBC(Z.sub(0), u_bcs, "on_boundary"),
       DirichletBC(Z.sub(1), B_bcs, "on_boundary"),
       DirichletBC(Z.sub(3), 0, "on_boundary"),
]

# Variational Form (steady part)
def form(z, test_z, Z):
        (u, B, p, r) = split(z)
        (v, c, q, s) = split(test_z)
        (u_rhs, B_rhs, p_rhs, r_rhs) = island_rhs()
        eps = lambda x: sym(grad(x))
        F = (
              + 2/Re * inner(eps(u), eps(v))*dx
              + inner(dot(grad(u), u), v)*dx
              - inner(p, div(v))*dx
              - inner(scurl(B),scross(B,v))*dx
              - inner(q, div(u))*dx
              + 1/Rem * inner(scurl(B), scurl(c))*dx
              - inner(scross(u, B), scurl(c))*dx
              - inner(grad(r), c)*dx
              - inner(grad(s), B)*dx
              - inner(u_rhs, v)*dx
              - inner(B_rhs, c)*dx
              - inner(p_rhs, q)*dx
              - inner(r_rhs, s)*dx
            )

        return F

lu = {
   "mat_type": "aij",
   "snes_type": "newtonls",
   "snes_monitor": None,
   "ksp_type": "preonly",
   "pc_type": "lu",
   "pc_factor_mat_solver_type": "mumps",
   "snes_rtol": 1.0e-8,
   "snes_atol": 1.0e-8,
}


# Initial Condition
z0 = initial_condition()
z1 = initial_condition()
z = Function(z0)

z_test = TestFunction(Z)

# Crank Nicolson form
F_cn = (
     inner(split(z)[0],split(z_test)[0])*dx
   - inner(split(z0)[0],split(z_test)[0])*dx
   + inner(split(z)[1],split(z_test)[1])*dx
   - inner(split(z0)[1],split(z_test)[1])*dx
   + 0.5*dt*(form(z,z_test,Z) + form(z0,z_test,Z))
  )

# BDF2 form
F_bdf2 = (
     inner(split(z)[0],split(z_test)[0])*dx
   - 4.0/3.0*inner(split(z0)[0],split(z_test)[0])*dx
   + 1.0/3.0*inner(split(z1)[0],split(z_test)[0])*dx
   + inner(split(z)[1],split(z_test)[1])*dx
   - 4.0/3.0*inner(split(z0)[1],split(z_test)[1])*dx
   + 1.0/3.0*inner(split(z1)[1],split(z_test)[1])*dx
   + 2.0/3.0*dt*form(z,z_test,Z)
  )


sp = lu
nvproblem_cn = NonlinearVariationalProblem(F_cn, z, bcs=bcs)
solver_cn = NonlinearVariationalSolver(nvproblem_cn, solver_parameters=sp)

# Set up nonlinear solver for later time steps
nvproblem_bdf2 = NonlinearVariationalProblem(F_bdf2, z, bcs=bcs)
solver_bdf2 = NonlinearVariationalSolver(nvproblem_bdf2, solver_parameters=sp)

pvd = VTKFile("output/island.pvd")
(u_, B_, p_, r_) = z.subfunctions
u_.rename("Velocity")
B_.rename("MagneticField")
p_.rename("Pressure")
r_.rename("LagrangeMultiplier")

j = Function(Q, name = "current")
j.interpolate(1/Rem * scurl(B_))

pvd.write(u_, B_, p_, r_, j, time = float(t))

def compute_beta(dt, u, up, j, jp):
    eps = 1e-16
    j_max = Function(Q).interpolate(dot(j, j))
    j_max_ = j_max.dat.data.max()
    jp_max = Function(Q).interpolate(dot(jp, jp))
    jp_max_ = jp_max.dat.data.max()
    
    w_max = Function(Q).interpolate(dot(scurl(u), scurl(u)))
    w_max_ = w_max.dat.data.max()
    wp_max = Function(Q).interpolate(dot(scurl(up), scurl(up)))
    wp_max_ = w_max.dat.data.max()

    beta = j_max_ + w_max_
    #beta = (np.log(j_inf + w_inf + eps)-np.log(jp_inf + w_inf + eps))/np.log(1/float(dt))
    return beta

def ens_production(dt, u, up, j, jp):
    eps = 1e-16
    j_max = Function(Q).interpolate(dot(j, j))
    j_max_ = j_max.dat.data.max()
    jp_max = Function(Q).interpolate(dot(jp, jp))
    jp_max_ = jp_max.dat.data.max()
    
    w_max = Function(Q).interpolate(dot(scurl(u), scurl(u)))
    w_max_ = w_max.dat.data.max()
    wp_max = Function(Q).interpolate(dot(scurl(up), scurl(up)))
    wp_max_ = w_max.dat.data.max()

    Phi = j_max + w_max
    Phip = jp_max + wp_max

    return assemble(1/float(dt)*(inner(Phi, Phi)*dx - inner(Phip, Phip)*dx))

def j_max(j):
    j_max = Function(Q).interpolate(dot(j, j))
    j_max_ = j_max.dat.data.max()
    return j_max_

jp = Function(Q)
jp.assign(j)

data_filename = "data.csv"
if mesh.comm.rank == 0:
    with open(data_filename, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["time", "beta", "ens_product", "ReconRate", "j_max"])

# reconnection rate
(u00, B00, p00, r00) = equilibrium_solution()
z00 = Function(Z)
z00.sub(1).interpolate(B00)

# functions to measure J
j00CG = Function(Q)
j_CG = Function(Q)
testCG = TestFunction(Q)
trialCG = TrialFunction(Q)

j00CG = project(scurl(z00.sub(1)), Q)
F_curlBCG = inner(trialCG, testCG) * dx - inner(curl(B_), testCG) * dx
postproc_params_CG = {
                         "snes_type": "ksponly",
                         "snes_lag_jacobian": -2,
                         "ksp_type": "cg",
                           }

problem_BCG = LinearVariationalProblem(lhs(F_curlBCG), rhs(F_curlBCG),
                                       j_CG, constant_jacobian=True)
solver_BCG = LinearVariationalSolver(problem_BCG, solver_parameters=postproc_params_CG)



while (float(t) < float(T-dt) + 1.0e-10):
    t.assign(t+dt)    
    dofs = Z.dim()

    if mesh.comm.rank==0:
        print(f"Solving for t = {float(t):.4f} .. dofs = {dofs}", flush=True)
    solver_cn.solve()
    solver_BCG.solve()
    
    j.interpolate(1/Rem * scurl(z.sub(1)))
    
    beta = compute_beta(dt, z.sub(0), z0.sub(0), j, jp)
    ens_product = ens_production(dt, z.sub(0), z0.sub(0), j, jp)
    ReconRate = (1/float(Rem)) * (j_CG((0, 0)) - j00CG((0, 0)))
    jmax = j_max(j)

    print(RED % f"blowup beta={beta}, ens_product={ens_product}, ReconRate={ReconRate}, jmax={jmax}")
    if mesh.comm.rank == 0:
        with open(data_filename, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([f"{float(t):.4f}", f"{beta}", f"{ens_product}", f"{ReconRate}", f"{jmax}"])
    

    if timestep % 10 ==0:
        pvd.write(u_, B_, p_, r_, j, time = float(t))


    z1.assign(z0)
    z0.assign(z)
    timestep += 1
    jp.assign(j)



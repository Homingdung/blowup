# 2d orsag-tang

from firedrake import *
import numpy as np

baseN = 200
mesh = PeriodicUnitSquareMesh(baseN, baseN)
mesh.coordinates.dat.data[:] *= 2 * pi
(x0, y0) = SpatialCoordinate(mesh)

Vg = VectorFunctionSpace(mesh, "CG", 1)
Vd = FunctionSpace(mesh, "RT", 1)

u = Function(Vg)
B = Function(Vd)
# initial condition
u1 = -sin(2*pi * y0)
u2 = sin(2*pi * x0)
u_init = as_vector([u1, u2])
u.interpolate(u_init)

B1 = -sin(2 * pi *y0) 
B2 = sin(4 * pi * x0)
B_init = as_vector([B1, B2])
B.interpolate(B_init)
#u_hat = np.fft.fft(u.dat.data)
#B_hat = np.fft.fft(B.dat.data)


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

spectrum(u, B)

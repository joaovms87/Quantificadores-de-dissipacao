import numpy as np
import scipy.linalg as sl


DELTA = 2  # total variation of gamma1
beta = 0.1


def h(t):  # protocol for gamma1 variation
    if t==0:
        return h_0
    else:
        return h_0 + DELTA/tau*t


# eigen-energies as functions of gamma1 = g
def E1(h):
    return 1 - h


def E2(h):
    return 1 - h


def E3(h):
    return h + 1


def E4(h):
    return h + 1


def E5(h):
    return -2*np.sqrt(h**2 - h + 1) - h - 1


def E6(h):
    return 2*np.sqrt(h**2 - h + 1) - h - 1

def E7(h):
    return -2*np.sqrt(h**2 + h + 1) + h - 1


def E8(h):
    return 2*np.sqrt(h**2 + h + 1) + h - 1


def partition(t):
    g = h(t)
    e1, e2, e3, e4, e5, e6, e7, e8 = E1(g), E2(g), E3(g), E4(g), E5(g), E6(g), E7(g), E8(g)
    return np.exp(-beta*e1) + np.exp(-beta*e2) + np.exp(-beta*e3) + np.exp(-beta*e4) + \
           np.exp(-beta*e5) + np.exp(-beta*e6) + np.exp(-beta*e7) + np.exp(-beta*e8)


def rho_eq(t):
    Z, g = partition(t), h(t)
    e1, e2, e3, e4, e5, e6, e7, e8 = E1(g), E2(g), E3(g), E4(g), E5(g), E6(g), E7(g), E8(g)

    v1 = np.array([0, -1, 0, 1, 0, 0, 0, 0], float)
    v1 /= np.linalg.norm(v1)
    v2 = np.array([0, -1, 1, 0, 0, 0, 0, 0], float)
    v2 /= np.linalg.norm(v2)
    v3 = np.array([0, 0, 0, 0, -1, 0, 1, 0], float)
    v3 /= np.linalg.norm(v3)
    v4 = np.array([0, 0, 0, 0, -1, 1, 0, 0], float)
    v4 /= np.linalg.norm(v4)
    v5 = np.array([-1+2*g+2*np.sqrt(1-g+g**2), 0, 0, 0, 1, 1, 1,0])
    v5 /= np.linalg.norm(v5)
    v6 = np.array([-1+2*g-2*np.sqrt(1-g+g**2), 0, 0, 0, 1, 1, 1,0])
    v6 /= np.linalg.norm(v6)
    v7 = np.array([0, -(-1-g-np.sqrt(1+g+g**2))/(1-g+np.sqrt(1+g+g**2)),
                   -(-1-g-np.sqrt(1+g+g**2))/(1-g+np.sqrt(1+g+g**2)),
                   -(-1-g-np.sqrt(1+g+g**2))/(1-g+np.sqrt(1+g+g**2)), 0, 0, 0, 1])
    v7 /= np.linalg.norm(v7)
    v8 = np.array([0, -(1+g-np.sqrt(1+g+g**2))/(-1+g+np.sqrt(1+g+g**2)),
                   -(1+g-np.sqrt(1+g+g**2))/(-1+g+np.sqrt(1+g+g**2)),
                   -(1+g-np.sqrt(1+g+g**2))/(-1+g+np.sqrt(1+g+g**2)), 0, 0, 0, 1])
    v8 /= np.linalg.norm(v8)

    p = np.exp(-beta*e1)*np.outer(v1, v1)
    p+= np.exp(-beta*e2)*np.outer(v2, v2)
    p+= np.exp(-beta*e3)*np.outer(v3, v3)
    p+= np.exp(-beta*e4)*np.outer(v4, v4)
    p+= np.exp(-beta*e5)*np.outer(v5, v5)
    p+= np.exp(-beta*e6)*np.outer(v6, v6)
    p+= np.exp(-beta*e7)*np.outer(v7, v7)
    p+= np.exp(-beta*e8)*np.outer(v8, v8)

    return p/Z


def rho_ad(t):
    Z, g, g0 = partition(0), h(t), h_0
    e10, e20, e30, e40, e50, e60, e70, e80 = E1(h_0), E2(h_0), E3(h_0), E4(h_0), E5(h_0), E6(h_0), E7(h_0), E8(h_0)

    v1 = np.array([0, -1, 0, 1, 0, 0, 0, 0])
    v1 = v1/np.linalg.norm(v1)
    v2 = np.array([0, -1, 1, 0, 0, 0, 0, 0])
    v2 = v2/np.linalg.norm(v2)
    v3 = np.array([0, 0, 0, 0, -1, 0, 1, 0])
    v3 = v3/np.linalg.norm(v3)
    v4 = np.array([0, 0, 0, 0, -1, 1, 0, 0])
    v4 = v4/np.linalg.norm(v4)
    v5 = np.array([-1+2*g+2*np.sqrt(1-g+g**2), 0, 0, 0, 1, 1, 1,0])
    v5 = v5/np.linalg.norm(v5)
    v6 = np.array([-1+2*g-2*np.sqrt(1-g+g**2), 0, 0, 0, 1, 1, 1,0])
    v6 = v6/np.linalg.norm(v6)
    v7 = np.array([0, -(-1-g-np.sqrt(1+g+g**2))/(1-g+np.sqrt(1+g+g**2)),
                   -(-1-g-np.sqrt(1+g+g**2))/(1-g+np.sqrt(1+g+g**2)),
                   -(-1-g-np.sqrt(1+g+g**2))/(1-g+np.sqrt(1+g+g**2)), 0, 0, 0, 1])
    v7 = v7/np.linalg.norm(v7)
    v8 = np.array([0, -(1+g-np.sqrt(1+g+g**2))/(-1+g+np.sqrt(1+g+g**2)),
                   -(1+g-np.sqrt(1+g+g**2))/(-1+g+np.sqrt(1+g+g**2)),
                   -(1+g-np.sqrt(1+g+g**2))/(-1+g+np.sqrt(1+g+g**2)), 0, 0, 0, 1])
    v8 = v8/np.linalg.norm(v8)

    p = np.exp(-beta*e10)*np.outer(v1, v1)
    p+= np.exp(-beta*e20)*np.outer(v2, v2)
    p+= np.exp(-beta*e30)*np.outer(v3, v3)
    p+= np.exp(-beta*e40)*np.outer(v4, v4)
    p+= np.exp(-beta*e50)*np.outer(v5, v5)
    p+= np.exp(-beta*e60)*np.outer(v6, v6)
    p+= np.exp(-beta*e70)*np.outer(v7, v7)
    p+= np.exp(-beta*e80)*np.outer(v8, v8)

    return p / Z


def Ham(t):
    g = h(t)
    H_t = np.zeros([8, 8], complex)
    H_t[0, 0], H_t[1, 1], H_t[2, 2], H_t[3, 3], H_t[4, 4], H_t[5, 5], H_t[6, 6], H_t[7, 7] = -3*g, -g, -g, -g, g, g, g, 3*g
    H_t[0, 4], H_t[0, 5], H_t[0, 6] = -1, -1, -1
    H_t[1, 2], H_t[1, 3], H_t[1, 7] = -1, -1, -1
    H_t[2, 1], H_t[2, 3], H_t[2, 7] = -1, -1, -1
    H_t[3, 1], H_t[3, 2], H_t[3, 7] = -1, -1, -1
    H_t[4, 0], H_t[4, 5], H_t[4, 6] = -1, -1, -1
    H_t[5, 0], H_t[5, 4], H_t[5, 6] = -1, -1, -1
    H_t[6, 0], H_t[6, 4], H_t[6, 5] = -1, -1, -1
    H_t[7, 1], H_t[7, 2], H_t[7, 3] = -1, -1, -1
    return H_t


def D(p1, p2):  # relative entropy between states p1 and p2
    ln1, ln2 = sl.logm(p1), sl.logm(p2)
    prod1, prod2 = np.matmul(p1, ln1), np.matmul(p1, ln2)
    return np.trace(prod1-prod2)


def D_eq(p, t):
    deltaF = -1/beta*np.log(partition(t)/partition(0))
    return beta*(W(p)-deltaF)


def W(p):  # calculates total work done in the process
    Ef = np.trace(np.matmul(p, Ham(tau)))
    return np.real(Ef - E0)


def W_ad(t):  # calculates the work done up to time t if the evolution is adiabatic
    Ef = np.trace(np.matmul(rho_ad(t), Ham(t)))
    return np.real(Ef - E0)


def f(r, t):  # array of functions at the right-hand-side of the simultaneous equations
    g = h(t)
    p = np.zeros([8, 8], complex)
    p[0] = r[:8].copy()
    p[1, 1:] = r[8:15].copy()
    p[2, 2:] = r[15:21].copy()
    p[3, 3:] = r[21:26].copy()
    p[4, 4:] = r[26:30].copy()
    p[5, 5:] = r[30:33].copy()
    p[6, 6:] = r[33:35].copy()
    p[7, 7:] = r[35:].copy()
    for k in range(1, 8):
        for z in range(k):
            p[k,z] = np.conj(p[z,k])

    d = - 1j * (Ham(t)@p - p@Ham(t))

    new_r = np.concatenate((d[0], d[1, 1:], d[2, 2:], d[3, 3:], d[4, 4:], d[5, 5:], d[6, 6:], d[7, 7:]), axis=0)

    return new_r


# The next functions are for the adaptive Runge-Kutta method
def ratio(a, b):
    aux = (a-b)/30
    eps = np.linalg.norm(aux)
    return u*delta/eps


def new_u(u, p):
    u_new = u*p**0.25
    if u_new>2*u:
        return 2*u
    else:
        return u_new


# list of the process durations that will be investigated
taupoints = np.logspace(-3, 5, 200)
l = len(taupoints)
# lists of total work and final D - each entry corresponds to a process with tau in taupoints:
Wpoints = np.empty(l, complex)
Dpoints = np.empty(l, complex)    # D[rho(tau)||rho_eq(tau)]
D2points = np.empty(l, complex)   # D[rho(tau)||rho_ad(tau)]
# compute these quantities for each tau

delta = 1e-10  # required accuracy per unit time

for h_0 in [-1.1, 0.1]:
    # array of rho elements at time 0; necessary to apply adaptive Runge-Kutta method
    rho0 = rho_eq(0)
    r0 = np.concatenate((rho0[0].copy(), rho0[1, 1:].copy(), rho0[2, 2:].copy(), rho0[3, 3:].copy(), rho0[4, 4:].copy(),
                    rho0[5, 5:].copy(), rho0[6, 6:].copy(), rho0[7, 7:].copy()), axis=0)
    for j, tau in enumerate(taupoints):
        t = 0
        N = 100
        u = tau/N  # Initial step size

        # Solve system of differential equations by the adaptive Runge-Kutta method
        r = r0.copy()
        while t<tau:
            # two steps of size h
            k1 = u*f(r, t)
            k2 = u*f(r+0.5*k1, t+0.5*u)
            k3 = u*f(r+0.5*k2, t+0.5*u)
            k4 = u*f(r+k3, t+u)
            r1 = r + (k1+2*k2+2*k3+k4)/6
            k1 = u*f(r1, t+u)
            k2 = u*f(r1+0.5*k1, t+1.5*u)
            k3 = u*f(r1+0.5*k2, t+1.5*u)
            k4 = u*f(r1+k3, t+2*u)
            r1 += (k1+2*k2+2*k3+k4)/6
            # one step of size 2h
            k1 = 2*u*f(r, t)
            k2 = 2*u*f(r+0.5*k1, t+u)
            k3 = 2*u*f(r+0.5*k2, t+u)
            k4 = 2*u*f(r+k3, t+2*u)
            r2 = r + (k1+2*k2+2*k3+k4)/6

            p = ratio(r1, r2)
            if p>=1:  # precision greater than required
                r = r1  # keep the most precise result
                t = t+2*u  # next t
            # if p>=1 is not satisfied, repeat everything without going to next t
            u = new_u(u, p)  # changes h regardless of the value of p

        # build the density matrix at time tau for the process of duration tau
        rho = np.empty([8, 8], complex)
        rho[0] = r[:8].copy()
        rho[1, 1:] = r[8:15].copy()
        rho[2, 2:] = r[15:21].copy()
        rho[3, 3:] = r[21:26].copy()
        rho[4, 4:] = r[26:30].copy()
        rho[5, 5:] = r[30:33].copy()
        rho[6, 6:] = r[33:35].copy()
        rho[7, 7:] = r[35:].copy()
        for k in range(1, 8):
            for z in range(k):
                rho[k,z] = np.conj(rho[z,k])

        # initial energy for computation of the work:
        E0 = np.trace(np.matmul(rho0, Ham(0)))
        # compute the work done in the process of duration tau and save it
        Wpoints[j] = W(rho)
        # compute the relative entropy and save it
        Dpoints[j] = D(rho, rho_eq(tau))
        # compute D[rho(tau)||rho_ad(tau)] and save it
        D2points[j] = D(rho, rho_ad(tau))

    # compute W_ad(tau) for comparison
    # (it is independent of tau, so we compute for the last tau only)
    WADpoints = W_ad(tau)*np.ones(l, complex)

    # save everything
    np.savetxt(f'new W beta={beta} h0={h_0}', Wpoints)
    np.savetxt(f'new W_ad beta={beta} h0={h_0}', WADpoints)
    np.savetxt(f'new D beta={beta} h0={h_0}', Dpoints)
    np.savetxt(f'new D2 beta={beta} h0={h_0}', D2points)

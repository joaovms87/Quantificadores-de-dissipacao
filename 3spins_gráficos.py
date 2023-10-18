import numpy as np
import matplotlib.pyplot as plt
from uncertainties import ufloat

beta = 0.1
DELTA = 2

# list of the process durations that will be investigated
taupoints = np.logspace(-3, 5, 200)

for i, tau in enumerate(taupoints):
    if tau > 500:
        break

for j, tau in enumerate(taupoints):
     if tau > 1e4:
        break

taupoints = taupoints[:j].copy()

for h0 in [-1.1, 0.1]:
    # reading the previously calculated values for W
    Wpoints = np.loadtxt(f'W beta={beta} h0={h0}', complex)[:j]
    WADpoints = np.loadtxt(f'W_ad beta={beta} h0={h0}', complex)[:j]
    n = len(Wpoints)

    Wex = np.real(Wpoints - WADpoints)

    Dpoints = np.loadtxt(f'D beta={beta} h0={h0}', complex)[:j]
    D2points = np.loadtxt(f'D2 beta={beta} h0={h0}', complex)[:j]


    fit1, cov1 = np.polyfit(np.log(taupoints[i:]), np.log(np.real(Wpoints[i:]-WADpoints[i:])), deg=1, cov=True)
    print(f'For h0={h0}, Wex fit...')
    print(ufloat(fit1[0], np.sqrt(cov1[0, 0])))
    W_fit1 = taupoints**fit1[0] * np.exp(fit1[1])

    # plot Wex in log scale
    fig, ax = plt.subplots()
    ax.loglog(taupoints, Wex, 'k.', label=r'$W_\tau(\tau)-W_{ad}(\tau)$')
    ax.loglog(taupoints, W_fit1, 'b--', label=r'Ajuste para $\tau \cdot J/\hbar > 500$')
    ax.legend(fancybox=False, edgecolor='black')
    ax.set_xlabel(r'$\tau\cdot J/\hbar$')
    ax.set_ylabel('$W \cdot J^{-1}$')
    ax.set_title(r'Dependência com $\tau$ do trabalho excedente')
    ax.set_xlim(1e-2, 1e4)
    ax.set_box_aspect(1)
    plt.savefig(f'D:\GRÁFICOS IC\FAPESP\spins3\h0={h0}/log(W)_log(tau).png', bbox_inches='tight')


    # plot W in linear scale
    fig, ax = plt.subplots()
    ax.plot(taupoints[:i], np.real(Wpoints[:i]), 'r.', label=r'$W_\tau(\tau)$')
    ax.plot(taupoints[:i], np.real(WADpoints[:i]), 'k--', label=r'$W_{ad}(\tau)$')
    ax.legend(fancybox=False, edgecolor='black')
    ax.set_xlabel(r'$\tau\cdot J/\hbar$')
    ax.set_ylabel('$W \cdot J^{-1}$')
    ax.set_title(r'Dependência com $\tau$ do trabalho total')
    ax.set_box_aspect(1)
    ax.set_xlim(1e-2, 100)
    plt.savefig(f'D:\GRÁFICOS IC\FAPESP\spins3\h0={h0}/W_linear.png', bbox_inches='tight')

    # plot D in log scale with Wex
    fig, ax = plt.subplots()
    ax.loglog(taupoints, Wex, 'r.', label=r'$W_{\tau,ex}(\tau)$')
    ax.loglog(taupoints, np.real(Dpoints), 'k.', label=r'$D[\rho_\tau (\tau) || \rho_{eq} (\tau)]$')
    ax.legend(fancybox=False, edgecolor='black', loc='lower left')
    ax.set_xlabel(r'$\tau\cdot J/\hbar$')
    ax.set_ylabel('$W \cdot J^{-1} \ $ ou $\ \ D$')
    ax.set_title(r'Comparação entre $W_{ex}$ e $D$')
    ax.set_xlim(1e-2, 1e4)
    ax.set_box_aspect(1)
    plt.savefig(f'D:\GRÁFICOS IC\FAPESP\spins3\h0={h0}/D vs Wex.png', bbox_inches='tight')

    # fit for D2
    fit1, cov1 = np.polyfit(np.log(taupoints[i:]), np.log(np.real(D2points[i:])), deg=1, cov=True)
    print(f'For h0={h0}, D2 fit...')
    print(ufloat(fit1[0], np.sqrt(cov1[0, 0])))
    D2_fit1 = taupoints**fit1[0] * np.exp(fit1[1])


    # plot D2 in log scale
    fig, ax = plt.subplots()
    ax.loglog(taupoints, np.real(D2points), 'k.', label=r'$D_{\tau,qs}(\tau)$')
    ax.loglog(taupoints, D2_fit1, 'b--', label=r'Ajuste para $\tau \cdot J/\hbar >500$')
    ax.legend(fancybox=False, edgecolor='black')
    ax.set_xlabel(r'$\tau\cdot J/\hbar$')
    ax.set_ylabel(r'$D_{\tau,qs}(\tau)$')
    ax.set_title(r'Dependência com $\tau$ de $D_{\tau,qs}(\tau)$')
    ax.set_xlim(1e-2, 1e4)
    ax.set_box_aspect(1)
    plt.savefig(f'D:\GRÁFICOS IC\FAPESP\spins3\h0={h0}/log(D2)_log(tau).png', bbox_inches='tight')

    # plot D2 in log scale together with Wex
    fig, ax = plt.subplots()
    ax.loglog(taupoints, Wex, 'r.', label=r'$W_{\tau,ex}(\tau)$')
    ax.loglog(taupoints, np.real(D2points), 'b.', label=r'$D_{\tau,qs}(\tau)$')
    ax.loglog(taupoints, W_fit1, 'k-', label=r'Ajuste de $W_{\tau, ex}(\tau)$ para $\tau \cdot J/\hbar > 500$')
    ax.loglog(taupoints, D2_fit1, 'k--', label=r'Ajuste de $D_{\tau, qs}(\tau)$ para $\tau \cdot J/\hbar > 500$')
    ax.legend(fancybox=False, edgecolor='black', loc='lower left')
    ax.set_xlabel(r'$\tau\cdot J/\hbar$')
    ax.set_ylabel('$W \cdot J^{-1} \ $ ou $\ \ D$')
    ax.set_title(r'Comparação entre $W_{\tau,ex}(\tau)$ e $D_{\tau,qs}(\tau)$')
    ax.set_ylim(0.03*min(np.real(D2points)), 1.5*max(Wex))
    ax.set_xlim(1e-2, 1e4)
    ax.set_box_aspect(1)
    plt.savefig(f'D:\GRÁFICOS IC\FAPESP\spins3\h0={h0}/Wex vs D2.png', bbox_inches='tight')

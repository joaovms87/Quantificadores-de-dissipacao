import numpy as np
import matplotlib.pyplot as plt
from uncertainties import ufloat


beta = 0.1
for B0 in [-1, -2]:

    # reading the previously calculated values for W
    Wpoints = np.loadtxt(f'W; B0={B0}; beta={beta}', complex)
    WADpoints = np.loadtxt(f'W_ad; B0={B0}; beta={beta}', complex)
    n = len(Wpoints)

    Dpoints = np.loadtxt(f'D; B0={B0}; beta={beta}', complex)
    D2points = np.loadtxt(f'D2; B0={B0}; beta={beta}', complex)

    # list of the process durations that will be investigated
    taupoints = np.logspace(-2, 5, n)

    for i, tau in enumerate(taupoints):
        if tau > 500:
            break

    for j, tau in enumerate(taupoints):
        if tau > 10000:
            break

    # fit for log scale (W)
    fit = np.polyfit(np.log(taupoints[i:j]), np.log(np.real(Wpoints[i:j]-WADpoints[i:j])), deg=1, cov=True)
    print(f'Wex... For B0={B0}:')
    print(ufloat(fit[0][0], np.sqrt(fit[1][0, 0])))
    W_fit = taupoints**fit[0][0] * np.exp(fit[0][1])

    # fit for log scale (D2)
    fit = np.polyfit(np.log(taupoints[i:j]), np.log(np.real(D2points[i:j])), deg=1, cov=True)
    print(f'D2... For B0={B0}:')
    print(ufloat(fit[0][0], np.sqrt(fit[1][0, 0])))
    D2_fit = taupoints**fit[0][0] * np.exp(fit[0][1])


    # plot W in linear scale
    fig, ax = plt.subplots()
    ax.plot(taupoints, np.real(Wpoints), 'r.', label=r'$W_\tau(\tau)$')
    ax.plot(taupoints, np.real(WADpoints), 'k--', label=r'$W_{qs}(\tau)$')
    ax.legend(fancybox=False, edgecolor='black')
    ax.set_xlabel(r'$\tau\cdot J/\hbar$')
    ax.set_ylabel('$W \cdot J^{-1}$')
    ax.set_title(r'Dependência com $\tau$ do trabalho total')
    ax.set_xlim(0, 100)
    ax.set_box_aspect(1)
    plt.savefig(f'D:\GRÁFICOS IC\FAPESP\LZ\B0={B0}/W(tau),B0={B0},beta={beta}.png', bbox_inches='tight')

    # plot Wex in log scale
    Wex = np.real(Wpoints-WADpoints)
    fig, ax = plt.subplots()
    ax.loglog(taupoints, Wex, 'r.', label=r'$W_{\tau,ex}(\tau)$')
    ax.loglog(taupoints, W_fit, 'k--', label=r'Ajuste para $\tau \cdot J/\hbar > 500$')
    ax.legend(fancybox=False, edgecolor='black')
    ax.set_xlabel(r'$\tau\cdot J/\hbar$')
    ax.set_ylabel('$W \cdot J^{-1}$')
    ax.set_title(r'Dependência com $\tau$ do trabalho excedente')
    ax.set_xlim(1e-2, 1e4)
    ax.set_ylim(min(Wex), 1.5*max(Wex))
    ax.set_box_aspect(1)
    plt.savefig(f'D:\GRÁFICOS IC\FAPESP\LZ\B0={B0}/log(Wex)_log(tau),B0={B0},beta={beta}.png', bbox_inches='tight')

    # plot D in log scale with Wex
    fig, ax = plt.subplots()
    ax.loglog(taupoints, Wex, 'r.', label=r'$W_{\tau,ex}(\tau)$')
    ax.loglog(taupoints, np.real(Dpoints), 'k.', label=r'$D[\rho_\tau (\tau) || \rho_{eq} (\tau)]$')
    ax.legend(fancybox=False, edgecolor='black', loc='lower left')
    ax.set_xlabel(r'$\tau\cdot J/\hbar$')
    ax.set_ylabel('$W \cdot J^{-1} \ $ ou $\ \ D$')
    ax.set_title(r'Comparação entre $W_{ex}$ e $D$')
    ax.set_xlim(1e-2, 1e4)
    ax.set_ylim(min(Wex), 1.5*max(Wex))
    ax.set_box_aspect(1)
    plt.savefig(f'D:\GRÁFICOS IC\FAPESP\LZ\B0={B0}/Wex vs D,B0={B0},beta={beta}.png', bbox_inches='tight')

    # plot D2 in log scale together with Wex
    fig, ax = plt.subplots()
    ax.loglog(taupoints, Wex, 'r.', label=r'$W_{\tau,ex}(\tau)$')
    ax.loglog(taupoints, W_fit, 'k-', label=r'Ajuste de $W_{ex}$ para $\tau \cdot J/\hbar > 500$')
    ax.loglog(taupoints, np.real(D2points), 'b.', label=r'$D_{\tau,qs}(\tau)$')
    ax.loglog(taupoints, D2_fit, 'k--', label=r'Ajuste de $D_{\tau,qs}$ para $\tau \cdot J/\hbar > 500$')
    ax.legend(fancybox=False, edgecolor='black', loc='lower left')
    ax.set_xlabel(r'$\tau\cdot J/\hbar$')
    ax.set_ylabel('$W \cdot J^{-1} \ $ ou $\ \ D$')
    ax.set_title(r'Comparação entre $W_{\tau,ex}(\tau)$ e $D_{\tau,qs}(\tau)$')
    ax.set_xlim(1e-2, 1e4)
    ax.set_ylim(0.1*min(Wex), 1.5*max(Wex))
    ax.set_box_aspect(1)
    plt.savefig(f'D:\GRÁFICOS IC\FAPESP\LZ\B0={B0}/Wex vs D(rho,rho_ad),B0={B0},beta={beta}.png', bbox_inches='tight')

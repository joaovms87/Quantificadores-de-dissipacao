import numpy as np
import matplotlib.pyplot as plt
from uncertainties import ufloat


beta = 0.1  # parâmetro de temperatura
for B0 in [-1, -2]:  # pplotar os gráficos para cada caso estudado

    # ler os valores previamente calculados para o trabalho
    Wpoints = np.loadtxt(f'W; B0={B0}; beta={beta}', complex)
    WQSpoints = np.loadtxt(f'W_ad; B0={B0}; beta={beta}', complex)
    n = len(Wpoints)

    # ler os valores previamente calculados para as entropias relativas
    Dpoints = np.loadtxt(f'D; B0={B0}; beta={beta}', complex)
    D2points = np.loadtxt(f'D2; B0={B0}; beta={beta}', complex)

    # lista de durações 'tau' que serão investigadas
    taupoints = np.logspace(-2, 5, n)

    # salvar o índice i de taupoints em que 'tau' fica maior que 500
    # esse índice é importante para saber a partir de qual 'tau' será feito o ajuste
    for i, tau in enumerate(taupoints):
        if tau > 500:
            break

    # salvar o índice j de taupoints em que 'tau' fica maior que 10000
    # esse índice é salvo porque os dados tais que tau>10000 serão descartados
    for j, tau in enumerate(taupoints):
        if tau > 10000:
            break

    # fit do trabalho excedente
    # a variável fit recebe dois arrays
    # o primeiro array dá os resultados do fit (o primeiro elemento desse array dá -n, segundo a notação do relatório)
    # o segundo array é a matriz de covariância do ajuste
    fit = np.polyfit(np.log(taupoints[i:j]), np.log(np.real(Wpoints[i:j]-WQSpoints[i:j])), deg=1, cov=True)
    print(f'Wex... For B0={B0}:')
    print(ufloat(fit[0][0], np.sqrt(fit[1][0, 0])))  # imprime -n e sua incerteza
    W_fit = taupoints**fit[0][0] * np.exp(fit[0][1])  # calcula a reta do ajuste para plotar posteriormente

    # fit da entropia relativa D[rho||rho_qs]
    # mesmo procedimento que o fit para o trabalho excedente
    fit = np.polyfit(np.log(taupoints[i:j]), np.log(np.real(D2points[i:j])), deg=1, cov=True)
    print(f'D2... For B0={B0}:')
    print(ufloat(fit[0][0], np.sqrt(fit[1][0, 0])))
    D2_fit = taupoints**fit[0][0] * np.exp(fit[0][1])


    # plotar o trabalho W e o trabalho quase-estático W_qs em escala linear
    fig, ax = plt.subplots()
    ax.plot(taupoints, np.real(Wpoints), 'r.', label=r'$W_\tau(\tau)$')
    ax.plot(taupoints, np.real(WQSpoints), 'k--', label=r'$W_{qs}(\tau)$')
    ax.legend(fancybox=False, edgecolor='black')
    ax.set_xlabel(r'$\tau\cdot J/\hbar$')
    ax.set_ylabel('$W \cdot J^{-1}$')
    ax.set_title(r'Dependência com $\tau$ do trabalho total')
    ax.set_xlim(0, 100)
    ax.set_box_aspect(1)
    plt.savefig(f'D:\GRÁFICOS IC\FAPESP\LZ\B0={B0}/W(tau),B0={B0},beta={beta}.png', bbox_inches='tight')

    # plotar Wex em escala logarítmica, junto com o ajuste
    Wex = np.real(Wpoints-WQSpoints)
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

    # plotar D[rho||rho_eq] junto com Wex em escala logarítmica
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

    # plotar Wex e D[rho||rho_qs] (com seus respectivos ajustes) em escala logarítmica
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

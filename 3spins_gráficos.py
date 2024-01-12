import numpy as np
import matplotlib.pyplot as plt
from uncertainties import ufloat

beta = 0.1  # parâmetro de temperatura
DELTA = 2  # variação total do parâmetro externo

# lista das durações 'tau' que serão investigadas
taupoints = np.logspace(-3, 5, 200)

# salvar o índice i de taupoints em que 'tau' fica maior que 500
# esse índice é importante para saber a partir de qual 'tau' será feito o ajuste
for i, tau in enumerate(taupoints):
    if tau > 500:
        break

# salvar o índice j de taupoints em que 'tau' fica maior que 10000
# esse índice é salvo porque os dados tais que tau>10000 serão descartados
for j, tau in enumerate(taupoints):
     if tau > 1e4:
        break

taupoints = taupoints[:j].copy()  # descarta os 'tau' maiores que 10000

for h0 in [-1.1, 0.1]:  # para cada caso considerado, plotar os gráficos
    # ler os valores previamente calculados para o trabalho, excluindo os dados para os quais tau>10000
    Wpoints = np.loadtxt(f'W beta={beta} h0={h0}', complex)[:j]
    WQSpoints = np.loadtxt(f'W_ad beta={beta} h0={h0}', complex)[:j]
    n = len(Wpoints)

    Wex = np.real(Wpoints - WQSpoints)

    Dpoints = np.loadtxt(f'D beta={beta} h0={h0}', complex)[:j]
    D2points = np.loadtxt(f'D2 beta={beta} h0={h0}', complex)[:j]

    # faz o fit linear na escala logarítmica
    # fit1 é um array bidimensional cuja primeira entrada dá -n, na notação do relatório
    # cov1 é a matriz de covariância do fit
    fit1, cov1 = np.polyfit(np.log(taupoints[i:]), np.log(Wex), deg=1, cov=True)
    print(f'For h0={h0}, Wex fit...')
    print(ufloat(fit1[0], np.sqrt(cov1[0, 0])))  # imprime o valor de -n com sua incerteza
    W_fit1 = taupoints**fit1[0] * np.exp(fit1[1])  # calcula a reta do ajuste para plotar no gráfico

    # plotar Wex em escala logarítmica
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


    # plotar W em escala linear
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

    # plotar D[rho||rho_eq] e Wex juntos em escala logarítmica
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

    # Ajuste para D[rho||rho_qs] (mesmo processo que o ajuste feito para Wex)
    fit1, cov1 = np.polyfit(np.log(taupoints[i:]), np.log(np.real(D2points[i:])), deg=1, cov=True)
    print(f'For h0={h0}, D2 fit...')
    print(ufloat(fit1[0], np.sqrt(cov1[0, 0])))
    D2_fit1 = taupoints**fit1[0] * np.exp(fit1[1])


    # plotar D[rho||rho_qs] em escala logarítmica
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

    # plot D[rho||rho_qs] e Wex (e seus ajustes) juntos em escala logarítmica
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

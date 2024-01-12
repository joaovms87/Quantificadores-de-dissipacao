import numpy as np
import scipy.linalg as sl


beta = 0.1  # parâmetro de temperatura
DELTA = 2  # variação total do parâmetro externo B


def B(t):  # protocolo para variação de B
    if t > 0:
        return B0 + DELTA/tau*t
    else:
        return B0


def theta(B):  # um parâmetro útil dependente de B
    return 0.5*np.arctan2(1, B)


def E(B):  # valor absoluto das auto-energias do sistema
    return np.sqrt(B**2 + 1)


def f(r, t):  # vetor de funções no lado direito do sistema de quações diferenciais (vetor f segundo notação do apêndice)
    p00, p01, p11 = r  # pega vetor r das incógnitas (elementos da matriz densidade) e separa cada incógnita
    dif = p01 - np.conj(p01)
    f00 = -1j * (-dif)  # primeira componente do vetor f
    f01 = -1j * (2*B(t)*p01 + (p11-p00))  # segunda componente do vetor f
    f11 = -1j * dif  # terceira componente do vetor f
    return np.array([f00, f01, f11], complex)  # retorna o vetor f como um array


def ln_partition(t):  # log natural da função de partição
    e = E(B(t))
    if beta*e < 100:  # neste caso, precisamos considerar ambos os termos da função de partição
        return np.log(np.exp(-beta*e) + np.exp(beta*e))
    else:  # neste caso, podemos fazer uma aproximação (desprezando o menor termo da função de partição)
        return beta*e
    # essa aproximação ajuda porque não preicsamos calcular exp(beta*e) se o expoente é muito grande.
    # calcular exp(beta*e) para expoentes muito grandes causaria overflow


def rho_qs(t):  # matriz densidade no tempo t se a evolução é quase-estática
    e, th, lnZ0 = E(B0), theta(B(t)), ln_partition(0)
    p = np.empty([2, 2], complex)
    # motar matriz segundo eqs. (9) e (26) do relatório novo
    p[0, 0] = np.exp(-beta * e - lnZ0) * (np.cos(th) ** 2) + np.exp(beta * e - lnZ0) * (np.sin(th)) ** 2
    p[0, 1] = np.cos(th) * np.sin(th) * (np.exp(-beta * e - lnZ0) - np.exp(beta * e - lnZ0))
    p[1, 0] = np.conj(p[0, 1])
    p[1, 1] = np.exp(-beta * e - lnZ0) * (np.sin(th) ** 2) + np.exp(beta * e - lnZ0) * (np.cos(th)) ** 2
    return p


def rho_eq(t):  # estado de Gibbs no tempo t
    e, th, lnZ = E(B(t)), theta(B(t)), ln_partition(t)
    p = np.empty([2, 2], complex)
    # montar matriz segundo eqs. (7) e (26) do relatório novo
    p[0, 0] = np.exp(-beta * e - lnZ) * (np.cos(th) ** 2) + np.exp(beta * e - lnZ) * (np.sin(th)) ** 2
    p[0, 1] = np.cos(th) * np.sin(th) * (np.exp(-beta * e - lnZ) - np.exp(beta * e - lnZ))
    p[1, 0] = np.conj(p[0, 1])
    p[1, 1] = np.exp(-beta * e - lnZ) * (np.sin(th) ** 2) + np.exp(beta * e - lnZ) * (np.cos(th)) ** 2
    return p


def D(p1, p2):  # entropia relativa entre as matrizes p1 e p2
    ln1, ln2 = sl.logm(p1), sl.logm(p2)  # calcula os logaritmos das matrizes
    prod1, prod2 = np.matmul(p1, ln1), np.matmul(p1, ln2)  # calcula os produtos matriciais necessários
    return np.trace(prod1 - prod2)  # calcula o traço e retorna a entropia relativa


def Ham(t):  # matriz do Hamiltoniano no tempo t, segundo eq. (25) do relatório novo
    H_t = np.ones([2, 2], complex)
    H_t[0, 0], H_t[1, 1] = B(t), -B(t)
    return H_t


def W(p):  # calcula o trabalho total feito no processo de duração 'tau'
    Ef = np.trace(np.matmul(p, Ham(tau)))  # energia final média calculada segundo eq. (4) do relatório
    return np.real(Ef - E0)  # diferença entre energia final e inicial (a qual é calculada em momento oportuno)


def W_qs(t):  # trabalho feito até o tempo t se a evolução é quase-estática
    Ef = np.trace(np.matmul(rho_qs(t), Ham(t)))  # mesmo processo da função anterior
    return np.real(Ef - E0)


# As 2 próximas funções são referentes à versão adaptativa do método (ver apêndice correspondente do relatório)
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


delta = 1e-10       # precisão exigida por unidade de tempo
for B0 in [-2, -1]:  # resolver as equações para cada caso considerado
        Bf = B0 + DELTA

        # lista de durações 'tau' que serão investigadas
        taupoints = np.logspace(-2, 5, 200)
        l = len(taupoints)
        # criar arrays referentes ao trabalho total e entropias relativas finais
        # cada entrada de um dado array se refere ao processo cuja duração 'tau' é a da posição correspondente em taupoints
        Wpoints = np.empty(l, complex)    # abrigará os valores do trabalho total
        Dpoints = np.empty(l, complex)    # abrigará os valores de D[rho(tau)||rho_eq(tau)]
        D2points = np.empty(l, complex)   # abrigará os valores de D[rho(tau)||rho_qs(tau)]
        # computar essas quantidades para cada 'tau'

        # array de incógnitas [rho00, rho01, rho11] em t=0; necessário para aplicar o método Runge-Kutta
        r0 = np.array([rho_eq(0)[0, 0], rho_eq(0)[0, 1], rho_eq(0)[1, 1]], complex)
        for j, tau in enumerate(taupoints):
            t = 0  # começar no tempo inicial (t=0)
            N = 100
            u = tau/N  # intervalo de discretização inicial; será alterado pela versão adaptativa do método

            # Resolver sistema de equações diferencias pelo método Runge-Kutta adaptativo de 4a ordem 
            r = r0.copy()  # array de incógnitas
            while t<tau:
                # dois passos de tamanho u
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
                # um passo de tamanho 2u
                k1 = 2*u*f(r, t)
                k2 = 2*u*f(r+0.5*k1, t+u)
                k3 = 2*u*f(r+0.5*k2, t+u)
                k4 = 2*u*f(r+k3, t+2*u)
                r2 = r + (k1+2*k2+2*k3+k4)/6

                p = ratio(r1, r2)
                if p>=1:  # precisão maior que a exigida
                    r = r1  # manter o resultado mais preciso
                    t = t+2*u  # passar para o próximo t
                # se p>=1 não é satisfeito, repetir tudo sem passar para o próximo t
                u = new_u(u, p)  # muda u independente de p ser maior ou menor que 1


            # construir a matriz densidade final (tempo 'tau') para o processo de duração 'tau'
            rho = np.empty([2, 2], complex)
            rho[0, 0], rho[0, 1], rho[1, 1] = r  # as entradas são calculadas usando o resultado r obtido pelo método 
            rho[1, 0] = np.conj(rho[0, 1])

            # calcular energia incial (segundo eq. (4) do relatório) para computar o trabalho
            E0 = np.trace(np.matmul(rho_eq(0), Ham(0)))
            # computar o trabalho feito no processo de duração 'tau' e salvar
            Wpoints[j] = W(rho)
            # computar a entropia relativa D[rho||rho_eq] final e salvar
            Dpoints[j] = D(rho, rho_eq(tau))
            # computar a entropia relativa D[rho||rho_qs] final e salvar
            D2points[j] = D(rho, rho_qs(tau))
            # computar W_qs(tau) e salvar
            WQSpoints = W_qs(tau)*np.ones(l, complex)

        # salvar tudo
        np.savetxt(f'W; B0={B0}; beta={beta}', Wpoints)
        np.savetxt(f'W_ad; B0={B0}; beta={beta}', WQSpoints)
        np.savetxt(f'D; B0={B0}; beta={beta}', Dpoints)
        np.savetxt(f'D2; B0={B0}; beta={beta}', D2points)

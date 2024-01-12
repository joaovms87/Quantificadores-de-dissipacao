import numpy as np
import scipy.linalg as sl


DELTA = 2  # variação total de gamma1
beta = 0.1 # parâmetro de temperatura


def h(t):  # protocolo de variação de gamma1
    if t==0:
        return h_0
    else:
        return h_0 + DELTA/tau*t


# auto-energias como funções de gamma1 = h
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


def partition(t):  # função de partição no tempo t
    g = h(t)
    e1, e2, e3, e4, e5, e6, e7, e8 = E1(g), E2(g), E3(g), E4(g), E5(g), E6(g), E7(g), E8(g)
    return np.exp(-beta*e1) + np.exp(-beta*e2) + np.exp(-beta*e3) + np.exp(-beta*e4) + \
           np.exp(-beta*e5) + np.exp(-beta*e6) + np.exp(-beta*e7) + np.exp(-beta*e8)


def rho_eq(t):  # estado de Gibbs no tempo t
    Z, g = partition(t), h(t)
    e1, e2, e3, e4, e5, e6, e7, e8 = E1(g), E2(g), E3(g), E4(g), E5(g), E6(g), E7(g), E8(g)

    # auto-vetores normalizados do Hamiltoniano:
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

    # contruir a matriz de rho_eq segundo a eq. (7) do novo relatório
    p = np.exp(-beta*e1)*np.outer(v1, v1)
    p+= np.exp(-beta*e2)*np.outer(v2, v2)
    p+= np.exp(-beta*e3)*np.outer(v3, v3)
    p+= np.exp(-beta*e4)*np.outer(v4, v4)
    p+= np.exp(-beta*e5)*np.outer(v5, v5)
    p+= np.exp(-beta*e6)*np.outer(v6, v6)
    p+= np.exp(-beta*e7)*np.outer(v7, v7)
    p+= np.exp(-beta*e8)*np.outer(v8, v8)

    return p/Z


def rho_qs(t):  # matriz densidade proveniente de evolução quase-estática
    Z, g, g0 = partition(0), h(t), h_0  # h_0 é o h no tempo 0
    e10, e20, e30, e40, e50, e60, e70, e80 = E1(h_0), E2(h_0), E3(h_0), E4(h_0), E5(h_0), E6(h_0), E7(h_0), E8(h_0)

    # auto-vetores do Hamiltoniano no tempo t
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

    # construir a matriz densidade rho_qs conforme eq. (9) do novo relatório
    p = np.exp(-beta*e10)*np.outer(v1, v1)
    p+= np.exp(-beta*e20)*np.outer(v2, v2)
    p+= np.exp(-beta*e30)*np.outer(v3, v3)
    p+= np.exp(-beta*e40)*np.outer(v4, v4)
    p+= np.exp(-beta*e50)*np.outer(v5, v5)
    p+= np.exp(-beta*e60)*np.outer(v6, v6)
    p+= np.exp(-beta*e70)*np.outer(v7, v7)
    p+= np.exp(-beta*e80)*np.outer(v8, v8)

    return p / Z


def Ham(t):  # Hamiltoniano no tempo t
    g = h(t)
    H_t = np.zeros([8, 8], complex)
    
    # construir a matriz do Hamiltoniano conforme eq. (28) do relatório
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


def D(p1, p2):  # entropia relativa entre as matrizes p1 e p2
    ln1, ln2 = sl.logm(p1), sl.logm(p2)  # calcula os logaritmos das matrizes
    prod1, prod2 = np.matmul(p1, ln1), np.matmul(p1, ln2)  # calcula cada um dos termos usando produto de matrizes
    return np.trace(prod1-prod2)  # calcula o traço e retorna a entropia relativa


def W(p):  # trabalho total no processo
    Ef = np.trace(np.matmul(p, Ham(tau)))  # Energia final média calculada segundo a eq. (4) do relatório
    return np.real(Ef - E0)  # diferença entre energia final e inicial. Desconsidera parte imaginária residual, se houver


def W_qs(t):  # calcula o trabalho feito até o tempo t se a evolução é quase-estática
    Ef = np.trace(np.matmul(rho_qs(t), Ham(t)))  # energia média no tempo desejado, para evolução quase-estática
    return np.real(Ef - E0)  # diferença entre energia final e inicial


def f(r, t):  # retorna o vetor de funções no lado direito das equações simultâneas, conforme notação do apêndice B
    g = h(t)  # parâmetro externo no tempo t

    # o trecho que segue pega o vetor de incógnitas (elementos da matriz densidade) e reconstitui a matriz densidade
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

    # calcular derivada da matriz densidade, conforme equação (23) do novo relatório
    d = - 1j * (Ham(t)@p - p@Ham(t))

    # transformar os elementos relevantes em um vetor (o vetor f no instante t) para poder aplicar o método Runge-Kutta
    new_f = np.concatenate((d[0], d[1, 1:], d[2, 2:], d[3, 3:], d[4, 4:], d[5, 5:], d[6, 6:], d[7, 7:]), axis=0)

    return new_f


# As próximas duas funções são para a versão adaptativa do método. Ver seção correspondente no relatório
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


# lista das durações 'tau' que serão investigadas
taupoints = np.logspace(-3, 5, 200)
l = len(taupoints)
# criar arrays referentes ao trabalho total e entropias relativas finais
# cada entrada de um determinado array se refere ao processo cuja duração 'tau' é a da posição correspondente em taupoints
Wpoints = np.empty(l, complex)    # abrigará os valores do trabalho total
Dpoints = np.empty(l, complex)    # abrigará os valores de D[rho(tau)||rho_eq(tau)]
D2points = np.empty(l, complex)   # abrigará os valores de D[rho(tau)||rho_qs(tau)]
# computar essas quantidade para cada 'tau'

delta = 1e-10  # precisão exigida por unidade de tempo

for h_0 in [-1.1, 0.1]:
    rho0 = rho_eq(0)  # rho no instante inicial
    # vetor de elementos de rho (incógnitas a serem encontradas pelo método) no instante t=0.
    # é necessário criar esse vetor a partir de rho0 para poder aplicar o método Runge-Kutta
    r0 = np.concatenate((rho0[0].copy(), rho0[1, 1:].copy(), rho0[2, 2:].copy(), rho0[3, 3:].copy(), rho0[4, 4:].copy(),
                    rho0[5, 5:].copy(), rho0[6, 6:].copy(), rho0[7, 7:].copy()), axis=0)

    # resolver o sistema de equações para cada duração 'tau'
    for j, tau in enumerate(taupoints):
        t = 0  # começar no tempo inicial (zero)
        N = 100
        u = tau/N  # tempo de discretização inicial. Será alterado pela versão adaptativa do método

        # Resolver o sistema de equações diferenciais usando o método Runge-Kutta adaptativo de 4a ordem
        r = r0.copy()  # vetor de incógnitas para aplicar o método
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
                r = r1  # mantém o resultado mais preciso
                t = t+2*u  # passa ao próximo t
            # se p>=1 não é satisfeito, repetir tudo sem ir para o próximo t
            u = new_u(u, p)  # muda u independente de se p>=1 ou não

        # construir (a partir do vetor r de incógnitas) a matriz densidade final para o processo de duração 'tau'
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

        # energia inicial (para calcular o trabalho):
        E0 = np.trace(np.matmul(rho0, Ham(0)))
        # calcular o trabalho feito no processo de duração 'tau' e salvar
        Wpoints[j] = W(rho)
        # calcular a entropia relativa final para o processo de duração 'tau' e salvar
        Dpoints[j] = D(rho, rho_eq(tau))
        # calcular D[rho(tau)||rho_qs(tau)] para o processo de duração 'tau' e salvar
        D2points[j] = D(rho, rho_qs(tau))

    # calcular W_qs(tau):
    # (ele é independente de 'tau', então calculamos apenas para o último 'tau')
    WQSpoints = W_qs(tau)*np.ones(l, complex)

    # salvar tudo
    np.savetxt(f'W beta={beta} h0={h_0}', Wpoints)
    np.savetxt(f'W_ad beta={beta} h0={h_0}', WQSpoints)
    np.savetxt(f'D beta={beta} h0={h_0}', Dpoints)
    np.savetxt(f'D2 beta={beta} h0={h_0}', D2points)

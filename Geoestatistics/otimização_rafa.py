"""
Otimização Iterativa com Derivadas de Ordem Superior.
Objetivo: Estimar os parâmetros da função Matérn (phi1, phi2, phi3),

"""
import numpy as np
import scipy.linalg as la
import scipy.stats as stats
import scipy.spatial.distance as sp_dist
import scipy.special as sp
import matplotlib.pyplot as plt

import func_aux

# =====================================================================
# 1. LEITURA DOS DADOS
# =====================================================================
x_file = "X.txt"
wypych_file = "Dados_Wypych.txt"

X, Y, gr = func_aux.read_dados_wypych(x_file, wypych_file)

X = np.array(X)
Y = np.array(Y)
gr = np.array(gr)

if Y.ndim == 1:
    Y = Y.reshape(-1, 1)

n = gr.shape[0]
I = np.identity(n)


# =====================================================================
# 2. CONFIGURAÇÃO INICIAL E MATRIZ DE DISTÂNCIAS
# =====================================================================
# Usando o prefixo sp_dist. para a distância
H = sp_dist.cdist(gr, gr)
H_safe_global = np.where(H == 0, 1e-10, H)

# Chutes iniciais
phi1 = 0.22  
phi2 = 0.15  
phi3 = 110.0 
k = 0.5      


# =====================================================================
# 3. ALGORITMO EM (t-Student Espacial)
# =====================================================================

def em_tstudent_spatial(X, Y, gr, theta_init, H, k, gl=4, max_iter=100, tol=1e-4):
    """
    Estimação de parâmetros espaciais robustos (t-Student) via Algoritmo EM.
    Passo Fischer Scoring para atualização de phi1 e phi2, e Newton 1D para phi3. (Utização de dK penas)
    """
    # Desempacotando parâmetros iniciais
    beta = np.array(theta_init[:-3]).reshape(-1, 1)
    phi1, phi2, phi3 = theta_init[-3:]
    
    n = len(Y)
    I = np.eye(n)
    
    print("Iniciando otimização EM (t-Student)...\nValores iniciais: beta (OLS), k, phi1,2 e 3 =", beta.flatten(), k, phi1, phi2, phi3)
    
    for it in range(1, max_iter + 1):
        
        # 1. Construção das Matrizes Espaciais 
        Rf3 = func_aux.matern_correlation(H, phi3, k)
        Sigma = phi1 * I + phi2 * Rf3 # Matriz de covariância -> Sigma = phi1 * I + phi2 * R(phi3)

        # Cholesky 
        try:
            c, lower = la.cho_factor(Sigma)
            Sigma_inv = la.cho_solve((c, lower), I) # Inversa via Cholesky -> Sigma_inv = Sigma^-1
        except la.LinAlgError:
            print(f"Aviso: Matriz instável na iteração {it}. Usando pseudo-inversa de fallback.")
            Sigma_inv = la.pinv(Sigma) # Pseudo-inversa -> Sigma_inv = Sigma^+
            
        # 2. Atualização do beta 
        X_invS = X.T @ Sigma_inv # Produto transposto -> X_invS = X^T * Sigma^-1
        beta_new = la.solve(X_invS @ X, X_invS @ Y) # Estimador de GLS -> beta_new = (X^T * Sigma^-1 * X)^-1 * (X^T * Sigma^-1 * Y)

        # 3. Cálculo do peso latente da t-Student (v)
        r = Y - X @ beta_new # Resíduo -> r = Y - X * beta
        u = (r.T @ Sigma_inv @ r).item() # Distância de Mahalanobis -> u = r^T * Sigma^-1 * r
        v = (gl + n) / (gl + u)          # Peso robusto -> v = (gl + n) / (gl + u)

        # 4. Derivadas Parciais (Matrizes d_phi)
        d_phi1 = I # Derivada em relação a phi1 -> d_phi1 = d(Sigma)/d(phi1) = I
        d_phi2 = Rf3 # Derivada em relação a phi2 -> d_phi2 = d(Sigma)/d(phi2) = Rf3

        # Chamando funções especiais do scipy e de func_aux
        dK_val = func_aux.dK(H, phi3, k)
        H_phi3_k1 = np.where(H > 0, (H / phi3)**(k + 1), 0)
        coef_M = 1.0 / ((2**(k - 1)) * sp.gamma(k))

        M = k * d_phi2 + coef_M * (H_phi3_k1 * dK_val)
        d_phi3 = phi2 * (-(1.0 / phi3) * M) # Derivada em relação a phi3 -> d_phi3 = phi2 * [d(Rf3)/d(phi3)]

        # 5. PASSO M: Fisher Scoring / Gradiente
        invS_r = Sigma_inv @ r # Resíduo ponderado -> invS_r = Sigma^-1 * r
        invS_d1 = Sigma_inv @ d_phi1 # Resulta no próprio Sigma_inv
        invS_d2 = Sigma_inv @ d_phi2 # Sigma_inv @ Rf3

        # Matriz A (2x2) baseada no traço (np.sum(A * B.T) é uma forma rápida de calcular tr(A@B))
        a11 = np.sum(invS_d1 * invS_d1.T) 
        a12 = np.sum(invS_d1 * invS_d2.T) 
        a22 = np.sum(invS_d2 * invS_d2.T) 

        A_2x2 = np.array([
            [a11, a12],
            [a12, a22]
        ])

        # Vetor Score (S)
        S1 = v * (invS_r.T @ d_phi1 @ invS_r).item() # Score phi1 -> S1 = v * (r^T * Sigma^-1 * I * Sigma^-1 * r)
        S2 = v * (invS_r.T @ d_phi2 @ invS_r).item() # Score phi2 -> S2 = v * (r^T * Sigma^-1 * Rf3 * Sigma^-1 * r)
        S_2x2 = np.array([S1, S2]) # Score total -> S = [S1, S2]

        # Matriz de Informação Esperada (A)
        invS_d1 = Sigma_inv @ d_phi1
        invS_d2 = Sigma_inv @ d_phi2
        invS_d3 = Sigma_inv @ d_phi3

        # Atualização linear dos componentes de variância
        Fi_2x2 = la.solve(A_2x2, S_2x2) 
        phi1_new = Fi_2x2[0]
        phi2_new = Fi_2x2[1]

        # 6. Atualização de phi3 (Fisher Scoring isolado / Newton 1D)
        invS_d3 = Sigma_inv @ d_phi3
        
        # Score de phi3 (U_3)
        U_phi3 = -0.5 * np.trace(invS_d3) + 0.5 * v * (invS_r.T @ d_phi3 @ invS_r).item()
        
        # Informação de Fisher Esperada para phi3 (I_33) em substituição à segunda derivada exata
        I_phi3 = 0.5 * np.sum(invS_d3 * invS_d3.T)
        
        # Passo de atualização não-linear
        phi3_new = phi3 + (U_phi3 / I_phi3)

        # 7. Verificação de Convergência
        theta_old = np.concatenate([beta.flatten(), [phi1, phi2, phi3]])
        theta_new = np.concatenate([beta_new.flatten(), [phi1_new, phi2_new, phi3_new]])

        erro = np.linalg.norm(theta_old - theta_new) / np.linalg.norm(theta_old) # Erro relativo -> erro = ||theta_old - theta_new|| / ||theta_old||

        # Atualiza variáveis para a próxima iteração
        beta = beta_new
        phi1, phi2, phi3 = max(1e-5, phi1_new), max(1e-5, phi2_new), max(1e-5, phi3_new)
        
        if it == 1 or it % 5 == 0 or erro < tol:
            print(f"Iter {it}: Erro = {erro:.6f} | beta = {beta_new.flatten()}, phi1 = {phi1:.4f}, phi2 = {phi2:.4f}, phi3 = {phi3:.4f}")
            
        if erro < tol:
            print(f"\n=== Convergência atingida em {it} iterações! ===")
            return {
                "beta": beta.flatten(),
                "phi1": phi1,
                "phi2": phi2,
                "phi3": phi3,
                "Sigma": Sigma,
                "iteracoes": it,
                "erro_final": erro
            }

    print("\n=== Não Convergência ! ===")
    return {
        "beta": beta.flatten(),
        "phi1": phi1,
        "phi2": phi2,
        "phi3": phi3,
        "Sigma": Sigma,
        "iteracoes": it,
        "erro_final": erro
    }


# =====================================================================
# 5. EXECUÇÃO PRINCIPAL
# =====================================================================
while True:
    # 1. Cálculo do Resíduo Inicial (OLS)
    beta_ols = np.linalg.solve(X.T @ X, X.T @ Y) # Estimador OLS clássico -> beta_ols = (X^T * X)^-1 * (X^T * Y)
    r_inicial = Y - X @ beta_ols # r = Real (Y) - Previsto (X * beta)

    # 3. INTERAÇÃO COM O USUÁRIO (PERGUNTA SE QUER VER O GRÁFICO)
    print( "="*50)
    resposta_visual = input("Deseja abrir o gráfico para validar visualmente os chutes iniciais? (s/n): ").strip().lower()
    if resposta_visual == 's':
        phi1, phi2, phi3, k = func_aux.ver_kappa(H, r_inicial, phi1, phi2, phi3, k)
        if k > 1:
            print("\nAtenção: Kappa > 1 pode levar a problemas de convergência devido à super-suavização. Considere usar k <= 1 para dados reais.")
            k = input("Digite um valor para Kappa (sugestão: 0.5 para fenômenos ruidosos, 1.0 para mais suaves): ")
    else:
        print("\nPulando gráfico. Prosseguindo com os chutes automáticos...")
    print("="*50)

    em_resultados = em_tstudent_spatial(X, Y, gr, theta_init=[*beta_ols.flatten(), phi1, phi2, phi3], H=H, k=k)
    print("\nem_resultados['beta'] =", em_resultados["beta"])

    if input("\nDeseja visualizar os resíduos e finalizar loop? (s/n) ").strip().lower() == 's':
        # Resíduo EM (Marginal)
        beta_final = em_resultados["beta"].reshape(-1, 1)
        r_final = Y - X @ beta_final
        
        # Resíduo EM (Decorrelacionado)
        Sigma_final = em_resultados["Sigma"]
        
        # Usando Decomposição Cholesky para calcular: L^-1 * r_final (raiz quadrada inversa de Sigma). Desvio padrão espacialmente ajustado para Matriz.
        L = la.cholesky(Sigma_final, lower=True)
        r_decorrelacionado = la.solve(L, r_final) # Remover o efeito da escala ou da variância de uma única variável para compará-la, o Escore-Z
        
        #func_aux.plot_residuo(r_inicial, r_final, r_decorrelacionado, gr)
        
        break

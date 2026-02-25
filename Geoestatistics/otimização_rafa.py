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
import EM_Matern

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
# 3. EXECUÇÃO PRINCIPAL
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

    em_resultados = EM_Matern.em_tstudent_Fischer(X, Y, gr, theta_init=[*beta_ols.flatten(), phi1, phi2, phi3], H=H, k=k)
    em_resultado_NRE = EM_Matern.em_tstudent_NRExato(X, Y, gr, theta_init=[*beta_ols.flatten(), phi1, phi2, phi3], H=H, k=k)
    print("\nem_resultados['phis'] =", em_resultados["phi1"], em_resultados["phi2"], em_resultados["phi3"])
    print("\nem_resultado_NRE['phis'] =", em_resultado_NRE["phi1"], em_resultado_NRE["phi2"], em_resultado_NRE["phi3"])

    if input("\nDeseja visualizar os resíduos e finalizar loop? (s/n) ").strip().lower() == 's':
        # Resíduo EM (Marginal)
        beta_final = em_resultados["beta"].reshape(-1, 1)
        r_final = Y - X @ beta_final
        
        # Resíduo EM (Decorrelacionado)
        Sigma_final = em_resultados["Sigma"]

        if(input("\nChamar Resumo dos Erros? (s/n) ").strip().lower() == 's'):
            kcx = func_aux.valid_cruzada(Y, Sigma_final, X=X)
            df_erros, resumo, ea = func_aux.relatorio_erros(kcx)
            print("\nRelatório de Erros:", df_erros)
        
        # Usando Decomposição Cholesky para calcular: L^-1 * r_final (raiz quadrada inversa de Sigma). Desvio padrão espacialmente ajustado para Matriz.
        L = la.cholesky(Sigma_final, lower=True)
        r_decorrelacionado = la.solve(L, r_final) # Remover o efeito da escala ou da variância de uma única variável para compará-la, o Escore-Z
        
        func_aux.plot_residuo(r_inicial, r_final, r_decorrelacionado, gr)

        break

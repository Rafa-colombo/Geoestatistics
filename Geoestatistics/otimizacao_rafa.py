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

w_file = "dados_exp.txt"# retirar gr's e Y (respectivamente colunas 1, 2, 3); retirar covariaveis (colunas 4 e 5) -> começando do 0

X, Y, gr = func_aux.data_to_var(w_file) # Ao passar os dados, certifique-se de que indice 0 exista(ele sera ignorado). Ordem: gr (1 e 2), Y (3), covariaveis (4 e 5). 

if Y.ndim == 1:
    Y = Y.reshape(-1, 1)

n = gr.shape[0]
I = np.identity(n)

print("\nDados carregados com sucesso. Dimensões: X =", X.shape, ", Y =", Y.shape, ", gr =", gr.shape)
print("Primeiras 5 linhas de X:\n", X[:5])
print("Primeiras 5 linhas de Y:\n", Y[:5])
print("Primeiras 5 linhas de gr:\n", gr[:5])
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
gl = 4

# =====================================================================
# 3. EXECUÇÃO PRINCIPAL
# =====================================================================
while True:
    # 1. Cálculo do Resíduo Inicial (OLS)
    beta_ols = np.linalg.solve(X.T @ X, X.T @ Y) # Estimador OLS clássico -> beta_ols = (X^T * X)^-1 * (X^T * Y)
    r_inicial = Y - X @ beta_ols # r = Real (Y) - Previsto (X * beta)

    # 3. INTERAÇÃO COM O USUÁRIO (PERGUNTA SE QUER VER O GRÁFICO)
    print( "="*50)
    if (input("Deseja abrir o gráfico para validar visualmente os chutes iniciais? (s/n): ").strip().lower()) == 's':
        phi1, phi2, phi3, k, gl = func_aux.update_values(H, r_inicial, phi1, phi2, phi3, k, gl)
        if k > 1:
            print("\nAtenção: Kappa > 1 pode levar a problemas de convergência devido à super-suavização. Considere usar k <= 1 para dados reais.")
            k = input("Digite um valor para Kappa (sugestão: 0.5 para fenômenos ruidosos, 1.0 para mais suaves): ")
    else:
        print("\nPulando gráfico. Prosseguindo com os chutes automáticos...")
    print("="*50)

    em_resultados = EM_Matern.fit_tstudent_fisher(X, Y, gr, theta_init=[*beta_ols.flatten(), phi1, phi2, phi3], H=H, k=k, gl=gl)
    em_resultado_NRE = EM_Matern.fit_tstudent_exact_nr(X, Y, gr, theta_init=[*beta_ols.flatten(), phi1, phi2, phi3], H=H, k=k, gl=gl)
    print("\nem_resultados['phis'] =", em_resultados["phi1"], em_resultados["phi2"], em_resultados["phi3"])
    print("\nem_resultado_NRE['phis'] =", em_resultado_NRE["phi1"], em_resultado_NRE["phi2"], em_resultado_NRE["phi3"])

    func_aux.interactive_stats_view(Y, X, em_resultados, r_inicial, H, gr, k)

    if input("Deseja finalizar loop? (s/n) ").strip().lower() == 's':
        break

"""
Otimização Iterativa com Derivadas de Ordem Superior.
Objetivo: Estimar os parâmetros da função Matérn (phi1, phi2, phi3),

"""
from util import data_to_var, interactive_stats_view
import EM_Matern

# =====================================================================
# 1. LEITURA DOS DADOS
# =====================================================================

w_file = "dados_exp.txt"

X, Y, gr, H, beta_ols, r_inicial = data_to_var(w_file) # Caso queira passar os dados sem a coluna de índice, use data_to_var(w_file, ind_0=False)

print("\nDados carregados com sucesso. Dimensões: X =", X.shape, ", Y =", Y.shape, ", gr =", gr.shape)
print("Primeiras 5 linhas de X:\n", X[:5])
print("Primeiras 5 linhas de Y:\n", Y[:5])
print("Primeiras 5 linhas de gr:\n", gr[:5])

# =====================================================================
# 2. Chutes iniciais
# =====================================================================

phi1 = 0.22  
phi2 = 0.15  
phi3 = 110.0 
k = 0.5      
gl = 4

theta_init=[*beta_ols.flatten(), phi1, phi2, phi3] # Beta como vetor 1D [b0, b1, b2]

# =====================================================================
# 3. EXECUÇÃO PRINCIPAL
# =====================================================================
while True:

    print("="*50)

    # em_resultados = EM_Matern.fit_tstudent_fisher(X, Y, gr, H, k, gl, theta_init=theta_init)
    em_resultados = EM_Matern.fit_tstudent_fisher(X, Y, gr, H, k, gl, beta_ols=beta_ols)
    em_resultados_comOtimizacao = EM_Matern.fit_tstudent_fisher(X, Y, gr, H, k, gl, beta_ols=beta_ols, gl_optimize=True)
    em_resultado_NRE = EM_Matern.fit_tstudent_exact_nr(X, Y, gr, H, k, gl, theta_init=theta_init)
    print("em_resultados['phis'] =", em_resultados["phi1"], em_resultados["phi2"], em_resultados["phi3"])
    print("em_resultados_comOtimizacao['phis'] =", em_resultados_comOtimizacao["phi1"], em_resultados_comOtimizacao["phi2"], em_resultados_comOtimizacao["phi3"])
    print("em_resultado_NRE['phis'] =", em_resultado_NRE["phi1"], em_resultado_NRE["phi2"], em_resultado_NRE["phi3"])

    interactive_stats_view(Y, X, em_resultados, r_inicial, H, gr, k)

    if input("Deseja finalizar loop? (0/1) ")== '1':
        break

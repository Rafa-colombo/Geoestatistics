import pandas as pd
import numpy as np
import scipy.special as sp
import scipy.linalg as la
import scipy.optimize as opt
import scipy.spatial.distance as sp_dist
import matplotlib.pyplot as plt


def read_data(filename):
    """Lê arquivo tentando diferentes separadores."""
    for sep in ["\t", ",", "\s+"]:
        try:
            df = pd.read_csv(filename, sep=sep, header=None, engine="python")
            return df # data frame do pandas
        except:
            continue
    raise ValueError(f"Não foi possível ler o arquivo {filename}")

def data_to_var(w_file, ind_0=True):
    """ Migrando tudo que abstrai dos dados diretos para essa função:
    - gr: coordenadas (indices 1 e 2)
    - Y: resposta (indices 3)
    - cov: covariáveis (indices 4 e 5)
    - X: matriz de delineamento (coluna de 1s + covariáveis)
    - H: matriz de distâncias entre os pontos (scipy.spatial.distance.cdist)
    - beta_ols: estimativa OLS para os parâmetros de tendência
    - r_inicial: resíduo inicial (Y - X @ beta_ols)
    """
    df = read_data(w_file) # data frame

    if ind_0:
        gr = df.iloc[:, 1:3].to_numpy() # Coordenadas (colunas 1 e 2)
        Y = df.iloc[:, 3].to_numpy().reshape(-1, 1) # Resposta (coluna 3)
        cov = df.iloc[:, 4:6].to_numpy() # Covariáveis (colunas 4 e 5)
    else: # Se os dados já vierem sem a coluna de índice (começando do indice 0)
        gr = df.iloc[:, 0:2].to_numpy() # Coordenadas (colunas 0 e 1)
        Y = df.iloc[:, 2].to_numpy().reshape(-1, 1) # Resposta (coluna 2)
        cov = df.iloc[:, 3:5].to_numpy() # Covariáveis (colunas 3 e 4)

    # 4. Construção da Matriz X (Intercepto + Covariáveis)
    n_linhas = cov.shape[0]                  # Descobre quantas linhas os dados têm
    coluna_uns = np.ones((n_linhas, 1))      # Cria uma coluna só com números 1
    X = np.hstack((coluna_uns, cov))         # Junta a coluna de uns com as covariáveis
    
    X = np.array(X)
    Y = np.array(Y)
    gr = np.array(gr)
    H = sp_dist.cdist(gr, gr) # Matriz de distâncias entre os pontos (scipy.spatial.distance.cdist)

    # Cálculo do Resíduo Inicial (OLS) -> parametros para o gráfico de semivariograma
    beta_ols = np.linalg.solve(X.T @ X, X.T @ Y) # Estimador OLS clássico -> beta_ols = (X^T * X)^-1 * (X^T * Y)
    r_inicial = Y - X @ beta_ols # r = Real (Y) - Previsto (X * beta)

    return X, Y, gr, H, beta_ols, r_inicial


def exponential_correlation(H, phi3): # Pagina 13, item a) Modelo Exponencial p(h)
    """Calcula a correlação Exponencial para semivariograma."""
    return np.exp(-H / phi3)

def gaussian_correlation(H, phi3): # Pagina 13, item b) Modelo Gaussiano p(h)
    """Calcula a correlação Gaussiana para semivariograma."""
    return np.exp(-(H / phi3)**2)

def matern_correlation(H, phi3, k): # Pagina 14, item c) Modelo da familia Matérn p(h)
    """Calcula a matriz de correlação Matérn."""
    # Evitar divisão por zero na diagonal principal
    H_safe = np.where(H > 0, H, 1e-10)
    uphi = H_safe / phi3 # Razão de distância -> u = h / phi3
    
    coef = 1.0 / ((2**(k - 1)) * sp.gamma(k)) # Constante de normalização -> coef = 1 / (2^(k-1) * Gamma(k))
    res = coef * (uphi**k) * sp.kv(k, uphi) # Função de correlação -> R(h) = coef * (u^k) * K_k(u)
    
    # VERIFICAÇÃO DE DIMENSÃO:
    if np.ndim(res) >= 2:
        # Se for matriz (usado no EM), preenchemos a diagonal
        np.fill_diagonal(res, 1.0) # Correlação unitária na diagonal -> R(0) = 1
    else:
        # Se for vetor (usado no gráfico), garantimos que onde H era 0, res é 1
        res = np.where(H == 0, 1.0, res)

    return res


def dK(H, phi3, k):
    """1ª Derivada de K_kappa em relação a u"""
    H_safe = np.where(H == 0, 1e-10, H)
    uphi = H_safe / phi3
    # Propriedade das funções de Bessel modificadas -> d/du K_k(u) = -1/2 * (K_{k-1}(u) + K_{k+1}(u))
    res = -1/2 * (sp.kv(k - 1, uphi) + sp.kv(k + 1, uphi)) 
    np.fill_diagonal(res, 0)
    return res

def dKK(H, phi3, k):
    """2ª Derivada de K_kappa em relação a u"""
    H_safe = np.where(H == 0, 1e-10, H)
    uphi = H_safe / phi3
    # Segunda derivada de Bessel -> d^2/du^2 K_k(u) = 1/4 * (K_{k-2}(u) + 2*K_k(u) + K_{k+2}(u))
    res = 1/4 * (sp.kv(k - 2, uphi) + 2 * sp.kv(k, uphi) + sp.kv(k + 2, uphi))
    np.fill_diagonal(res, 0)
    return res
    

def update_values(H, r_inicial, phi1=None, phi2=None, phi3=None, k=None, gl=None):
    """
    Funções de auteração de valores baseado no plot do gráfico.
    """
    
    while True:
        if phi1 is not None and phi2 is not None and phi3 is not None and k is not None and gl is not None:
            resp = input("\nDeseja testar novos valores no gráfico? (0/1): ")
        else:
            print("\nPor favor, insira os valores para phi1, phi2, phi3, k e gl.")
            resp = '1' # Força a entrada de parâmetros

        if resp == '0' and phi1 is not None and phi2 is not None and phi3 is not None and k is not None and gl is not None:
            plt.close('all')
            print("\nProsseguindo com a otimização...\n")
            return phi1, phi2, phi3, k, gl
        elif resp == '1':
            try:
                phi1 = float(input("Valor para Phi1 (Nugget): "))
                phi2 = float(input("Valor para Phi2 (Sill): "))
                phi3 = float(input("Valor para Phi3 (Range): "))
                k = float(input("Valor para Kappa: "))
                gl = float(input("Valor para Graus de Liberdade: "))
                plt.close('all') 
                plot_semivariogram_curves(H, r_inicial, phi1, phi2, phi3, k, plot_curves=True)

                print("\n--- AVALIAÇÃO VISUAL ---")
                print(f"1. phi1 (Nugget) = {phi1:.4f}")
                print(f"2. phi2 (Sill)   = {phi2:.4f}")
                print(f"3. phi3 (Range)  = {phi3:.4f}")
                print(f"4. Kappa         = {k:.4f}")
                print(f"5. Graus de Liberdade = {gl:.4f}")
            except ValueError:
                print("Entrada inválida! Digite apenas números.")
   
def plot_semivariogram_curves(H, r_inicial, phi1=None, phi2=None, phi3=None, k=None, plot_curves=True):
    """Gera o Semivariograma Empírico vs Teórico para avaliar os chutes."""

    dist_flat = H.flatten()

    res_diff = (r_inicial - r_inicial.T)**2 # Diferença quadrática -> res_diff = (e_i - e_j)^2
    variograma_exp = 0.5 * res_diff.flatten() # Semivariância pontual -> gamma_ij = 0.5 * (e_i - e_j)^2 -> definição clássica para cada par de pontos.
    
    num_lags = 20
    bins = np.linspace(0, np.max(H), num_lags + 1)
    
    dist_medias = []
    gama_medias = []

    for i in range(num_lags):
        mask = (dist_flat > bins[i]) & (dist_flat <= bins[i+1])
        if np.any(mask): 
            dist_medias.append(np.mean(dist_flat[mask])) 
            gama_medias.append(np.mean(variograma_exp[mask])) # Média por lag -> gamma(h) = 1/(2N) * sum((e_i - e_j)^2)
            
    dist_medias = np.array(dist_medias)
    gama_medias = np.array(gama_medias)

    # 2. Criar Curva Teórica
    h_teorico = np.linspace(0, np.max(H), 100) 

    if plot_curves:
        # --- Modelo Matérn ---
        correl_teorica_matern = matern_correlation(h_teorico, phi3, k)
        variograma_matern = phi1 + phi2 * (1 - correl_teorica_matern) # Pagina 14, item c) Modelo da familia Matérn y(h)
        
        # --- Modelo Exponencial ---
        correl_teorica_exponencial = exponential_correlation(h_teorico, phi3)
        variograma_exponencial = phi1 + phi2 * (1 - correl_teorica_exponencial) # Pagina 13, item a) y(h)
        
        # --- Modelo Gaussiano ---
        correl_teorica_gaussiana = gaussian_correlation(h_teorico, phi3)
        variograma_gaussiana = phi1 + phi2 * (1 - correl_teorica_gaussiana) # Pagina 13, item b) y(h)

    # 3. Plotagem em painel (2x2)
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    if plot_curves:
        fig.suptitle(f"Comparação de Modelos Teóricos (φ1={phi1}, φ2={phi2}, φ3={phi3})", fontsize=15)
    else:
        fig.suptitle("Semivariograma Empírico", fontsize=15)

    # --- Configuração base comum para todos os 4 subgráficos ---
    for ax in axs.flat:
        # Plot dos dados empíricos
        ax.scatter(dist_flat, variograma_exp, alpha=0.1, color='gray', label='Dados (Pares)')
        ax.scatter(dist_medias, gama_medias, color='black', s=50, zorder=3, label='Empírico (Médias)')
        # Plot do patamar
        if plot_curves:
            ax.axhline(y=phi1+phi2, color='purple', linestyle=':', alpha=0.6, label=f'Patamar')
        
        # Configurações de eixos e grid
        ax.set_xlabel("Distância (h)")
        ax.set_ylabel("Semivariância γ(h)")
        if plot_curves:
            ax.set_ylim(0, (phi1 + phi2) * 1.3)
        ax.grid(True, alpha=0.3)

    if plot_curves:
        # Gráfico 1: Exponencial (Top Left -> axs[0, 0])
        axs[0, 0].plot(h_teorico, variograma_exponencial, color='blue', lw=2, linestyle='--', label='Exponencial')
        axs[0, 0].set_title("Modelo Exponencial")
        axs[0, 0].legend()

        # Gráfico 2: Gaussiano (Top Right -> axs[0, 1])
        axs[0, 1].plot(h_teorico, variograma_gaussiana, color='green', lw=2, linestyle='-.', label='Gaussiano')
        axs[0, 1].set_title("Modelo Gaussiano")
        axs[0, 1].legend()

        # Gráfico 3: Matérn (Bottom Left -> axs[1, 0])
        axs[1, 0].plot(h_teorico, variograma_matern, color='red', lw=2.5, label=f'Matérn (k={k})')
        axs[1, 0].set_title("Modelo Matérn")
        axs[1, 0].legend()

        # Gráfico 4: Todos os modelos juntos (Bottom Right -> axs[1, 1])
        axs[1, 1].plot(h_teorico, variograma_exponencial, color='blue', lw=2, linestyle='--', label='Exponencial')
        axs[1, 1].plot(h_teorico, variograma_gaussiana, color='green', lw=2, linestyle='-.', label='Gaussiano')
        axs[1, 1].plot(h_teorico, variograma_matern, color='red', lw=2.5, label='Matérn')
        axs[1, 1].set_title("Comparação Conjunta")
        axs[1, 1].legend()
    else:
        # Set titles for empirical only
        axs[0, 0].set_title("Empirical Semivariogram")
        axs[0, 1].set_title("Empirical Semivariogram")
        axs[1, 0].set_title("Empirical Semivariogram")
        axs[1, 1].set_title("Empirical Semivariogram")
        for ax in axs.flat:
            ax.legend()

    # Ajusta o layout para que os títulos e legendas não fiquem sobrepostos
    plt.tight_layout()
    plt.show()
        

def plot_residuals(r_inicial, r_final, r_decorr, gr):
    """ VISUALIZAÇÃO ESPACIAL COMPARATIVA DOS RESÍDUOS """

    res_ini_flat = r_inicial.flatten()
    res_fin_flat = r_final.flatten()
    res_dec_flat = r_decorr.flatten()
    
    longitudes = gr[:, 1]
    latitudes = gr[:, 0]

    max_abs_res = max(np.max(np.abs(res_ini_flat)), 
                      np.max(np.abs(res_fin_flat)), 
                      np.max(np.abs(res_dec_flat)))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 7), sharey=True)

    # --- PLOT 1: Resíduos OLS ---
    ax1.scatter(longitudes, latitudes, c=res_ini_flat, cmap='RdBu', s=80, 
                edgecolor='black', linewidth=0.8, alpha=0.9,
                vmin=-max_abs_res, vmax=max_abs_res)
    ax1.set_title('1. Resíduos Iniciais (OLS)\nMarginais', fontsize=14, pad=15)
    ax1.set_xlabel('Longitude (X)', fontsize=12)
    ax1.set_ylabel('Latitude (Y)', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # --- PLOT 2: Resíduos EM (Marginais) ---
    ax2.scatter(longitudes, latitudes, c=res_fin_flat, cmap='RdBu', s=80, 
                edgecolor='black', linewidth=0.8, alpha=0.9,
                vmin=-max_abs_res, vmax=max_abs_res)
    ax2.set_title('2. Resíduos Finais (EM)\nMarginais', fontsize=14, pad=15)
    ax2.set_xlabel('Longitude (X)', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.5)

    # --- PLOT 3: Resíduos EM (Decorrelacionados) ---
    sc3 = ax3.scatter(longitudes, latitudes, c=res_dec_flat, cmap='RdBu', s=80, 
                      edgecolor='black', linewidth=0.8, alpha=0.9,
                      vmin=-max_abs_res, vmax=max_abs_res)
    ax3.set_title('3. Resíduos Decorrelacionados\n(Prova do Modelo Espacial)', fontsize=14, pad=15)
    ax3.set_xlabel('Longitude (X)', fontsize=12)
    ax3.grid(True, linestyle='--', alpha=0.5)

    plt.subplots_adjust(left=0.05, right=0.88, top=0.82, bottom=0.1, wspace=0.05)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.65]) 
    cbar = fig.colorbar(sc3, cax=cbar_ax)
    cbar.set_label('Valor do Resíduo', rotation=270, labelpad=20, fontsize=12)

    plt.suptitle('Análise da Estrutura Espacial dos Resíduos (t-Student)', fontsize=16, weight='bold', y=0.98)
    plt.show()


def cross_validation(Y, Sigma, X=None):
    """
    Validação Cruzada Leave-One-Out (LOOCV) para modelos espaciais.
    Equivalente à função 'kc' do R.
    """
    n = len(Y)
    kcx = np.zeros((n, 3)) # Colunas: [Observado, Predito, Desvio Padrão Krigagem]
    
    # Se a matriz de delineamento (covariáveis) não for passada,
    # assume-se Krigagem Ordinária (vetor coluna de 1s, modelo de média constante)
    if X is None:
        X = np.ones((n, 1))
        
    for i in range(n):
        # 1. Ponto observado a ser deixado de fora
        Z_i = Y[i]
        x_i = X[i].reshape(-1, 1) # Tendência no ponto i
        
        # 2. Removendo o i-ésimo elemento para isolar os "vizinhos"
        Z_menos_i = np.delete(Y, i, axis=0)
        X_menos_i = np.delete(X, i, axis=0)
        
        # 3. Construindo as matrizes de covariância particionadas usando np.delete
        # K_block: Covariância entre os vizinhos (n-1 x n-1)
        K_block = np.delete(np.delete(Sigma, i, axis=0), i, axis=1)
        
        # L_block: Covariância entre o ponto isolado e seus vizinhos (n-1 x 1)
        L_block = np.delete(Sigma[:, i], i).reshape(-1, 1)
        
        # 4. Montando o Sistema de Equações da Krigagem (com multiplicadores de Lagrange)
        num_tendencias = X_menos_i.shape[1]
        
        # Bloco superior: [ K_block  |  X_menos_i ]
        K_top = np.hstack([K_block, X_menos_i])
        
        # Bloco inferior: [ X_menos_i^T |  0 ]
        K_bottom = np.hstack([X_menos_i.T, np.zeros((num_tendencias, num_tendencias))])
        
        # Sistema completo (K)
        K_sys = np.vstack([K_top, K_bottom])
        
        # Vetor Lado Direito (L)
        L_sys = np.vstack([L_block, x_i])
        
        # 5. Resolvendo o sistema para encontrar Pesos e Lagrange (Lamb)
        try:
            Lamb = la.solve(K_sys, L_sys)
        except la.LinAlgError:
            Lamb = la.pinv(K_sys) @ L_sys # Fallback para instabilidade numérica
            
        weights = Lamb[:-num_tendencias] # Pesos W
        mu = Lamb[-num_tendencias:]      # Multiplicador de Lagrange mu
        
        # 6. Cálculo da Predição (Z1)
        Z_pred = (weights.T @ Z_menos_i).item()
        
        # 7. Cálculo da Variância e Desvio Padrão
        # Fórmula correta para funções Covariância: C(0) - W^T*L - mu^T*x
        var_krig = Sigma[i, i] - (weights.T @ L_block + mu.T @ x_i).item()
        
        # Evita variâncias negativas minúsculas devido a arredondamento numérico
        sd_krig = np.sqrt(max(0, var_krig)) 
        
        # 8. Armazenando no array final
        kcx[i, 0] = Z_i.item()
        kcx[i, 1] = Z_pred
        kcx[i, 2] = sd_krig
        
    return kcx

def error_report(kcx):
    """
    Reproduz o bloco final do seu código R (Estatísticas do DF e DF_std).
    """
    df_obs_pred = kcx[:, 0] - kcx[:, 1] # Erro Absoluto (DF)
    df_std = df_obs_pred / kcx[:, 2]    # Erro Padronizado (DF.)
    
    # Criando um DataFrame do Pandas para ter o 'summary' bonito do R
    df_erros = pd.DataFrame({
        'Erro (DF)': df_obs_pred,
        'Erro Padronizado (DF.)': df_std
    })
    
    resumo = df_erros.describe().T # Análogo ao summary()
    ea = np.sum(np.abs(df_obs_pred)) / len(kcx) # Seu 'EA' no R
    
    print("\n=== Resumo dos Erros de Validação Cruzada ===")
    print(resumo[['min', '25%', '50%', 'mean', '75%', 'max', 'std']])
    print(f"\nErro Absoluto Médio (EA): {ea:.4f}")
    
    return df_erros, resumo, ea

def interactive_stats_view(Y, X, em_resultados, r_inicial, H, gr, k):
    """
    Aciona o menu interativo de diagnóstico do modelo.
    Calcula resíduos marginais e decorrelacionados, exibe erros de validação e plota gráficos finais.
    """
    
    if input("\nDeseja visualizar os gráficos? (0/1) ") == '1':
        # Resíduo EM (Marginal)
        beta_final = em_resultados["beta"].reshape(-1, 1)
        r_final = Y - X @ beta_final
        
        # Matriz de Covariância
        Sigma_final = em_resultados["Sigma"]

        # Resumo de Erros (Opcional)
        if input("Chamar Resumo dos Erros? (0/1) ") == '1':
            kcx = cross_validation(Y, Sigma_final, X=X)
            df_erros, resumo, ea = error_report(kcx)
            print("\nRelatório de Erros:\n", df_erros)
            print("\nResumo dos Erros:\n", resumo)
            print("\nErro Absoluto Médio (EA):", ea)
        
        # Usando Decomposição Cholesky para calcular: L^-1 * r_final (raiz quadrada inversa de Sigma). Desvio padrão espacialmente ajustado para Matriz.
        L = la.cholesky(Sigma_final, lower=True)
        r_decorrelacionado = la.solve(L, r_final) # Remover o efeito da escala ou da variância de uma única variável para compará-la, o Escore-Z
        
        plot_semivariogram_curves(H, r_final, em_resultados["phi1"], em_resultados["phi2"], em_resultados["phi3"], k, plot_curves=True)
        
        plot_residuals(r_inicial, r_final, r_decorrelacionado, gr)
        
        return

    return


def gl_optmizer(gl_atual, n, v):
    """
     Possivel otimização para os graus de liberdade (gl) na t-Student Multivariada.
    """
    # 1. Cálculo da esperança do log(v) 
    E_ln_v = sp.digamma((gl_atual + n) / 2.0) - np.log((gl_atual + n) / 2.0) + np.log(v) # E[ln(v)] = digamma((gl + n)/2) - ln((gl + n)/2) + ln(v)
    
    # 2. Calcula log-verossimilhança assumindo Gamma(v/2, v/2)
    def custo_gl(nu):
        termo1 = (nu / 2.0) * np.log(nu / 2.0) # const normalização -> (nu/2) * ln(nu/2)
        termo2 = -sp.gammaln(nu / 2.0) # denominador da constante de normalização -> -ln(Gamma(nu/2))
        termo3 = (nu / 2.0 - 1.0) * E_ln_v # interação da "forma" da distribuição v -> (nu/2 - 1) * E[ln(v)]
        termo4 = -(nu / 2.0) * v # decaimento exponencial característico da distribuição Gama 
        
        # Retorna negativo porque queremos maximizar a verossimilhança (opt minimiza por padrão)
        return -(termo1 + termo2 + termo3 + termo4)

    # 3. Entrega dos termos para a função de otimização (usando método 'bounded' para garantir que gl seja positivo e razoável)
    resultado = opt.minimize_scalar(custo_gl, bounds=(2.1, 100.0), method='bounded') # bounds sao os limites para gl
    
    if resultado.success:
        return resultado.x # Retorna o novo gl otimizado
    else:
        return gl_atual # Se falhar, mantém o gl antigo por segurança

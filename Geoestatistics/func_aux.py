import pandas as pd
import numpy as np
import scipy.special as sp
import scipy.linalg as la
import matplotlib.pyplot as plt


# Manipulação de dados
def read_data(filename):
    """Lê arquivo tentando diferentes separadores."""
    for sep in ["\t", ",", "\s+"]:
        try:
            df = pd.read_csv(filename, sep=sep, header=None, engine="python")
            return df
        except:
            continue
    raise ValueError(f"Não foi possível ler o arquivo {filename}")

def read_dados_wypych(x_file, wypych_file):
    """Prepara X, Y e gr a partir dos arquivos."""
    X = read_data(x_file).to_numpy()
    df = read_data(wypych_file)
    Y = df.iloc[:, 3].to_numpy().reshape(-1, 1)
    gr = df.iloc[:, 1:3].to_numpy()
    return X, Y, gr


def exponencial_correlation(H, phi3): # Pagina 13, item a) Modelo Exponencial p(h)
    """Calcula a correlação Exponencial para semivariograma."""
    return np.exp(-H / phi3)

def gaussiano_correlation(H, phi3): # Pagina 13, item b) Modelo Gaussiano p(h)
    """Calcula a correlação Gaussiana para semivariograma."""
    return np.exp(-(H / phi3)**2)

# Matern
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


# Derivadas de K_kappa
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
    

# Funções de visualização
def ver_kappa(H, r_inicial, phi1, phi2, phi3, k):
    while True:
        plot_analise_chutes(H, r_inicial, phi1, phi2, phi3, k)
        
        print("\n--- AVALIAÇÃO VISUAL ---")
        print(f"1. phi1 (Nugget) = {phi1:.4f}")
        print(f"2. phi2 (Sill)   = {phi2:.4f}")
        print(f"3. phi3 (Range)  = {phi3:.4f}")
        print(f"4. Kappa         = {k:.4f}")
        
        resp = input("\nDeseja testar novos valores no gráfico? (s/n): ").strip().lower()
        if resp == 'n':
            plt.close('all')
            print("\nFechando gráfico e prosseguindo com a otimização...\n")
            return phi1, phi2, phi3, k
        elif resp == 's':
            try:
                phi1 = 0.22
                phi2 = 0.15
                phi3 = 110.0
                k = float(input("Novo valor para Kappa: "))
                plt.close('all') 
            except ValueError:
                print("Entrada inválida! Digite apenas números.")
   
def plot_analise_chutes(H, residuo, phi1, phi2, phi3, k):
    """Gera o Semivariograma Empírico vs Teórico para avaliar os chutes."""

    dist_flat = H.flatten()

    res_diff = (residuo - residuo.T)**2 # Diferença quadrática -> res_diff = (e_i - e_j)^2
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

    # --- Modelo Matérn ---
    correl_teorica_matern = matern_correlation(h_teorico, phi3, k)
    variograma_matern = phi1 + phi2 * (1 - correl_teorica_matern) # Pagina 14, item c) Modelo da familia Matérn y(h)
    
    # --- Modelo Exponencial ---
    correl_teorica_exponencial = exponencial_correlation(h_teorico, phi3)
    variograma_exponencial = phi1 + phi2 * (1 - correl_teorica_exponencial) # Pagina 13, item a) y(h)
    
    # --- Modelo Gaussiano ---
    correl_teorica_gaussiana = gaussiano_correlation(h_teorico, phi3)
    variograma_gaussiana = phi1 + phi2 * (1 - correl_teorica_gaussiana) # Pagina 13, item b) y(h)

    # 3. Plotagem em painel (2x2)
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"Comparação de Modelos Teóricos (φ1={phi1}, φ2={phi2}, φ3={phi3})", fontsize=16)

    # --- Configuração base comum para todos os 4 subgráficos ---
    for ax in axs.flat:
        # Plot dos dados empíricos
        ax.scatter(dist_flat, variograma_exp, alpha=0.1, color='gray', label='Dados (Pares)')
        ax.scatter(dist_medias, gama_medias, color='black', s=50, zorder=3, label='Empírico (Médias)')
        # Plot do patamar
        ax.axhline(y=phi1+phi2, color='purple', linestyle=':', alpha=0.6, label=f'Patamar')
        
        # Configurações de eixos e grid
        ax.set_xlabel("Distância (h)")
        ax.set_ylabel("Semivariância γ(h)")
        ax.set_ylim(0, (phi1 + phi2) * 1.3)
        ax.grid(True, alpha=0.3)

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

    # Ajusta o layout para que os títulos e legendas não fiquem sobrepostos
    plt.tight_layout()
    plt.show()

def plot_residuo(r_inicial, r_final, r_decorr, gr):
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
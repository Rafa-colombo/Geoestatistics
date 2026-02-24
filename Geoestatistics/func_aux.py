import pandas as pd
import numpy as np
import scipy.special as sp
import scipy.linalg as la
import matplotlib.pyplot as plt

# =================================================================================================
# DICIONÁRIO DE FUNÇÕES DE BIBLIOTECAS UTILIZADAS
# =================================================================================================
# PANDAS (pd)
# Função pd.read_csv que lê um arquivo de dados tabular para um DataFrame -> Parametros: Caminho do arquivo, Separador, Cabeçalho (None=sem cabeçalho), Engine de processamento

# NUMPY (np)
# Função np.abs que calcula o valor absoluto elemento por elemento -> Parametros: Array de entrada
# Função np.any que testa se algum elemento do array é avaliado como True -> Parametros: Array contendo a condição lógica
# Função np.array que cria um array numpy a partir de uma estrutura de dados existente -> Parametros: Objeto a ser convertido (ex: lista)
# Função np.fill_diagonal que preenche a diagonal principal de um array -> Parametros: Array multidimensional, Valor de preenchimento
# Função np.linspace que cria uma sequência de números uniformemente espaçados -> Parametros: Onde começa, Onde termina, Quantos cortes fazer(de tamanhos iguais)
# Função np.max que retorna o valor máximo de um array -> Parametros: Array de entrada
# Função np.mean que calcula a média aritmética ao longo do eixo especificado -> Parametros: Array contendo os valores
# Função np.ndim que retorna o número de dimensões de um array -> Parametros: Array a ser verificado
# Função np.sum que calcula a soma dos elementos de um array -> Parametros: Array contendo os valores a serem somados
# Função np.where que retorna elementos escolhidos dependendo da condição -> Parametros: Condição, Valor se verdadeiro, Valor se falso

# SCIPY (sp)
# Função sp.gamma que calcula a função Gama -> Parametros: Valor numérico a ser avaliado
# Função sp.kv que calcula a função de Bessel modificada de segunda espécie -> Parametros: Ordem da função (k), Argumento de avaliação (uphi)

# MATPLOTLIB (plt / fig)
# Função fig.add_axes que adiciona um eixo à figura em um retângulo customizado -> Parametros: Lista com [posição_esquerda, posição_inferior, largura, altura]
# Função fig.colorbar que adiciona uma barra de cores a um plot -> Parametros: Objeto mapeável, Eixo onde desenhar a barra de cores (cax)
# Função plt.close que fecha janelas de figuras abertas -> Parametros: Qual janela fechar ('all' fecha todas)
# Função plt.figure que cria uma nova figura -> Parametros: Dimensões da figura em polegadas (largura, altura)
# Função plt.grid que configura as linhas de grade -> Parametros: Booleano para ligar/desligar, Nível de transparência (alpha)
# Função plt.legend que insere a legenda nos eixos -> Parametros: Nenhum (usa as labels previamente definidas nas plotagens)
# Função plt.plot que plota y em função de x como linhas e/ou marcadores -> Parametros: Valores de X, Valores de Y, Cor da linha, Espessura da linha (lw), Rótulo da legenda
# Função plt.scatter que plota um gráfico de dispersão de y em função de x -> Parametros: Valores de X, Valores de Y, Nível de transparência/Cor/Tamanho/Ordem/Rótulo
# Função plt.show que exibe todas as figuras abertas -> Parametros: Nenhum
# Função plt.subplots que cria uma figura e um conjunto de subplots -> Parametros: Número de linhas, Número de colunas, Tamanho da figura, Se o eixo Y será compartilhado
# Função plt.subplots_adjust que ajusta os parâmetros de layout dos subplots -> Parametros: Bordas esquerda/direita/superior/inferior, Espaçamento horizontal (wspace)
# Função plt.suptitle que adiciona um título centralizado à figura inteira -> Parametros: Texto do título, Tamanho da fonte, Espessura da fonte, Posição vertical (y)
# Função plt.title que define o título do gráfico -> Parametros: Texto do título
# Função plt.xlabel que define o rótulo do eixo X -> Parametros: Texto do rótulo
# Função plt.ylabel que define o rótulo do eixo Y -> Parametros: Texto do rótulo
# Função plt.ylim que obtém ou define os limites do eixo Y atual -> Parametros: Limite inferior, Limite superior
# =================================================================================================

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


# Matern
def matern_correlation(H, phi3, k):
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
    variograma_exp = 0.5 * res_diff.flatten() # Semivariância pontual -> gamma_ij = 0.5 * (e_i - e_j)^2
    
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
    correl_teorica = matern_correlation(h_teorico, phi3, k)
    variograma_teorico = phi1 + phi2 * (1 - correl_teorica) # Variograma Teórico -> gamma(h) = phi1 + phi2 * (1 - R(h))
    
    # 3. Plotagem
    plt.figure(figsize=(10, 5))
    plt.scatter(dist_flat, variograma_exp, alpha=0.1, color='gray', label='Dados (Pares)')
    plt.scatter(dist_medias, gama_medias, color='blue', s=50, zorder=3, label='Empírico (Médias)')
    plt.plot(h_teorico, variograma_teorico, color='red', lw=2, 
             label=f'Teórico (phi1={phi1}, phi2={phi2}, phi3={phi3}, k={k})')
    
    plt.title("Validação Visual do Modelo Matérn")
    plt.xlabel("Distância (h)")
    plt.ylabel("Semivariância γ(h)")
    plt.ylim(0, (phi1 + phi2) * 1.5)
    plt.legend()
    plt.grid(True, alpha=0.3)
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
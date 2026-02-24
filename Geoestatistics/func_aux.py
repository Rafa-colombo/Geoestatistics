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


# Matern
def matern_correlation(H, phi3, k):
    """Calcula a matriz de correlação Matérn."""
    # Evitar divisão por zero na diagonal principal
    H_safe = np.where(H > 0, H, 1e-10)
    uphi = H_safe / phi3
    
    coef = 1.0 / ((2**(k - 1)) * sp.gamma(k))
    res = coef * (uphi**k) * sp.kv(k, uphi)
    
    # VERIFICAÇÃO DE DIMENSÃO:
    if np.ndim(res) >= 2:
        # Se for matriz (usado no EM), preenchemos a diagonal
        np.fill_diagonal(res, 1.0) # A correlação do ponto com ele mesmo é 1
    else:
        # Se for vetor (usado no gráfico), garantimos que onde H era 0, res é 1
        res = np.where(H == 0, 1.0, res)

    return res


# Derivadas de K_kappa, acredito que use só uma dk
def dK(H, phi3, k):
    """1ª Derivada de K_kappa em relação a u"""
    H_safe = np.where(H == 0, 1e-10, H)
    uphi = H_safe / phi3
    res = -1/2 * (sp.kv(k - 1, uphi) + sp.kv(k + 1, uphi))
    np.fill_diagonal(res, 0)
    return res

def dKK(H, phi3, k):
    """2ª Derivada de K_kappa em relação a u"""
    H_safe = np.where(H == 0, 1e-10, H)
    uphi = H_safe / phi3
    res = 1/4 * (kv(k - 2, uphi) + 2 * kv(k, uphi) + kv(k + 2, uphi))
    np.fill_diagonal(res, 0)
    return res
    

# Funções de visualização
def ver_kappa(H, r_inicial, phi1, phi2, phi3, k):
    while True:
        # Chama a função externa via func_aux.
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
                phi1 = 0.22#float(input("Novo valor para phi1 (Nugget): "))
                phi2 = 0.15#float(input("Novo valor para phi2 (Sill): "))
                phi3 = 110.0#float(input("Novo valor para phi3 (Range): "))
                k = float(input("Novo valor para Kappa: "))
                plt.close('all') 
            except ValueError:
                print("Entrada inválida! Digite apenas números.")
   
def plot_analise_chutes(H, residuo, phi1, phi2, phi3, k):
    """Gera o Semivariograma Empírico vs Teórico para avaliar os chutes."""

    # Pegamos as distâncias únicas e os resíduos para estimar a variância espacial
    dist_flat = H.flatten()

    # Residuo.T transpõe o vetor/matriz. O 'broadcasting' do numpy cria uma matriz de diferenças.
    res_diff = (residuo - residuo.T)**2
    variograma_exp = 0.5 * res_diff.flatten() # Semivariância de cada par. Multiplica por 0.5 e transforma em vetor 1D
    
    num_lags = 20
    bins = np.linspace(0, np.max(H), num_lags + 1) # Onde começa, Onde termina, Quantos cortes fazer(de tamanhos iguais)
    
    dist_medias = []
    gama_medias = []

    for i in range(num_lags):
        # Cria uma máscara selecionando só os pontos que estão dentro do intervalo atual
        mask = (dist_flat > bins[i]) & (dist_flat <= bins[i+1])
        print(f"Bin {i+1}: Distância entre {bins[i]:.2f} e {bins[i+1]:.2f} - Pares encontrados: {np.sum(mask)}")
        
        if np.any(mask): 
            dist_medias.append(np.mean(dist_flat[mask])) # Media das distâncias
            gama_medias.append(np.mean(variograma_exp[mask]))# Media Semivariância 
            
    # Converte as listas de volta para vetores do numpy
    dist_medias = np.array(dist_medias)
    gama_medias = np.array(gama_medias)

    # 2. Criar Curva Teórica baseada nos parâmetros passados
    h_teorico = np.linspace(0, np.max(H), 100) # Cria 100 pontos de distância espaçados uniformemente de 0 até a distância máxima em H
    # Variograma Matérn: γ(h) = Nugget + Sill_Parcial * (1 - Correl)
    correl_teorica = matern_correlation(h_teorico, phi3, k) # Calcula a correlação de Matérn teórica para as distâncias de h_teorico
    variograma_teorico = phi1 + phi2 * (1 - correl_teorica) # Equação do semivariograma a partir da correlação: Nugget + Sill_Parcial * (1 - Correl)
    
    # 3. Plotagem
    plt.figure(figsize=(10, 5))
    # Plotamos os pontos experimentais (com transparência para ver a densidade)
    plt.scatter(dist_flat, variograma_exp, alpha=0.1, color='gray', label='Dados (Pares)')
    plt.scatter(dist_medias, gama_medias, color='blue', s=50, zorder=3, label='Empírico (Médias)')
    # Plotamos a curva teórica
    plt.plot(h_teorico, variograma_teorico, color='red', lw=2, 
             label=f'Teórico (phi1={phi1}, phi2={phi2}, phi3={phi3}, k={k})')
    
    plt.title("Validação Visual do Modelo Matérn")
    plt.xlabel("Distância (h)")
    plt.ylabel("Semivariância γ(h)")
    plt.ylim(0, (phi1 + phi2) * 1.5) # Limita o eixo Y para melhor visualização
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_residuo(r_inicial, r_final, r_decorr, gr): # metodo que recebe os residuos e as coordenadas para plotar o grafico, ambos como arrays do numpy.
    """ VISUALIZAÇÃO ESPACIAL COMPARATIVA DOS RESÍDUOS (Lado a Lado) """

    # 1. Achatando os vetores (N x 1) para (N,)
    res_ini_flat = r_inicial.flatten()
    res_fin_flat = r_final.flatten()
    res_dec_flat = r_decorr.flatten()
    
    longitudes = gr[:, 1]
    latitudes = gr[:, 0]

    # 2. Escala máxima unificada para a barra de cores 
    max_abs_res = max(np.max(np.abs(res_ini_flat)), 
                      np.max(np.abs(res_fin_flat)), 
                      np.max(np.abs(res_dec_flat)))

    # 3. Criando a figura com 1 linha e 3 colunas
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

    # 4. Barra de Cores (Eixo Fixo)
    
    # Primeiro, encolhemos a área dos gráficos para eles terminarem em 88% da largura da tela
    plt.subplots_adjust(left=0.05, right=0.88, top=0.82, bottom=0.1, wspace=0.05)

    # Criamos uma "caixa" manual para a barra de cores: [esquerda, base, largura, altura]
    # Ela vai ficar na posição 90% da tela (depois dos gráficos)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.65]) 
    
    # Desenhamos a barra estritamente dentro dessa caixa (usando cax=cbar_ax)
    cbar = fig.colorbar(sc3, cax=cbar_ax)
    cbar.set_label('Valor do Resíduo', rotation=270, labelpad=20, fontsize=12)

    plt.suptitle('Análise da Estrutura Espacial dos Resíduos (t-Student)', fontsize=16, weight='bold', y=0.98)

    # Remova qualquer plt.tight_layout() que estiver aqui, pois já ajustamos manualmente!
    
    plt.show()


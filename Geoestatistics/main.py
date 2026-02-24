import numpy as np
from func_aux import read_dados_wypych
from alg_em import EM
from analiseKap import ver_grafico

# --- Arquivos ---
x_file = "X.txt"
wypych_file = "Dados_Wypych.txt"

# --- Lê dados ---
X, Y, gr = read_dados_wypych(x_file, wypych_file)

# --- Inicializa parâmetros ---
theta_init = [1.0, 1.0, 1.0, 0.5, 0.5, 0.5]  # beta1,beta2,beta3, phi1,phi2,phi3
k = 0.5

# Ver grafico de kappa e possivel alteração de phi
if int(input("Gostaria de realizar chamada de analise Kappa? (1 para sim, 0 para não): ")) == 1:
    Sigma = ver_grafico(theta_init, X, Y, gr, k)
    print(Sigma)
    if int(input("Refatorar valores de phi e kappa? (1 para sim, 0 para não): ")) == 1:
        for i in range(3):
            novo_valor = float(input(f"Novo valor para theta_init[{i+2}]: "))
            theta_init[i+2] = novo_valor
        novo_valor = float(input("Novo valor para Kappa: "))
        k = novo_valor

# --- Roda EM ---
theta_final = EM(theta_init, X, Y, gr, k)

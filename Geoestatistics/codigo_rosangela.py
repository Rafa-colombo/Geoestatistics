"""
Inicialmente apenas refatoração do codigo feito pela professora no colab.
Objetivo: Analisar o comportamento da função Matérn para diferentes direções espaciais e parâmetros, permitindo visualizar e comparar phi1, phi2, phi3.

"""
import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import gamma, kv
from func_aux import read_dados_wypych, plot_residuo

# =====================================================================
# 1. LEITURA DOS DADOS
# =====================================================================

x_file = "X.txt"
wypych_file = "Dados_Wypych.txt"

X, Y, gr = read_dados_wypych(x_file, wypych_file)

# Garante que os retornos sejam arrays do NumPy para as operações matriciais
X = np.array(X)
Y = np.array(Y)
gr = np.array(gr)

n = gr.shape[0]

# =====================================================================
# 2. CONFIGURAÇÃO INICIAL E MÍNIMOS QUADRADOS ORDINÁRIOS (OLS)
# =====================================================================
# Mínimos Quadrados Ordinários para achar os Betas iniciais
beta = np.linalg.solve(X.T @ X, X.T @ Y)

# Parâmetros espaciais iniciais
phi1 = 0.22  # Efeito Pepita (Nugget)
phi2 = 0.15  # Patamar Parcial (Sill)
phi3 = 110   # Alcance (Range)
k = 0.5      # Parâmetro de suavidade da família Matérn
gl = 5       # Graus de liberdade

# Vetor de parâmetros completo
theta = np.vstack((beta, [[phi1], [phi2], [phi3]]))

# =====================================================================
# 3. MATRIZ DE DISTÂNCIAS E FUNÇÕES MATÉRN
# =====================================================================
I = np.identity(n)
H = cdist(gr, gr)

def matern_correlation(H, phi3, k):
    
    H_safe = np.where(H == 0, 1e-10, H) # Evita divisão por zero
    uphi = H_safe / phi3
    R = (1.0 / ((2.0**(k - 1)) * gamma(k))) * (uphi**k) * kv(k, uphi)
    np.fill_diagonal(R, 1.0) # A correlação do ponto com ele mesmo é 1
    return R

# Matriz de correlação (Rf3) e Matriz de Covariância Total (Sigma)
Rf3 = matern_correlation(H, phi3, k)
Sigma_mat = phi1 * I + phi2 * Rf3

# =====================================================================
# 4. DERIVADAS DE PRIMEIRA ORDEM (JACOBIANO)
# =====================================================================
def dK(H, phi3, k):
    H_safe = np.where(H == 0, 1e-10, H)
    uphi = H_safe / phi3
    res = -1/2 * (kv(k - 1, uphi) + kv(k + 1, uphi))
    np.fill_diagonal(res, 0)
    return res

# Variáveis de apoio para evitar recálculo e divisões por zero
H_safe_global = np.where(H == 0, 1e-10, H)
uphi_global = H_safe_global / phi3

d_phi1 = I
d_phi2 = Rf3
d_phi3 = phi2 * (-(1/phi3) * k * d_phi2 + (1 / ((2**(k-1)) * gamma(k))) * (np.multiply((uphi_global**(k+1)), dK(H, phi3, k))))
np.fill_diagonal(d_phi3, 0) # Ajuste de segurança para a diagonal

# =====================================================================
# 5. DERIVADAS DE SEGUNDA ORDEM (HESSIANO)
# =====================================================================
O = np.zeros((n, n))

def dKK(H, phi3, k):
    H_safe = np.where(H == 0, 1e-10, H)
    uphi = H_safe / phi3
    res = 1/4 * (kv(k - 2, uphi) + 2 * kv(k, uphi) + kv(k + 2, uphi))
    np.fill_diagonal(res, 0)
    return res

d_phi1_phi1 = O
d_phi1_phi2 = O
d_phi1_phi3 = O
d_phi2_phi1 = O
d_phi2_phi2 = O

d_phi2_phi3 = -(1/phi3) * (k * Rf3 + (1 / (2**(k-1) * gamma(k))) * (np.multiply((uphi_global)**(k+1), dK(H, phi3, k))))
np.fill_diagonal(d_phi2_phi3, 0)

d_phi3_phi1 = O
d_phi3_phi2 = (Rf3 / phi3)

d_phi3_phi3 = ((k * (k + 1) * H_safe_global) / phi3**2) + ((1 / ((phi3**2) * 2**(k-1) * gamma(k))) * ((uphi_global)**(k+1))) * \
              ((2 * (k + 1)) * dK(H, phi3, k) + np.multiply(uphi_global, dKK(H, phi3, k)))
np.fill_diagonal(d_phi3_phi3, 0)

# =====================================================================
# 6. RESÍDUOS E AVALIAÇÃO (FORMA QUADRÁTICA)
# =====================================================================
r = Y - X @ beta
IC = np.linalg.inv(Sigma_mat) 
u = (r.T @ IC @ r) # Residuos sao "erros/ruidos" do modelo

print("--- RESULTADOS INICIAIS ---")
print("beta = \n", beta)
print("phi1 (Nugget) =", theta[-3, 0])
print("phi2 (Sill)   =", theta[-2, 0])
print("phi3 (Range)  =", theta[-1, 0])
print("-" * 50)
print(f"Soma ponderada dos quadrados dos resíduos (u): {u[0, 0]:.4f}")

plot_residuo(r, gr)
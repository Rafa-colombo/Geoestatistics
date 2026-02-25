import numpy as np
import scipy.linalg as la
import scipy.special as sp
import func_aux 

# Ambas funções são bem semelhantes, mudando apenas a forma de atualização de phi3 (Fisher Scoring isolado vs Newton-Raphson exato).

# =====================================================================
# 3. ALGORITMO EM (t-Student Espacial)
# =====================================================================

def em_tstudent_Fischer(X, Y, gr, theta_init, H, k, gl=4, max_iter=100, tol=1e-4):
    """
    Estimação de parâmetros espaciais robustos (t-Student) via Algoritmo EM.
    Passo Fischer Scoring para atualização de phi1 e phi2, e Newton 1D para phi3. (Utização de dK penas) -> Para NR exato, seria necessário dKK na aplicação.
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
# 3. ALGORITMO EM (t-Student) com Newton-Raphson EXATO para phi3
# =====================================================================

def em_tstudent_NRExato(X, Y, gr, theta_init, H, k, gl=4, max_iter=100, tol=1e-4):
    """
    Estimação de parâmetros espaciais robustos (t-Student) via Algoritmo EM.
    Utiliza Fisher Scoring para phi1 e phi2, e Newton-Raphson exato para phi3.
    """
    # Desempacotando parâmetros iniciais
    beta = np.array(theta_init[:-3]).reshape(-1, 1)
    phi1, phi2, phi3 = theta_init[-3:]
    
    n = len(Y)
    I = np.eye(n)
    
    print(f"Iniciando otimização EM (t-Student)...\nValores iniciais: beta={beta.flatten()}, k={k}, phi1={phi1}, phi2={phi2}, phi3={phi3}")
    
    for it in range(1, max_iter + 1):
        
        # 1. Construção das Matrizes Espaciais 
        Rf3 = func_aux.matern_correlation(H, phi3, k)
        Sigma = phi1 * I + phi2 * Rf3 

        # Cholesky para inversão eficiente e estável
        try:
            c, lower = la.cho_factor(Sigma)
            Sigma_inv = la.cho_solve((c, lower), I) 
        except la.LinAlgError:
            print(f"Aviso: Matriz instável na iteração {it}. Usando pseudo-inversa de fallback.")
            Sigma_inv = la.pinv(Sigma) 
            
        # 2. Atualização do beta 
        X_invS = X.T @ Sigma_inv 
        beta_new = la.solve(X_invS @ X, X_invS @ Y) 

        # 3. Cálculo do peso latente da t-Student (v)
        r = Y - X @ beta_new 
        u = (r.T @ Sigma_inv @ r).item() 
        v = (gl + n) / (gl + u)          

        # 4. Derivadas Parciais e Auxiliares
        d_phi1 = I 
        d_phi2 = Rf3 

        # Chamando funções especiais de func_aux
        dK_val = func_aux.dK(H, phi3, k)
        dKK_val = func_aux.dKK(H, phi3, k)

        uphi = np.where(H > 0, H / phi3, 0)
        H_phi3_k1 = np.where(H > 0, uphi**(k + 1), 0)
        
        coef_M = 1.0 / ((2**(k - 1)) * sp.gamma(k))

        # Primeira derivada da matriz de correlação em relação a phi3 (Base para o F do R)
        M = k * d_phi2 + coef_M * (H_phi3_k1 * dK_val)
        d_phi3 = phi2 * (-(1.0 / phi3) * M) # Equivalente ao 'F' no R

        # Segunda derivada da matriz em relação a phi3 (Base para o H do R)
        M2 = M * (1 + k) + coef_M * H_phi3_k1 * ((k + 1) * dK_val + uphi * dKK_val)
        d2_phi3 = phi2 * ((1.0 / (phi3**2)) * M2) # Equivalente ao 'H' no R

        # 5. PASSO M: Fisher Scoring para phi1 e phi2 (Sistema 2x2 linear)
        invS_r = Sigma_inv @ r 
        invS_d1 = Sigma_inv @ d_phi1 # Resulta no próprio Sigma_inv
        invS_d2 = Sigma_inv @ d_phi2 # Sigma_inv @ Rf3

        a11 = np.sum(invS_d1 * invS_d1.T) 
        a12 = np.sum(invS_d1 * invS_d2.T) 
        a22 = np.sum(invS_d2 * invS_d2.T) 

        A_2x2 = np.array([
            [a11, a12],
            [a12, a22]
        ])

        S1 = v * (invS_r.T @ d_phi1 @ invS_r).item() 
        S2 = v * (invS_r.T @ d_phi2 @ invS_r).item() 
        S_2x2 = np.array([S1, S2])

        Fi_2x2 = la.solve(A_2x2, S_2x2) 
        phi1_new = Fi_2x2[0]
        phi2_new = Fi_2x2[1]

        # 6. Atualização de phi3 (Newton-Raphson 1D EXATO)
        invS_d3 = Sigma_inv @ d_phi3
        
        # Qfi3 (Primeira derivada / Score log-verossimilhança)
        Qfi3 = -0.5 * np.trace(invS_d3) + 0.5 * v * (invS_r.T @ d_phi3 @ invS_r).item()

        # Qffi3 (Segunda derivada / Hessiana)
        F_invS_F = d_phi3 @ invS_d3
        H_minus_F_invS_F = d2_phi3 - F_invS_F
        H_minus_2F_invS_F = d2_phi3 - 2 * F_invS_F

        traço_hessiana = np.trace(Sigma_inv @ H_minus_F_invS_F)
        termo_r_hessiana = (invS_r.T @ H_minus_2F_invS_F @ invS_r).item()

        Qffi3 = -0.5 * traço_hessiana + 0.5 * v * termo_r_hessiana

        # Passo de atualização de Newton-Raphson
        phi3_new = phi3 - (Qfi3 / Qffi3)

        # 7. Verificação de Convergência
        theta_old = np.concatenate([beta.flatten(), [phi1, phi2, phi3]])
        theta_new = np.concatenate([beta_new.flatten(), [phi1_new, phi2_new, phi3_new]])

        erro = np.linalg.norm(theta_old - theta_new) / np.linalg.norm(theta_old) 

        # Atualiza variáveis para a próxima iteração garantindo positividade da variância
        beta = beta_new
        phi1 = max(1e-5, phi1_new)
        phi2 = max(1e-5, phi2_new)
        phi3 = max(1e-5, phi3_new)
        
        if it == 1 or it % 5 == 0 or erro < tol:
            print(f"Iter {it}: Erro = {erro:.6f} | beta = {beta_new.flatten()[0]:.4f}, phi1 = {phi1:.4f}, phi2 = {phi2:.4f}, phi3 = {phi3:.4f}")
            
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

    print("\n=== Limite de iterações atingido sem convergência estrita! ===")
    return {
        "beta": beta.flatten(),
        "phi1": phi1,
        "phi2": phi2,
        "phi3": phi3,
        "Sigma": Sigma,
        "iteracoes": it,
        "erro_final": erro
    }



    
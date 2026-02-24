import numpy as np
from func_aux import matern


def EM(theta_init, X, Y, gr, k=0.5, max_iter=1000, tol=0.005):
    theta = np.array(theta_init, dtype=float)
    n = len(Y)
    
    for it in range(1, max_iter + 1):
        beta = theta[:3].reshape(-1, 1)
        phi = theta[3:]
        
        # --- Passo E: calcula Sigma (simplificado) --- h[i, j] = distância euclidiana entre o ponto i e o ponto j.
        h = np.sqrt(((gr[:, None, :] - gr[None, :, :]) ** 2).sum(axis=2))
        Sigma = matern(h, phi[0], k) + np.eye(n) * 1e-6  # distancia, phi1 e kappa + mat Indep
        
        # --- Passo M: atualiza beta e phi ---
        try:
            Sigma_inv = np.linalg.inv(Sigma) # tenta inversa de sigma
        except np.linalg.LinAlgError:
            Sigma_inv = np.linalg.pinv(Sigma) # se for n invertivel usa pseudo inversa (Moore-Penrose)
        
        beta_new = np.linalg.solve(X.T @ Sigma_inv @ X, X.T @ Sigma_inv @ Y)
        phi_new = phi * 0.9 + 0.1  # ajuste simplificado para manter positivo
        
        erro = np.linalg.norm(beta_new.flatten() - beta.flatten()) + np.linalg.norm(phi_new - phi)
        
        # Atualiza theta
        theta[:3] = beta_new.flatten()
        theta[3:] = phi_new
        
        print(f"Iter {it}: erro={erro:.6f}, phi={phi_new}")
        
        if erro < tol:
            print("\n=== Convergência atingida ===")
            print(f"Iterações: {it}, Erro final: {erro:.6f}")
            break
    print(Sigma)
    print("\n=== Resultado Final ===")
    print(f"beta: {theta[:3]}")
    print(f"phi: {theta[3:]}")
    
    return theta

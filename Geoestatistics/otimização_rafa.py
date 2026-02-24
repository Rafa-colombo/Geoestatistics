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

# =====================================================================
# 1. LEITURA DOS DADOS
# =====================================================================
x_file = "X.txt"
wypych_file = "Dados_Wypych.txt"

X, Y, gr = func_aux.read_dados_wypych(x_file, wypych_file)

X = np.array(X)
Y = np.array(Y)
gr = np.array(gr)

if Y.ndim == 1:
    Y = Y.reshape(-1, 1)

n = gr.shape[0]
I = np.identity(n)


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


# =====================================================================
# 3. ALGORITMO EM (t-Student Espacial)
# =====================================================================

def em_tstudent_spatial(X, Y, gr, theta_init, H, k, gl=4, max_iter=100, tol=1e-4):
    """
    Estimação de parâmetros espaciais robustos (t-Student) via Algoritmo EM.
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
        Sigma = phi1 * I + phi2 * Rf3
        
        # OTIMIZAÇÃO: Cholesky ao invés de Inversa Direta
        try:
            c, lower = la.cho_factor(Sigma)
            Sigma_inv = la.cho_solve((c, lower), I)
        except la.LinAlgError:
            print(f"Aviso: Matriz instável na iteração {it}. Usando pseudo-inversa de fallback.")
            Sigma_inv = la.pinv(Sigma)
            
        # 2. Atualização do Beta (GLS)
        X_invS = X.T @ Sigma_inv
        beta_new = la.solve(X_invS @ X, X_invS @ Y)
        
        # 3. PASSO E: Cálculo do peso latente da t-Student (v)
        r = Y - X @ beta_new
        u = (r.T @ Sigma_inv @ r).item() # Distância de Mahalanobis
        v = (gl + n) / (gl + u)          # Peso robusto
        
        # 4. Derivadas Parciais (Matrizes d_phi)
        d_phi1 = I
        d_phi2 = Rf3
        
        # Chamando funções especiais do scipy e de func_aux
        dK_val = func_aux.dK(H, phi3, k)
        H_phi3_k1 = np.where(H > 0, (H / phi3)**(k + 1), 0)
        coef_M = 1.0 / ((2**(k - 1)) * sp.gamma(k))
        
        M = k * d_phi2 + coef_M * (H_phi3_k1 * dK_val)
        d_phi3 = phi2 * (-(1.0 / phi3) * M)
        
        # 5. PASSO M: Fisher Scoring / Gradiente
        invS_r = Sigma_inv @ r
        
        # Vetor Score (S)
        S1 = v * (invS_r.T @ d_phi1 @ invS_r).item()
        S2 = v * (invS_r.T @ d_phi2 @ invS_r).item()
        S3 = v * (invS_r.T @ d_phi3 @ invS_r).item()
        S = np.array([S1, S2, S3])
        
        # Matriz de Informação Esperada (A)
        invS_d1 = Sigma_inv @ d_phi1
        invS_d2 = Sigma_inv @ d_phi2
        invS_d3 = Sigma_inv @ d_phi3
        
        a11 = np.sum(invS_d1 * invS_d1.T)
        a12 = np.sum(invS_d1 * invS_d2.T)
        a13 = np.sum(invS_d1 * invS_d3.T)
        
        a21 = np.sum(invS_d1 * (Sigma_inv @ Rf3).T)
        a22 = np.sum(invS_d2 * (Sigma_inv @ Rf3).T)
        
        invS_M = Sigma_inv @ M
        a33 = np.sum(invS_M * (Sigma_inv @ (phi2 * Rf3)).T)
        
        A = np.array([
            [a11, a21, 0],
            [a12, a22, 0],
            [a13, 0, a33]
        ])
        
        # 6. Atualização dos Parâmetros Espaciais
        Fi = S @ la.inv(A)
        phi1_new = Fi[0]
        phi2_new = Fi[1]
        Tau = Fi[2]
        phi3_new = -phi2_new / Tau if Tau != 0 else phi3
        
        # 7. Verificação de Convergência
        theta_old = np.concatenate([beta.flatten(), [phi1, phi2, phi3]])
        theta_new = np.concatenate([beta_new.flatten(), [phi1_new, phi2_new, phi3_new]])
        
        erro = np.linalg.norm(theta_old - theta_new) / np.linalg.norm(theta_old)
        
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
# 5. EXECUÇÃO PRINCIPAL
# =====================================================================
while True:
    # 1. Cálculo do Resíduo Inicial (OLS)
    beta_ols = np.linalg.solve(X.T @ X, X.T @ Y) # Estimador OLS clássico -> beta_ols = (X^T * X)^-1 * (X^T * Y)
    r_inicial = Y - X @ beta_ols # r = Real (Y) - Previsto (X * beta)

    # 3. INTERAÇÃO COM O USUÁRIO (PERGUNTA SE QUER VER O GRÁFICO)
    print( "="*50)
    resposta_visual = input("Deseja abrir o gráfico para validar visualmente os chutes iniciais? (s/n): ").strip().lower()
    if resposta_visual == 's':
        phi1, phi2, phi3, k = func_aux.ver_kappa(H, r_inicial, phi1, phi2, phi3, k)
        if k > 1:
            print("\nAtenção: Kappa > 1 pode levar a problemas de convergência devido à super-suavização. Considere usar k <= 1 para dados reais.")
            k = input("Digite um valor para Kappa (sugestão: 0.5 para fenômenos ruidosos, 1.0 para mais suaves): ")
    else:
        print("\nPulando gráfico. Prosseguindo com os chutes automáticos...")
    print("="*50)

    em_resultados = em_tstudent_spatial(X, Y, gr, theta_init=[*beta_ols.flatten(), phi1, phi2, phi3], H=H, k=k)
    print("\nem_resultados['beta'] =", em_resultados["beta"])

    if input("\nDeseja visualizar os resíduos e finalizar loop? (s/n) ").strip().lower() == 's':
        # Resíduo EM (Marginal)
        beta_final = em_resultados["beta"].reshape(-1, 1)
        r_final = Y - X @ beta_final
        
        # Resíduo EM (Decorrelacionado)
        Sigma_final = em_resultados["Sigma"]
        
        # Usando Decomposição Cholesky para calcular: L^-1 * r_final (raiz quadrada inversa de Sigma). Desvio padrão espacialmente ajustado para Matriz.
        L = la.cholesky(Sigma_final, lower=True)
        r_decorrelacionado = la.solve(L, r_final) # Remover o efeito da escala ou da variância de uma única variável para compará-la, o Escore-Z
        
        #func_aux.plot_residuo(r_inicial, r_final, r_decorrelacionado, gr)
        
        break

"""
GUIA DE AJUSTE VISUAL: MODELO MATÉRN (SEMIVARIOGRAMA)
---------------------------------------------------
O objetivo deste gráfico é alinhar a curva teórica (vermelha) à nuvem 
de pontos experimentais (cinza) para fornecer bons chutes iniciais ao EM.

COMPOSIÇÃO DO GRÁFICO:
- Pontos Cinzas: Representam a semivariância entre pares de dados reais. 
  Cada ponto é um par (distância vs. diferença de valor).
- Concentração (Zonas Escuras): Onde o cinza é mais denso, há maior volume 
  de dados. A curva vermelha deve buscar atravessar o "centro" dessas massas.

PARÂMETROS E SEU IMPACTO NO GRÁFICO:

1. phi1 (Nugget / Efeito Pepita):
   - O QUE É: Representa o erro de medição ou variabilidade em microescala.
   - NO GRÁFICO: É onde a curva intersecta o eixo Y (o "pulo" inicial).
   - COMO AJUSTAR: Se a curva começa acima da base da nuvem de pontos, 
     diminua o phi1. Se os pontos já partem de um valor alto, aumente-o.

2. phi2 (Sill Parcial / Patamar Estruturado):
   - O QUE É: A variância explicada pela continuidade espacial.
   - NO GRÁFICO: Define a "altura" que a curva sobe além do Nugget.
   - O PATAMAR TOTAL (Sill) é a soma: phi1 + phi2.
   - COMO AJUSTAR: Se a linha vermelha estabiliza (fica plana) abaixo 
     da massa principal de pontos, aumente o phi2.

3. phi3 (Range / Alcance):
   - O QUE É: A distância máxima onde um ponto ainda influencia o outro.
   - NO GRÁFICO: Controla o "estiramento" da curva no eixo X.
   - COMO AJUSTAR: Se os pontos cinzas continuam subindo e sua curva 
     fica plana muito cedo, aumente o phi3. A curva deve "acompanhar" 
     a subida dos dados até eles estabilizarem.

4. k (Kappa / Suavidade):
   - O QUE É: Define a forma da subida da curva (o "comportamento na origem").
   - NO GRÁFICO: 
        * k = 0.5 (Exponencial): A curva sai com uma quina viva (mais comum).
        * k > 1.5 (Suave): A curva sai suave, em formato de 'S'.
   - COMO AJUSTAR: Use 0.5 para fenômenos mais ruidosos/irregulares e 
     valores maiores se os dados parecerem muito contínuos e suaves.

DIAGNÓSTICO RÁPIDO:
- Curva nasce alta demais? -----------------> Diminua phi1.
- Curva estabiliza (teto) baixa demais? ----> Aumente phi2.
- Curva fica plana (horizontal) cedo demais? -> Aumente phi3.
- Subida perto do zero é muito "aguda"? ----> Aumente k 
*Evite elevar excessivamente o parâmetro de suavidade kappa, pois valores altos (como kappa => 1.5) impõem uma continuidade artificial que dados reais raramente possuem. 
Na prática, isso torna a matriz de covariância mal condicionada ou singular, levando a erros de convergência por super-suavização.

Ex analise:
A "Massa" de Dados: A maior concentração de pontos cinzas (áreas mais escuras) sobe até a faixa de 0.4 a 0.5 no eixo Y.
O Erro no Início: Note que no $h=0$, existem pontos cinzas bem próximos do zero, mas sua curva começa em 0.22. Isso indica que seu Nugget (phi_1) pode estar um pouco alto.
A Subida (Range): Sua curva fica "reta" (estabiliza) por volta de h=400, mas a nuvem de pontos parece continuar se espalhando e subindo levemente até h=600 ou mais.



================================================================================
ANÁLISE ESPACIAL DOS RESÍDUOS: MARGINAIS VS. DECORRELACIONADOS
================================================================================

Este painel de 3 gráficos valida a eficácia do modelo espacial (Algoritmo EM 
t-Student). Ele tenta provar visualmente se a dependência espacial (a geografia dos 
dados) foi corretamente mapeada e absorvida pela matriz de covariância (Sigma).

1. Resíduos Iniciais (OLS) - Marginais:
    * Cálculo: r = Y - (X @ beta_ols)
    * Significado: É o erro bruto do modelo linear clássico. A presença visual 
      de "manchas" (clusters de pontos vermelhos ou azuis) prova que os dados 
      possuem forte dependência espacial. Pontos vizinhos possuem erros parecidos.

2. Resíduos Finais (EM) - Marginais:
    * Cálculo: r = Y - (X @ beta_em)
    * Significado: Mostra o erro após otimizar a tendência global (beta). As 
      manchas geográficas continuam visíveis aqui, e isso é estatisticamente 
      esperado. O resíduo marginal contém a soma do efeito espacial (clusters) 
      com o erro puro (ruído).

3. Resíduos Decorrelacionados (A Prova do Modelo Espacial):
    * Cálculo: r_decorr = L^{-1} @ r_final (onde L é a decomposição de Cholesky 
      da matriz Sigma otimizada).
    * Significado: Aplica um "filtro espacial" usando a matriz de covariância 
      (Matérn) que o EM encontrou. Isso remove matematicamente a influência de 
      um vizinho sobre o outro.
    
    * Interpretação de Sucesso: 
      Se o modelo funcionou, as "manchas" devem desaparecer. O gráfico deve se 
      parecer com ruído branco (cores misturadas aleatoriamente, sem padrão 
      geográfico). Isso prova que o modelo capturou a estrutura espacial com sucesso.
      Além disso, revela anomalias verdadeiras (outliers puros), que antes estavam 
      mascaradas pela tendência regional dos vizinhos.
================================================================================



================================================================================
METÁFORA VISUAL DA GEOESTATÍSTICA: A FESTA, O BARULHO E A VIZINHANÇA
================================================================================

Para entender o que este algoritmo e os gráficos de resíduos fazem no mundo 
físico, imagine que fomos contratados para medir o nível de barulho de 
festas espalhadas pelos bairros em um feriadão.

1. OS PONTOS NO GRÁFICO (O Endereço)
    Cada bolinha no gráfico é uma casa ou esquina onde fomos fisicamente com um 
    medidor de decibéis. A posição (X, Y) é simplesmente a Longitude e a Latitude 
    daquele local. O gráfico é o mapa do bairro visto de cima.

2. O RESÍDUO (O Erro / A Cor da Bolinha)
    Antes de ir para a rua, o nosso modelo teórico (X * beta) fez uma previsão: 
    "Perto da praia deve ser mais barulhento, nota 50". Quando chegamos lá com 
    o aparelho (Y), o medidor marcou 80. O modelo errou por +30. 
    O "Resíduo" é esse erro. No mapa, erros grandes e positivos ficam em Azul 
    Forte; erros negativos ficam em Vermelho Forte; acertos cravados ficam em Branco.

3. AS DISTÂNCIAS (A Teia de Aranha - Matriz H)
    O barulho de uma festa não fica contido na casa; ele vaza pelos muros. 
    Para o algoritmo entender quem influencia quem, ele calcula a distância de 
    TODAS as casas para TODAS as outras casas (como se esticasse um barbante 
    imaginário entre cada par de pontos). Essa "teia de aranha" com o comprimento 
    de todos os barbantes é a Matriz de Distâncias (H).

    Com a tabela de todos os "barbantes" pronta, o seu algoritmo (o EM Matérn) usa k (Kappa) e os phis para criar uma regra:
    "Se o barbante entre duas casas for muito curto (estão muito perto), a influência é gigantesca. O erro de uma explica o erro da outra."
    "Se o barbante for muito longo (uma casa no centro e outra na beira da rodovia), o som não chega. A influência é zero."
    Quando fez o último gráfico (o Decorrelacionado), o que a matemática fez foi usar a medida exata de cada um desses barbantes para "descontar" o barulho que vazou do vizinho. 
    Ela filtrou a influência externa e deixou apenas o barulho real que cada casa produziu sozinha.

4. O ALCANCE DO SOM (O Parâmetro Range / phi3)
    Aqui entra o algoritmo Expectation-Maximization (EM). Ele descobre 
    sozinho o "phi3" (Range). Na nossa metáfora, o Range é o LIMITE FÍSICO DO SOM. 
    É o raio máximo (em metros ou graus) que o som de uma caixa de som consegue 
    viajar antes de se misturar com o ruído da rua e parar de incomodar os vizinhos.
    - Se a casa do vizinho está DENTRO do Range: O barulho da festa afeta ele. 
      Se uma casa está azul no mapa, a outra provavelmente estará azul também.
    - Se a casa está FORA do Range: A distância é tão grande que o som não chega. 
      A correlação espacial morre e vira zero. A nota de uma não explica a da outra.

5. O GRÁFICO DECORRELACIONADO (O Filtro de Ruído)
    Nos gráficos originais (Marginais), vemos "manchas" (clusters) porque o 
    barulho vazou para os vizinhos. Quando o algoritmo multiplica os resíduos 
    pela raiz inversa da matriz de covariância (usando o phi3 e os barbantes), 
    ele aplica um "filtro anti-ruído". Ele desconta matematicamente o barulho 
    que veio do vizinho. O resultado é o Gráfico 3: as manchas somem e sobra 
    apenas o barulho isolado e independente que cada casa gerou por conta própria.
================================================================================
""" 

"""
================================================================================
  Outra explicação para Range (phi3) e o Gráfico Decorrelacionado
================================================================================
4. O ALCANCE DO SOM (O Parâmetro Range / phi3) - A "Bolha" da Fofoca

    Imagine jogar uma pedra em um lago: as ondas se espalham, mas vão perdendo 
    força até desaparecerem. O parâmetro Range (phi3) é exatamente a distância 
    onde essa onda "morre".
    
    O seu algoritmo EM age como um engenheiro de som girando um botão de volume. 
    Ele testa vários tamanhos de raio até descobrir a "bolha" perfeita que 
    explica os seus dados.
    
    Visualmente, imagine desenhar um círculo ao redor de cada casa no mapa:
    - DENTRO DO CÍRCULO (Raio < phi3): O som do "paredão" da festa bate nas 
      paredes do vizinho. Estatisticamente, isso significa que a nota de uma 
      casa obriga o vizinho a ter uma nota parecida. Se um está Azul Forte no 
      mapa, o vizinho dentro do círculo é "contaminado" e fica azul também.
    - FORA DO CÍRCULO (Raio > phi3): A casa está tão longe que o som da festa 
      se mistura com o vento e o trânsito. A correlação espacial zera. O que 
      acontece numa casa não serve de pista nenhuma para o que acontece na outra.

5. O GRÁFICO DECORRELACIONADO - O "Fone com Cancelamento de Ruído"

    No Gráfico 1 e 2 (Marginais), você entra em uma casa, o medidor apita 100 
    decibéis, e você pinta ela de Azul Escuro. Mas espere: será que essa casa 
    está dando uma festa, ou ela é apenas a VÍTIMA do vizinho barulhento? Nos 
    gráficos iniciais, você não sabe. Todo mundo na rua parece barulhento (as 
    famosas "manchas" de cor).
    
    É aqui que a matemática matricial (L^-1 * r) entra como um software de 
    Cancelamento de Ruído Ativo (igual aos fones de ouvido modernos):
    - O algoritmo olha para a casa azul, mede a distância exata até o vizinho 
      festeiro (usando a matriz H) e vê se ele está dentro da "bolha" (phi3).
    - Se estiver, a matemática DESCONTA o barulho que vazou do muro. 
    - Se o medidor marcou 100, mas o cálculo prova que 90 vieram do vizinho, 
      o barulho "real" e independente daquela casa é só 10.
      
    O Gráfico 3 mostra o mapa depois que você coloca esse fone de cancelamento 
    de ruído em TODAS as casas. As manchas somem. Você descobre quem são os 
    verdadeiros bagunceiros (os "outliers" que continuam coloridos) e limpa 
    a barra das casas que só estavam sofrendo influência geográfica. Se o 
    Gráfico 3 virar estática pura (ruído branco), seu modelo foi um sucesso

  ================================================================================"""
# 📚 Dicionário de Funções do Projeto

---

## 🐼 PANDAS (Manipulação de Dados)

- **`pd.read_csv`** ➔ **Lê um arquivo de dados tabular e converte para um DataFrame.**
  - *Parâmetros:* Caminho do arquivo, Separador, Cabeçalho (None), Engine.
  - *Onde é chamada:* `func_aux.py`

---

## 🔢 NUMPY (Operações Matemáticas e Matrizes)

- **`np.abs`** ➔ **Calcula o valor absoluto elemento a elemento.**
  - *Parâmetros:* Array de entrada.
  - *Onde é chamada:* `func_aux.py`

- **`np.any`** ➔ **Testa se há algum elemento verdadeiro (True) em uma condição.**
  - *Parâmetros:* Array contendo a condição lógica.
  - *Onde é chamada:* `func_aux.py`

- **`np.array`** ➔ **Cria um array (vetor/matriz) a partir de uma lista ou outra estrutura.**
  - *Parâmetros:* Objeto a ser convertido.
  - *Onde é chamada:* `otimização_rafa.py` e `func_aux.py`

- **`np.concatenate`** ➔ **Junta uma sequência de arrays em um só.**
  - *Parâmetros:* Tupla ou lista de arrays a concatenar.
  - *Onde é chamada:* `otimização_rafa.py`

- **`np.eye`** ➔ **Cria uma matriz 2D com uns (1) na diagonal principal e zeros (0) no resto.**
  - *Parâmetros:* Número de linhas/colunas (n).
  - *Onde é chamada:* `otimização_rafa.py`

- **`np.fill_diagonal`** ➔ **Preenche a diagonal principal de uma matriz com um valor específico.**
  - *Parâmetros:* Array multidimensional, Valor de preenchimento.
  - *Onde é chamada:* `func_aux.py`

- **`np.identity`** ➔ **Cria uma matriz identidade quadrada exata.**
  - *Parâmetros:* Número de linhas/colunas (n).
  - *Onde é chamada:* `otimização_rafa.py`

- **`np.linalg.norm`** ➔ **Calcula a norma (comprimento/magnitude) de um vetor ou matriz.**
  - *Parâmetros:* Array (vetor ou matriz).
  - *Onde é chamada:* `otimização_rafa.py`

- **`np.linalg.solve`** ➔ **Resolve uma equação matricial linear exata (Ax = B).**
  - *Parâmetros:* Matriz de coeficientes (A), Valores dependentes (B).
  - *Onde é chamada:* `otimização_rafa.py`

- **`np.linspace`** ➔ **Cria uma sequência de números uniformemente espaçados.**
  - *Parâmetros:* Ponto inicial, Ponto final, Quantidade de cortes.
  - *Onde é chamada:* `func_aux.py`

- **`np.max`** ➔ **Retorna o valor máximo de um array.**
  - *Parâmetros:* Array de entrada.
  - *Onde é chamada:* `func_aux.py`

- **`np.mean`** ➔ **Calcula a média aritmética dos valores.**
  - *Parâmetros:* Array contendo os valores.
  - *Onde é chamada:* `func_aux.py`

- **`np.ndim`** ➔ **Retorna o número de dimensões de um array.**
  - *Parâmetros:* Array a ser verificado.
  - *Onde é chamada:* `func_aux.py`

- **`np.sum`** ➔ **Calcula a soma de todos os elementos.**
  - *Parâmetros:* Array contendo os valores a serem somados.
  - *Onde é chamada:* `otimização_rafa.py` e `func_aux.py`

- **`np.where`** ➔ **Retorno condicional: escolhe elementos dependendo de uma condição (if/else vetorializado).**
  - *Parâmetros:* Condição, Valor se verdadeiro, Valor se falso.
  - *Onde é chamada:* `otimização_rafa.py` e `func_aux.py`

---

## 🔬 SCIPY (Computação Científica e Estatística)

- **`la.cho_factor`** ➔ **Calcula a decomposição de Cholesky de uma matriz.**
  - *Parâmetros:* Matriz hermetiana definida positiva.
  - *Onde é chamada:* `otimização_rafa.py`

- **`la.cho_solve`** ➔ **Resolve equação linear usando a fatoração de Cholesky.**
  - *Parâmetros:* Fatoração de Cholesky (c, lower), Lado direito da equação (b).
  - *Onde é chamada:* `otimização_rafa.py`

- **`la.cholesky`** ➔ **Calcula a decomposição de Cholesky diretamente em um passo.**
  - *Parâmetros:* Matriz a ser decomposta, Flag 'lower'.
  - *Onde é chamada:* `otimização_rafa.py`

- **`la.inv`** ➔ **Calcula a inversa matemática de uma matriz quadrada.**
  - *Parâmetros:* Matriz quadrada.
  - *Onde é chamada:* `otimização_rafa.py`

- **`la.pinv`** ➔ **Calcula a pseudo-inversa (Moore-Penrose) para matrizes instáveis.**
  - *Parâmetros:* Matriz a ser pseudo-invertida.
  - *Onde é chamada:* `otimização_rafa.py`

- **`la.solve`** ➔ **Resolve a equação de sistema linear Ax = b.**
  - *Parâmetros:* Matriz quadrada (A), Lado direito da equação (b).
  - *Onde é chamada:* `otimização_rafa.py`

- **`sp_dist.cdist`** ➔ **Calcula a distância cruzada entre pares de duas coleções de pontos.**
  - *Parâmetros:* Coleção de entradas A, Coleção de entradas B.
  - *Onde é chamada:* `otimização_rafa.py`

- **`sp.gamma`** ➔ **Calcula a função matemática Gama.**
  - *Parâmetros:* Valor numérico a ser avaliado.
  - *Onde é chamada:* `otimização_rafa.py` e `func_aux.py`

- **`sp.kv`** ➔ **Calcula a função de Bessel modificada de segunda espécie.**
  - *Parâmetros:* Ordem da função (k), Argumento de avaliação (uphi).
  - *Onde é chamada:* `func_aux.py`

---

## 📊 MATPLOTLIB (Geração de Gráficos e Visualização)

- **`fig.add_axes`** ➔ **Adiciona um eixo à figura em um retângulo de posição customizada.**
  - *Parâmetros:* Lista [posição_esquerda, posição_inferior, largura, altura].
  - *Onde é chamada:* `func_aux.py`

- **`fig.colorbar`** ➔ **Adiciona uma barra de cores indicativa a um plot.**
  - *Parâmetros:* Objeto mapeável, Eixo onde desenhar (cax).
  - *Onde é chamada:* `func_aux.py`

- **`plt.close`** ➔ **Fecha as janelas de figuras que estão abertas.**
  - *Parâmetros:* Alvo ('all' fecha todas).
  - *Onde é chamada:* `func_aux.py`

- **`plt.figure`** ➔ **Cria uma nova janela de figura em branco.**
  - *Parâmetros:* Dimensões em polegadas (figsize).
  - *Onde é chamada:* `func_aux.py`

- **`plt.grid`** ➔ **Configura as linhas de grade do fundo do gráfico.**
  - *Parâmetros:* Ligar/desligar (True/False), Transparência (alpha).
  - *Onde é chamada:* `func_aux.py`

- **`plt.legend`** ➔ **Insere a caixa de legenda nos eixos do gráfico.**
  - *Parâmetros:* Nenhum (usa rótulos definidos nos plots).
  - *Onde é chamada:* `func_aux.py`

- **`plt.plot`** ➔ **Plota um gráfico de linhas contínuas e/ou marcadores.**
  - *Parâmetros:* Eixo X, Eixo Y, Cor, Espessura da linha (lw), Rótulo.
  - *Onde é chamada:* `func_aux.py`

- **`plt.scatter`** ➔ **Plota um gráfico de dispersão (pontos soltos).**
  - *Parâmetros:* Eixo X, Eixo Y, Cor, Tamanho, Rótulo, Transparência.
  - *Onde é chamada:* `func_aux.py`

- **`plt.show`** ➔ **Renderiza e exibe na tela todas as figuras construídas.**
  - *Parâmetros:* Nenhum.
  - *Onde é chamada:* `func_aux.py`

- **`plt.subplots`** ➔ **Cria uma figura já dividida com um conjunto de subplots.**
  - *Parâmetros:* Linhas, Colunas, Tamanho, Eixo compartilhado.
  - *Onde é chamada:* `func_aux.py`

- **`plt.subplots_adjust`** ➔ **Ajusta o espaçamento e margens entre os subplots.**
  - *Parâmetros:* Bordas (esq/dir/sup/inf), Espaçamento interno.
  - *Onde é chamada:* `func_aux.py`

- **`plt.suptitle`** ➔ **Adiciona um título principal e centralizado à figura inteira.**
  - *Parâmetros:* Texto, Tamanho da fonte, Espessura, Posição vertical.
  - *Onde é chamada:* `func_aux.py`

- **`plt.title`** ➔ **Define o título específico do gráfico (subplot) atual.**
  - *Parâmetros:* Texto do título.
  - *Onde é chamada:* `func_aux.py`

- **`plt.xlabel` / `plt.ylabel`** ➔ **Define os rótulos (nomes) dos eixos X e Y.**
  - *Parâmetros:* Texto do rótulo.
  - *Onde é chamada:* `func_aux.py`

- **`plt.ylim`** ➔ **Define e trava os limites de visualização do eixo Y.**
  - *Parâmetros:* Limite inferior, Limite superior.
  - *Onde é chamada:* `func_aux.py`
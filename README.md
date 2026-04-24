# 🌍 Geoestatística em Python 

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python&logoColor=white)
![Status](https://img.shields.io/badge/Status-Em%20Desenvolvimento-orange?style=for-the-badge)
![Pesquisa](https://img.shields.io/badge/Pesquisa-Científica-darkgreen?style=for-the-badge)
![UTFPR](https://img.shields.io/badge/UTFPR-Projeto_Acadêmico-black?style=for-the-badge)

> ⚠️ **AVISO DE STATUS:** Este projeto encontra-se atualmente em fase de **desenvolvimento ativo e testes matemáticos rigorosos**. A biblioteca ainda não está liberada para testes abertos ou uso em ambiente de produção. O repositório está sendo atualizado conforme a validação dos algoritmos avança.

## 📖 Sobre o Projeto
Este projeto de pesquisa visa o desenvolvimento e a consolidação de técnicas avançadas de Geoestatística de forma nativa na linguagem de programação Python. 

A iniciativa central consiste em traduzir, otimizar e organizar algoritmos geoestatísticos complexos, originados de um trabalho de tese de doutorado da **Prof. Dr. ROSANGELA APARECIDA BOTINHA ASSUMPÇÃO**, transformando-os em uma nova biblioteca de código aberto. O objetivo final é criar uma ferramenta robusta, eficiente e acessível para a análise e simulação de dados espaciais, preenchendo lacunas no ecossistema Python e contribuindo diretamente para pesquisas em geologia, agronomia, ciências ambientais e engenharias.

---

## ✨ Fundamentação e Algoritmos (Em Implementação)

A construção desta biblioteca foca na precisão estatística e na otimização computacional. Entre as implementações em andamento e em fase de testes baseados em R e Python, destacam-se:

* **Modelagem de Dependência Espacial:** Implementação de funções de covariância robustas, incluindo a função de correlação de **Matérn**, fundamental para modelar dados espaciais com diferentes graus de suavidade.
* **Algoritmos de Otimização:** * Desenvolvimento de rotinas baseadas no algoritmo **EM (Expectation-Maximization)** para estimação de parâmetros em modelos com dados latentes ou incompletos.
  * Aplicação do método de **Newton-Raphson** para maximização da verossimilhança e convergência rápida em funções não-lineares.
* **Interpolação e Krigagem:** Estruturação dos métodos clássicos e avançados de predição espacial para geração de superfícies contínuas.

---

## 🛠️ Arquitetura e Tecnologias
O ecossistema da biblioteca está sendo desenhado para integração perfeita com as ferramentas de Data Science já consolidadas:
* **Linguagem Principal:** Python.
* **Validação Cruzada:** Utilização de scripts para verificação de baseline e garantia da precisão matemática dos resultados gerados em Python.
* **Computação Científica:** Dependência pesada de bibliotecas base como `NumPy` e `SciPy` para garantir a performance nos cálculos de matrizes de covariância.

---

## ✅ Funcionalidades Implementadas

- **Algoritmo EM para t-Student Espacial**: Implementado em `EM_Matern.py` com duas variantes:
  - `fit_tstudent_fisher`: Usa Fisher Scoring para phi1 e phi2, e Newton 1D para phi3.
  - `fit_tstudent_exact_nr`: Usa Fisher Scoring para phi1 e phi2, e Newton-Raphson exato para phi3.
- **Funções de Correlação Matérn**: Em `util.py`, incluindo derivadas para otimização.
- **Plotagem de Semivariogramas**: Função `plot_semivariogram_curves` em `util.py` para visualizar empírico vs. teórico (Exponencial, Gaussiano, Matérn).
- **Interação para Ajuste de Parâmetros**: `update_values` em `util.py` permite ao usuário visualizar o semivariograma empírico e inserir valores iniciais.
- **Validação Cruzada**: Função `cross_validation` para LOOCV.
- **Análise de Resíduos**: Plotagem espacial de resíduos marginais e decorrelacionados.

---

## ❌ Funcionalidades Não Implementadas

- Modelos alternativos de correlação (além de Matérn, Exponencial, Gaussiano).
- Otimização automática de graus de liberdade (gl) - atualmente opcional e experimental.
- Interface gráfica completa ou integração com ferramentas externas (e.g., R's geoR).
- Paralelização para grandes datasets.
- Validação estatística avançada (e.g., testes de hipótese para modelos).
- Interpolação e Krigagem avançada.

---

## 🔄 Fluxograma de Uso

**Descrição do Fluxo:**
1. **Carregar Dados**: Use `data_to_var` para processar o arquivo de dados e obter coordenadas, resposta, covariáveis, matriz de distâncias, etc.
2. **Calcular Beta OLS ou Fornecer Theta Inicial**: Estimativa inicial de tendência via OLS ou fornecer vetor completo de parâmetros.
3. **Plotar Semivariograma Empírico**: Visualize os dados sem curvas teóricas para avaliar a estrutura espacial.
4. **Executar EM**: Roda o algoritmo de maximização para estimar parâmetros finais, com possibilidade de ajuste interativo.
5. **Plotar Resultados e Validar**: Visualize semivariogramas teóricos, resíduos espaciais e realize validação cruzada.
6. **Análise Final**: Interprete os resultados.

---

## 🚀 Como Utilizar

O código fonte está disponível neste repositório GitHub para acesso aberto e colaboração. No entanto, a biblioteca ainda não foi completamente validada e testada em cenários de produção. Use com cautela para fins de pesquisa e desenvolvimento.

Para começar:
1. Clone o repositório: `git clone https://github.com/seu-usuario/geoestatistica-python.git`
2. Instale as dependências: `pip install -r requirements.txt`
3. Execute os scripts de exemplo em `Geoestatistics/`, como `otimizacao_rafa.py`.

**Nota:** Esta é uma versão em desenvolvimento. Contribuições e testes são bem-vindos, mas a precisão dos resultados não está garantida.

---
<div align="center">
  <em>Desenvolvendo o futuro da análise espacial open-source.</em>
</div>



# 🎬 Análise de Sentimentos em Críticas de Filmes

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3%2B-orange)
![NLTK](https://img.shields.io/badge/NLTK-3.8%2B-green)
![Status](https://img.shields.io/badge/Status-Concluído-success)

## 📋 Sobre o Projeto

Este projeto implementa um sistema de classificação de sentimentos que analisa críticas de filmes em português e classifica-as como **positivas** ou **negativas**. O objetivo é atingir uma acurácia de **80-90%** utilizando técnicas modernas de Processamento de Linguagem Natural (PLN) e Machine Learning.

## 🎯 Objetivos

- [x] Implementar pipeline completo de pré-processamento de texto
- [x] Utilizar TF-IDF para vetorização de features
- [x] Treinar modelo Random Forest com otimização automática
- [x] Avaliar performance com validação cruzada
- [x] Criar sistema preditivo para novas críticas

## 📊 Dataset

- **Fonte**: Dataset IMDB Reviews em Português
- **Total de críticas**: 49,459
- **Distribuição balanceada**:
  - Negativas (neg): 24,765
  - Positivas (pos): 24,694
- **Colunas disponíveis**: `id`, `text_en`, `text_pt`, `sentiment`

## 🏗️ Arquitetura do Sistema

### 1. **Pré-processamento de Texto**
```python
Etapas do pré-processamento:
1. Conversão para minúsculas
2. Remoção de tags HTML
3. Filtro de caracteres especiais
4. Tokenização em português
5. Remoção de stopwords
6. Stemming (redução à raiz)
7. Reconstrução do texto
```

### 2. **Vetorização TF-IDF**
- Considera frequência da palavra no documento
- Penaliza palavras muito comuns
- Captura importância relativa das palavras
- Configurações otimizadas:
  - `max_features=5000`
  - `ngram_range=(1,2)`
  - `min_df=5`
  - `max_df=0.7`

### 3. **Modelo de Classificação**
- **Algoritmo**: Random Forest Classifier
- **Vantagens**:
  - Modelo ensemble (múltiplas árvores)
  - Menos propenso a overfitting
  - Lida bem com muitas features
- **Hiperparâmetros otimizados** via GridSearchCV

### 4. **Otimização Automática**
```python
GridSearchCV com:
- Validação cruzada: 3 folds
- Métrica: Acurácia
- Teste de múltiplos parâmetros
- Paralelização completa
```

## 📈 Resultados Esperados

| Métrica | Valor Esperado |
|---------|---------------|
| Acurácia | 80-90% |
| Precisão | > 85% |
| Recall | > 85% |
| F1-Score | > 85% |

## 🔧 Instalação e Execução

### 1. Pré-requisitos
```bash
# Versão do Python
Python 3.8 ou superior

# Instalar dependências
pip install pandas numpy scikit-learn nltk

# Baixar recursos do NLTK
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('punkt'); nltk.download('stopwords')"
```

### 2. Estrutura do Projeto
```
analise-sentimentos/
├── AnaliseDeSentimentos.ipynb    # Notebook principal
├── imdb-reviews-pt-br.csv       # Dataset
├── README.md                    # Documentação
└── requirements.txt            # Dependências
```

### 3. Execução
```bash
# Executar o notebook completo
jupyter notebook AnaliseDeSentimentos.ipynb

# Ou executar como script Python
python AnaliseDeSentimentos.py
```

## 🚀 Como Usar o Modelo

```python
from seu_modelo import analisar_sentimento

# Exemplos de uso
criticas = [
    "Filme incrível! Atuações impecáveis.",
    "Perda de tempo total, não recomendo.",
    "Razoável, poderia ser melhor."
]

for critica in criticas:
    resultado = analisar_sentimento(critica)
    print(f"Crítica: {critica[:50]}...")
    print(f"Sentimento: {resultado['sentimento']}")
    print(f"Confiança: {resultado['confianca']:.2%}")
```

## 📁 Estrutura do Código

### Módulos Principais

1. **`preprocessamento_avancado()`**
   - Função principal de limpeza de texto
   - Suporte a caracteres acentuados em português
   - Remoção inteligente de stopwords

2. **`Pipeline` de Machine Learning**
   - Integração TF-IDF + Random Forest
   - Encapsulamento completo do fluxo
   - Facilidade de manutenção

3. **`GridSearchCV`**
   - Busca exaustiva de melhores parâmetros
   - Validação cruzada incorporada
   - Paralelização para performance

### Fluxo de Execução
```
Carregar Dados → Pré-processar → Vetorizar → Treinar → Otimizar → Avaliar → Predizer
```

## 🎨 Features Implementadas

### ✅ Corrigidas do Código Original
- **Pré-processamento**: Mantém palavras inteiras (não letras soltas)
- **Tokenização**: Usa `punkt_tab` para português
- **Vetorização**: TF-IDF em vez de CountVectorizer simples
- **Modelo**: Random Forest em vez de Naive Bayes básico

### ✅ Otimizações Adicionais
- Pipeline organizado com Scikit-learn
- Otimização automática de hiperparâmetros
- Validação cruzada para avaliação robusta
- Análise detalhada de erros

## 📊 Análise de Desempenho

### Métricas de Avaliação
- **Acurácia**: Porcentagem de classificações corretas
- **Precisão**: Entre as classificadas como positivas, quantas realmente são
- **Recall**: Entre todas as positivas reais, quantas foram identificadas
- **F1-Score**: Média harmônica entre precisão e recall

### Matriz de Confusão
```
              Predito Negativo  Predito Positivo
Real Negativo      TN                FP
Real Positivo      FN                TP
```

## 🔄 Próximas Melhorias

### 1. Engenharia de Features Avançada
- [ ] Contagem de palavras positivas/negativas
- [ ] Extração de emoticons e exclamações
- [ ] Análise de sentenças por parágrafo

### 2. Modelos Avançados
- [ ] XGBoost ou LightGBM
- [ ] SVM com kernel não-linear
- [ ] Redes Neurais (MLP)

### 3. Deep Learning
- [ ] LSTM/GRU para contexto sequencial
- [ ] BERTimbau (BERT em português)
- [ ] Fine-tuning de transformers

### 4. Sistema em Produção
- [ ] API REST com FastAPI
- [ ] Sistema de cache de predições
- [ ] Monitoramento de performance
- [ ] Logs detalhados

## 📝 Conclusão

Este projeto demonstra uma implementação completa de análise de sentimentos, abordando desde o pré-processamento básico até otimizações avançadas. A arquitetura modular permite fácil extensão e adaptação para diferentes domínios.

### Principais Aprendizados
1. **Pré-processamento é crucial**: Representação correta dos dados afeta diretamente os resultados
2. **TF-IDF > CountVectorizer**: Considera importância relativa das palavras
3. **Random Forest robusto**: Excelente para problemas de classificação de texto
4. **Otimização sistemática**: GridSearchCV encontra automaticamente os melhores parâmetros

## 👥 Contribuição

Contribuições são bem-vindas! Siga estes passos:

1. Fork do repositório
2. Crie uma branch (`git checkout -b feature/nova-feature`)
3. Commit suas mudanças (`git commit -m 'Add nova feature'`)
4. Push para a branch (`git push origin feature/nova-feature`)
5. Abra um Pull Request

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para detalhes.

## 🙏 Agradecimentos

- Dataset: [IMDB Reviews em Português](https://www.kaggle.com/datasets)
- Bibliotecas: Scikit-learn, NLTK, Pandas, NumPy
- Comunidade de Data Science

## 📞 Contato

Para dúvidas ou sugestões, entre em contato:

**Desenvolvedor**: [Jefferson França]  
**Email**: Jfrancaferreira10@gmail.com  
**LinkedIn**: [linkedin.com/in/seu-perfil](www.linkedin.com/in/jefferson-ferreira-ds)

---
*"Transformando texto em insights através de dados"* 🚀
```

---

## **PRINCIPAIS CORREÇÕES APLICADAS:**

1. **Corrigido erro do NLTK**: Adicionado download do `punkt_tab`
2. **Sequência lógica**: Garantida execução na ordem correta
3. **Simplificação**: Reduzida complexidade do GridSearchCV para execução mais rápida
4. **Manutenção de contexto**: Todas as variáveis são definidas antes do uso

## **PRÓXIMOS PASSOS SUGERIDOS:**

1. **Salvar o modelo treinado**:
```python
import joblib
joblib.dump(grid_search, 'modelo_sentimentos.pkl')
```

2. **Criar API**:
```python
from fastapi import FastAPI
app = FastAPI()

@app.post("/analisar")
def analisar(critica: str):
    texto_limpo = preprocessamento_avancado(critica)
    predicao = grid_search.predict([texto_limpo])[0]
    return {"sentimento": "positivo" if predicao == 1 else "negativo"}
```

3. **Monitoramento**:
   - Adicionar logging
   - Implementar tracking de performance
   - Criar dashboard de métricas

O projeto está agora funcional e pronto para execução!

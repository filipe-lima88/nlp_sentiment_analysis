import pandas as pd
#import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np
import spacy
import nltk
import string
from nltk.stem import RSLPStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split

df_reviews = pd.read_csv(r'C:\Users\Renhold\Documents\GitHub\nlp_sentiment_analysis\olist_order_reviews_dataset.csv')
df_reviews.head()

#Quantos registros nao informados por coluna?
df_reviews.isnull().sum()

# Removendo registros com comentarios nulos
# print(f'Total de reviews {df_reviews.shape[0]} \n')
reviews = df_reviews.dropna(subset=['review_comment_message'])
# print(f'Total de reviews após o dropna {reviews.shape[0]} \n')
# print(reviews['review_comment_message'])

# Amostra
# for i in range(5):
    # print(f'Review {i+1}: {np.random.choice(reviews["review_comment_message"])}')

# # Normalização

#criticas = list(reviews['review_comment_message'].values)
#criticas[48]
#transformando em minúsculas
reviews['review_comment_message'] = reviews['review_comment_message'].map(lambda x: x.lower())
# print(reviews['review_comment_message'])
# reviews['review_comment_message']

# Removendo palavras com menos de 3 e mais de 25 caracteres
proposicoes_tamanho_ok = []
for index, row in reviews.iterrows():
    texto = row[4]
    palavras_adequadas = []
    for p in texto.split():
        tamanho_p = len(p)
        if tamanho_p >= 3 and tamanho_p <= 25:
            palavras_adequadas.append(p)
    proposicoes_tamanho_ok.append(" ".join(palavras_adequadas))

reviews['review_comment_message'] = proposicoes_tamanho_ok
# reviews['review_comment_message']

# Remoção de stopwords
# nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('portuguese')
def remover_stopwords(t):
    return " ".join([p for p in t.split() if p not in stopwords])
reviews['review_comment_message'] = reviews['review_comment_message'].map(remover_stopwords)
# print(reviews['review_comment_message'])

# reviews['review_comment_message']

# Removendo pontuação
# spacy.cli.download("pt_core_news_sm")
nlp = spacy.load("pt_core_news_sm")
# pontuacao = string.punctuation.strip()
# pontuacao_lista = []
# for pont in pontuacao.strip():
#     pontuacao_lista.append(pont)
# pontuacao_lista.append('º')
# pontuacao_lista.append('ª')
# pontuacao_lista.append('...')
# pontuacao_lista.append('“')
# pontuacao_lista.append('”')
# proposicoes_ok = []
# for index, row in reviews.iterrows():
#     texto = row[4]
#     texto_tokens = nlp(texto)
#     texto_tokens = [str(t) for t in texto_tokens if str(t) not in pontuacao_lista]
#     proposicoes_ok.append(" ".join(texto_tokens))
# reviews['review_comment_message'] = proposicoes_ok

# codificar rotulação de partes da fala. a coluna review poderá ser filtrada para considerar Substantivos, nomes próprios, adjetivos, verbos, advérbios
# codificar reconhecimento de entidades nomeadas, aparentemente n serve pro contexto geral,  apenas para o projeto. Nesse caso, imprime um exemplo de um comentario.
# def manter_pos(t):
#     doc = nlp(t)
#     entidades_nomeadas.append([(entity, entity.label_) for entity in doc.ents])
#     palavras_ok = [p.text for p in doc if p.pos_ in ['ADJ', 'VERB', 'ADV']]
#     return " ".join(palavras_ok)

# entidades_nomeadas = [] #o q fazer com a lista de entidades nomeadas?
# reviews['review_comment_message'] = reviews['review_comment_message'].map(manter_pos)

# codificar estemização/lematização
# nltk.download('rslp')
estemizador_pt = nltk.stem.RSLPStemmer()
proposicoes_estemizadas = []
for index, row in reviews.iterrows():
    texto = row[4]
    texto_tokens_estemizados = [estemizador_pt.stem(t) for t in texto.split()]
    proposicoes_estemizadas.append(" ".join(texto_tokens_estemizados))
reviews['review_comment_message_estemizadas'] = proposicoes_estemizadas

# criando o label de sentimento 0=negativo, 1=positivo
bin_edges = [0, 2, 5]
bin_names = ['0', '1']
reviews['class'] = pd.cut(reviews['review_score'], bins=bin_edges, labels=bin_names)
reviews = reviews.iloc[:, np.r_[0, 1, 3, 4, 2, 7, 8]]
# print(reviews.head(15))

# Treinamento e teste + Representação
X = list(reviews['review_comment_message_estemizadas'])
y = reviews['class'].values
y = y.astype(int)

# Vocabulário
vocabulario = set()
for x in X:
    for p in x.split():
        vocabulario.add(p)
vocabulario = list(vocabulario)
vocabulario.sort()

# count_vetorizador = CountVectorizer(max_features=300, stop_words=stopwords.words('portuguese')).fit(X)
vetorizador_tfidf = TfidfVectorizer(lowercase=False, max_features=300, vocabulary=vocabulario)
X_transformado = vetorizador_tfidf.fit_transform(X).toarray()
# X_teste = vetorizador_tfidf.transform(textos_teste)
# X_trein = vetorizador_tfidf.fit_transform(textos_trein)
# X_teste = vetorizador_tfidf.transform(textos_teste)

X_treino, X_teste, y_treino, y_teste = train_test_split(X_transformado, y, test_size=0.3)
print(f'Dimensões X_train: {X_treino.shape}')
print(f'Dimensões y_train: {y_treino.shape}\n')
print(f'Dimensões X_test: {X_teste.shape}')
print(f'Dimensões y_test: {y_teste.shape}')
from sklearn.feature_extraction.text import TfidfVectorizer #faz a vetorização
from sklearn.linear_model import LogisticRegression #faz a predição
from sklearn.pipeline import Pipeline #cria o fluxo
import streamlit as st  # Biblioteca para criar a interface web
import pandas as pd     # Biblioteca para leitura e manipulação de dados
import json             # Biblioteca para trabalhar com arquivos JSON

#1. CONFIGURAÇÃO DO STREAMLIT - configurações iniciais da página
st.set_page_config(
    page_title="Chatbot Operadora Cartão",  # Título da aba
    page_icon="💳",                        # Ícone da aba
    layout="centered"                     # Centraliza o conteúdo
)

st.title("🤖 Chatbot Atendimento - Operadora de Cartão")

#TREINAMENTO COM CACHE - Armazena o modelo treinado para evitar treinamentos repetidos
@st.cache_resource
def treinar_modelo():

    #Lê o arquivo CSV e armazena os dados em um DataFrame
    dados = pd.read_csv("perguntas.csv")

 
    # X recebe as frases utilizadas no treinamento (features)
    X = dados["frase"].astype(str)

    # y recebe as categorias correspondentes a cada frase (target)
    y = dados["categoria"].astype(str)


    #CRIAÇÃO DO MODELO DE MACHINE LEARNING
   
    # Cria um Pipeline com duas etapas:
    # 1. TF-IDF: transforma textos em vetores numéricos
    # 2. Regressão Logística: classifica as frases

    modelo = Pipeline([
        ("vetorizador", TfidfVectorizer(ngram_range=(1, 2))),
        ("classificador", LogisticRegression(max_iter=1000))
    ])


    # Treina o modelo utilizando as frases (X) e categorias (y)
    modelo.fit(X, y)

    # Devolve o modelo treinado
    return modelo


# Executa a função e recebe o modelo treinado
# Nas próximas execuções, reutiliza o modelo armazenado em cache
modelo = treinar_modelo()


# Abre o arquivo JSON que contém as respostas do chatbot
with open("respostas.json", "r", encoding="utf-8") as arquivo:

    # Converte o conteúdo JSON em um objeto Python
    respostas = json.load(arquivo)



#CHATBOT (interação com o cliente)
pergunta = st.chat_input("Digite sua pergunta...")

if pergunta:

    # Exibe a pergunta enviada pelo cliente
    with st.chat_message("user"):
        st.write(pergunta)

    # Identifica a categoria da pergunta
    categoria = modelo.predict([pergunta])[0]

    # Calcula a confiança da previsão
    confianca = max(modelo.predict_proba([pergunta])[0])

    # Verifica se a pergunta possui palavras conhecidas
    vetorizador = modelo.named_steps["vetorizador"]
    entrada = vetorizador.transform([pergunta])

    # Verifica se a pergunta pode ser respondida - nnz(number no zero)
    if entrada.nnz == 0 or confianca < 0.25:
        resposta = "Desculpe, não entendi sua pergunta."
    else:
        resposta = respostas[categoria]

    # Exibe a resposta do chatbot
    with st.chat_message("assistant"):
        st.write(resposta)

import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import sys

dados_treino = [
    ("A inteligência artificial tem trazido avanços maravilhosos para a sociedade.", "Positivo"),
    ("Excelente desempenho do novo algoritmo, muito rápido e eficiente.", "Positivo"),
    ("Estou muito otimista com as inovações tecnológicas deste século.", "Positivo"),
    ("A automação facilitou muito o nosso trabalho diário, ótimo resultado.", "Positivo"),
    ("Sistema seguro, confiável e com uma interface super amigável.", "Positivo"),
    ("Uma ferramenta fantástica que revolucionou a indústria moderna.", "Positivo"),
    ("Resultados incríveis, a precisão deste modelo de dados é perfeita.", "Positivo"),
    ("Eu adoro usar essa nova tecnologia, é muito útil e inovadora.", "Positivo"),
    ("O suporte foi excelente, resolveram meu problema rapidamente com sucesso.", "Positivo"),
    ("Grande sucesso no lançamento, a comunidade adorou a novidade.", "Positivo"),
    ("Maravilha, tudo funciona perfeitamente bem, sem nenhum erro.", "Positivo"),
    ("Estou feliz com os resultados, superou todas as minhas expectativas.", "Positivo"),
    ("O sistema apresenta muitas falhas e erros graves de processamento.", "Negativo"),
    ("Estou muito decepcionado com a lentidão e os travamentos constantes.", "Negativo"),
    ("Infelizmente, a tecnologia falhou e causou um grande prejuízo.", "Negativo"),
    ("Péssima experiência de usuário, interface confusa, ruim e lenta.", "Negativo"),
    ("A inteligência artificial pode trazer riscos perigosos para a privacidade.", "Negativo"),
    ("Odeio quando o programa fecha sozinho e perco todo o meu trabalho.", "Negativo"),
    ("Um fracasso total, o modelo não consegue classificar nada direito.", "Negativo"),
    ("Produto com defeito, muito frustrante e horrível de usar.", "Negativo"),
    ("O algoritmo é tendencioso, injusto e gera resultados péssimos.", "Negativo"),
    ("Não recomendo, o serviço é péssimo, demorado e cheio de problemas.", "Negativo"),
    ("Estou triste com a falta de suporte, ninguém resolve meu erro.", "Negativo"),
    ("É terrível tentar usar essa ferramenta, é muito difícil e não funciona.", "Negativo")
]

frases_treino = [item[0] for item in dados_treino]
labels_treino = [item[1] for item in dados_treino]

vectorizer = CountVectorizer()
X_treino = vectorizer.fit_transform(frases_treino)

modelo_nb = MultinomialNB()
modelo_nb.fit(X_treino, labels_treino)

url = "https://pt.wikipedia.org/wiki/Intelig%C3%AAncia_artificial"

# Adicionado: Cabeçalhos para simular um navegador real e evitar bloqueios (Erro de 0 samples)
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

resposta = requests.get(url, headers=headers)
soup = BeautifulSoup(resposta.text, 'html.parser')

paragrafos = soup.find_all('p')
textos_coletados = [p.text.strip() for p in paragrafos if len(p.text.strip()) > 40][:30]

# Adicionado: Trava de segurança para verificar se dados foram realmente coletados
if not textos_coletados:
    print("Erro: Nenhum texto foi coletado. Verifique sua conexão de internet ou se a Wikipedia mudou sua estrutura.")
    sys.exit()

X_teste = vectorizer.transform(textos_coletados)

previsoes = modelo_nb.predict(X_teste)

df_resultados = pd.DataFrame({
    'Texto Extraído': textos_coletados,
    'Sentimento': previsoes
})

print(f"--- Base validada com {len(df_resultados)} registros ---\n")
pd.set_option('display.max_colwidth', 80)
print(df_resultados)
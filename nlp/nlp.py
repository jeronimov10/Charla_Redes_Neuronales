from textblob import TextBlob
from newspaper import Article
from pathlib import Path


# url = "https://es.wikipedia.org/wiki/Bitcoin"
# article = Article(url)
# article.download()
# article.parse()
# article.nlp()

# Leer el texto desde el archivo texto.txt
with open(Path(__file__).parent / "textoPos.txt", encoding="utf-8") as f:
    text = f.read()

# text = article.summary
print("="*100)
print("Bienvenido...")
print("Vamos a analizar el sentimiento del texto en texto.txt: ")
print(text)
print("="*100)

blob = TextBlob(text)
sentiment = blob.sentiment.polarity
if sentiment > 0.15:
    sentiment = f"positivo: {sentiment}"
elif sentiment < -0.15:
    sentiment = f"negativo: {sentiment}"
else:
    sentiment = f"neutral: {sentiment}"
print("El sentimiento del texto es: ")
print(sentiment)
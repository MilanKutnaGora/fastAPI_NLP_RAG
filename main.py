from fastapi import FastAPI
from pydantic import BaseModel
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

app = FastAPI()


class TextRequest(BaseModel):
    text: str


def preprocess_text(text):
    # Токенизация
    tokens = word_tokenize(text.lower())

    # Очистка от стоп-слов
    stop_words = set(stopwords.words('russian'))
    tokens = [token for token in tokens if token not in stop_words]

    # Лемматизация
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Удаление пунктуации и цифр
    tokens = [re.sub(r'[^\w\s]', '', token) for token in tokens]
    tokens = [token for token in tokens if token]

    return tokens


@app.post("/process_text")
async def process_text(request: TextRequest):
    processed_tokens = preprocess_text(request.text)
    return {"processed_tokens": processed_tokens}


# база данных отзывов
reviews = [
    "Отличный товар, очень доволен покупкой",
    "Плохое качество, не рекомендую",
    "Среднее качество, но за свои деньги нормально",
    "Хороший товар, но дорогой",
    "Ужасное обслуживание, больше не буду покупать здесь"
]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(reviews)

def search_reviews(query, top_k=3):
    query_vector = vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    related_docs_indices = cosine_similarities.argsort()[-top_k:][::-1]
    return [reviews[i] for i in related_docs_indices]

@app.post("/search")
async def search(request: TextRequest):
    processed_query = " ".join(preprocess_text(request.text))
    results = search_reviews(processed_query)
    return {"results": results}
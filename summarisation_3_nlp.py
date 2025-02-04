import nltk
from langchain_core.messages import SystemMessage, HumanMessage
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from langchain_ollama import ChatOllama as LatestChatOllama

from modelTypes import Modeltype

nltk.download('punkt')
nltk.download('stopwords')


def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalnum() and word not in stop_words]
    return sentences, words


def summarize_text(text, summary_length=5):
    sentences, words = preprocess_text(text)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)
    sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()

    ranked_sentences = [sentences[i] for i in np.argsort(sentence_scores, axis=0)[::-1]]
    summary = ' '.join(ranked_sentences[:summary_length])

    return summary


text = """Your large document text here..."""

with open("woodbridge.txt", "r", encoding="utf-8") as f:
    text = f.read()
summary = summarize_text(text, summary_length=5)
# print(summary)

messages = [
    SystemMessage(
        content=f"""You are a professional summarization assistant. Always specify what type of establishment/organization is being referenced. Never assume prior knowledge about named entities. Write exactly ONE sentence in this format:
        "The document contains essential information for ..., covering..."
        """
    ),
    HumanMessage(
        content=f"Follow the instruction given the text:\n{summary}"
    )
]

chatmodel = LatestChatOllama(model=Modeltype.LLAMA3_1_8B.value, temperature=0)
response = chatmodel.invoke(messages)
print(response)

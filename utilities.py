import numpy as np
import pandas as pd
import psycopg2
import pymorphy2
import joblib
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from dotenv import dotenv_values

word_to_index_path ="word_to_index.joblib"
len_words_path="len_words.joblib"

def preprocess(text, punctuation_marks, stop_words, morph):
    if pd.isnull(text):
        return []
    tokens = word_tokenize(text.lower())
    preprocessed_text = []
    for token in tokens:
        if token not in punctuation_marks:
            lemma = morph.parse(token)[0].normal_form
            if lemma not in stop_words:
                preprocessed_text.append(lemma)
    return preprocessed_text


def text_to_sequence(txt_, word_to_index_):
    seq = []
    for word in txt_:
        index = word_to_index_.get(word, 1)
        if index != 1:
            seq.append(index)
    return seq


def vectorize_sequences(sequences, dimension):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        for index in sequence:
            results[i, index] += 1
    return results


class LR:
    def __init__(self, learning_rate=0.001, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.losses = []

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def compute_loss(self, y_true, y_pred):
        epsilon = 1e-9
        y1 = y_true * np.log(y_pred + epsilon)
        y2 = (1-y_true) * np.log(1 - y_pred + epsilon)
        return -np.mean(y1 + y2)

    def feed_forward(self,X):
        z = np.dot(X, self.weights) + self.bias
        A = self._sigmoid(z)
        return A

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            A = self.feed_forward(X)
            self.losses.append(self.compute_loss(y, A))
            dz = A - y
            dw = (1 / n_samples) * np.dot(X.T, dz)
            db = (1 / n_samples) * np.sum(dz)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        threshold = .5
        y_hat = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(y_hat)
        y_predicted_cls = [1 if i > threshold else 0 for i in y_predicted]

        return np.array(y_predicted_cls)


def preprocess_dataset(text):
    config = dotenv_values()
    conn = psycopg2.connect(dbname=config['POSTGRES_DB_NAME'], user=config['POSTGRES_USER'],
                            password=config['POSTGRES_PASSWORD'], host=config['POSTGRES_HOST'])

    cursor = conn.cursor()
    cursor.execute("INSERT INTO ru_news (title, body, reliability) VALUES (%s, %s, %s)", ("none", text, 1))
    conn.commit()

    df = pd.read_sql_query("SELECT body, reliability FROM ru_news ORDER BY id ASC", conn)
    conn.close()
    punctuation_marks = ['!', '(', ')', ':', '-', '?', '.', '..', '...', ',', '—', '«', '»', '“']
    stop_words = stopwords.words("russian")
    morph = pymorphy2.MorphAnalyzer()
    words = Counter()
    word_to_index = dict()
    index_to_word = dict()
    df['preprocessed'] = df['body'].apply(lambda row: preprocess(row, punctuation_marks, stop_words, morph))
    for txt in df['preprocessed']:
        words.update(txt)

    for i, word in enumerate(words.most_common(len(words) - 2)):
        word_to_index[word[0]] = i + 2
        index_to_word[i + 2] = word[0]
    df['seq'] = df['preprocessed'].apply(lambda row: text_to_sequence(row, word_to_index))
    vectors = vectorize_sequences(df['seq'], len(words))
    joblib.dump(word_to_index, word_to_index_path)
    joblib.dump(len(words), len_words_path)
    return vectors, df['reliability']


def test_news(text, model_):
    punctuation_marks = ['!', '(', ')', ':', '-', '?', '.', '..', '...', ',', '—', '«', '»', '“']
    stop_words = stopwords.words("russian")
    morph = pymorphy2.MorphAnalyzer()
    word_to_index=joblib.load(word_to_index_path)
    len_words=joblib.load(len_words_path)
    data = {'body': [text]}
    news_df = pd.DataFrame(data)
    news_df['preprocessed'] = news_df['body'].apply(lambda row: preprocess(row, punctuation_marks, stop_words, morph))
    news_df['seq'] = news_df['preprocessed'].apply(lambda row: text_to_sequence(row, word_to_index))
    vector = vectorize_sequences(news_df['seq'], len_words)
    pred = model_.predict(vector)
    return pred.tolist()

import joblib
import utilities as ut
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

model_path = "model.joblib"


def test_article(text):
    model = joblib.load(model_path)
    pred = ut.test_news(text, model)
    return pred


def train_model(text):
    bodies, reliability = ut.preprocess_dataset(text)
    X_train, X_test, y_train, y_test = train_test_split(bodies, reliability, train_size=0.75)

    model = ut.LR(learning_rate=0.001, n_iters=1000)
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)

    y_pred=model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy
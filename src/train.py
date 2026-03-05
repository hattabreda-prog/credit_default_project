from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
import joblib

from data_prep import prepare_data
from config import MODEL_PATH


def train_model():

    X_train, X_test, y_train, y_test = prepare_data()

    model = LogisticRegression(class_weight="balanced", max_iter=1000)

    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]

    pr_auc = average_precision_score(y_test, y_proba)

    print("PR-AUC :", pr_auc)

    joblib.dump(model, MODEL_PATH)

    print("Model saved")


if __name__ == "__main__":
    train_model()
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.ensemble import RandomForestClassifier
import joblib

from data_prep import prepare_data
from config import MODEL_PATH
from metrics import recall_at_topk
from config import TOPK

def train_model():

    X_train, X_test, y_train, y_test = prepare_data()

    model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    class_weight="balanced",
    random_state=42
)

    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]

    pr_auc = average_precision_score(y_test, y_proba)

    print("PR-AUC :", pr_auc)
    recall_topk = recall_at_topk(y_test, y_proba, TOPK)

    print("Recall@TopK :", recall_topk)

    joblib.dump(model, MODEL_PATH)

    print("Model saved")


if __name__ == "__main__":
    train_model()
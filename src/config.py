DATA_PATH = "data/raw/credit_card_default.csv"

ID_COL = "id"
TARGET_COL = "default_payment_next_month"
LEAK_COL = "predicted_default_payment_next_month"

RANDOM_STATE = 42

TEST_SIZE = 0.2
VALID_SIZE = 0.2

TOPK = 0.10
MODEL_PATH = "models/model.joblib"
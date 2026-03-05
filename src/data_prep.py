import pandas as pd
from sklearn.model_selection import train_test_split
from config import DATA_PATH, TARGET_COL, LEAK_COL, ID_COL, RANDOM_STATE


def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


def prepare_data():

    df = load_data()

    X = df.drop(columns=[TARGET_COL, LEAK_COL])
    y = df[TARGET_COL]

    X = X.drop(columns=[ID_COL])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    return X_train, X_test, y_train, y_test
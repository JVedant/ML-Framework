import os
import numpy as np
import pandas as pd
import joblib

TEST_DATA = os.environ.get("TEST_DATA")
MODEL = os.environ.get("MODEL")

def predict():
    df = pd.read_csv(TEST_DATA)
    test_idx = df["id"].values
    prediction = None

    for fold in range(5):
        df = pd.read_csv(TEST_DATA)
        encoders = joblib.load(os.path.join('models', f'{MODEL}_{fold}_label_encoder.pkl'))
        cols = joblib.load(os.path.join('models', f'{MODEL}_{fold}_columns.pkl'))
        for c in encoders:
            lbl = encoders[c]
            df.loc[:, c] = df.loc[:, c].astype(str).fillna("NONE")
            df.loc[:, c] = lbl.transform(df[c].values.tolist())

        clf = joblib.load(os.path.join('models', f'{MODEL}_{fold}.pkl'))
        df = df[cols] 

        preds = clf.predict_proba(df)[:, 1]

        if fold == 0:
            prediction = preds
        else:
            prediction += preds

    prediction /= 5

    sub = pd.DataFrame(np.column_stack((test_idx, prediction)), columns=["id", "targets"])

    return sub

if __name__ == "__main__":
    submission = predict()
    submission.to_csv(os.path.join('input', 'submission.csv'), index=False)
import os
import pandas as pd
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import metrics
import joblib

from . import dispatcher

TRAINING_DATA = os.environ.get("TRAINING_DATA")
FOLD = int(os.environ.get("FOLD"))
MODEL = os.environ.get("MODEL")
TEST_DATA = os.environ.get("TEST_DATA")

'''FOLD_MAPPING ={
    0: [1, 2, 3, 4, 5, 6, 7, 8, 9],
    1: [0, 2, 3, 4, 5, 6, 7, 8, 9],
    2: [1, 0, 3, 4, 5, 6, 7, 8, 9],
    3: [1, 2, 0, 4, 5, 6, 7, 8, 9],
    4: [1, 2, 3, 0, 5, 6, 7, 8, 9],
    5: [1, 2, 3, 4, 0, 6, 7, 8, 9],
    6: [1, 2, 3, 4, 5, 0, 7, 8, 9],
    7: [1, 2, 3, 4, 5, 6, 0, 8, 9],
    8: [1, 2, 3, 4, 5, 6, 7, 0, 9],
    9: [1, 2, 3, 4, 5, 6, 7, 8, 0],
}
'''
if __name__ == "__main__":
    df = pd.read_csv(TRAINING_DATA)
    test_df = pd.read_csv(TEST_DATA)

    train_df = df[df.kfold != FOLD].reset_index(drop=True)
    valid_df = df[df.kfold == FOLD].reset_index(drop=True)

    y_train = train_df.target.values
    y_valid = valid_df.target.values

    train_df = train_df.drop(['id', 'target', 'kfold'], axis=1)
    valid_df = valid_df.drop(['id', 'target', 'kfold'], axis=1)
    test_df = test_df.drop(['id'], axis=1)

    valid_df = valid_df[train_df.columns]

    label_encoders = {}

    for c in train_df.columns:
        lbl = preprocessing.LabelEncoder()
        train_df.loc[:, c] = train_df.loc[:, c].astype(str).fillna("NONE")
        valid_df.loc[:, c] = valid_df.loc[:, c].astype(str).fillna("NONE")
        lbl.fit(
            train_df[c].values.tolist() + 
            valid_df[c].values.tolist() + 
            test_df[c].values.tolist()
        )
        train_df.loc[:, c] = lbl.transform(train_df[c].values.tolist())
        valid_df.loc[:, c] = lbl.transform(valid_df[c].values.tolist())
        label_encoders[c] = lbl
    # data ready to train

    clf = dispatcher.MODELS[MODEL]
    clf.fit(train_df, y_train)
    preds = clf.predict_proba(valid_df)[:, 1]
    print(f'roc_auc_score : {metrics.roc_auc_score(y_valid, preds)}')

    joblib.dump(label_encoders, os.path.join('models', f'{MODEL}_{FOLD}_label_encoder.pkl'))
    joblib.dump(clf, os.path.join('models', f'{MODEL}_{FOLD}.pkl'))
    joblib.dump(train_df.columns, os.path.join('models', f'{MODEL}_{FOLD}_columns.pkl'))
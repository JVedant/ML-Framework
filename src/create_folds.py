import pandas as pd
from sklearn.model_selection import StratifiedKFold

if __name__ == "__main__":
    df = pd.read_csv('input/train.csv')
    df['kfold'] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    
    kfold = StratifiedKFold(n_splits=5, shuffle=False, random_state=42)

    for fold_ , (t_, v_) in enumerate(kfold.split(X=df, y=df.target.values)):
        print(len(t_), len(v_))
        df.loc[v_, 'kfold'] = fold_

    df.to_csv('input/train_folds.csv', index=False)
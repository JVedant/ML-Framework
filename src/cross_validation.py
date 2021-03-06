import pandas as pd
from sklearn import model_selection

"""
- binary classification
- multi class classification
- multi label classification
- single column regression
- multi column regression
- holdout
"""

class CrossValidation:
    def __init__(
            self,
            df, 
            target_cols,
            shuffle,
            problem_type="binary_classification",
            num_folds=5,  
            random_state=42,
            multilabel_delimiter=","
            ):
        self.dataframe = df
        self.target_cols = target_cols
        self.problem_type = problem_type
        self.num_targets = len(target_cols)
        self.num_folds = num_folds
        self.random_state = random_state
        self.shuffle = shuffle
        self.multilabel_delimiter = multilabel_delimiter

        if self.shuffle == True:
            self.dataframe = self.dataframe.sample(frac=1).reset_index(drop=True)
            
        self.dataframe['kfold'] = -1

    def split(self):
        if self.problem_type in ("binary_classification", "multiclass_classification"):
            if self.num_targets != 1:
                raise Exception(f"Invalid number of targets for {self.problem_type}")
            target = self.target_cols[0]
            unique_values = self.dataframe[target].nunique()
            if unique_values == 1:
                raise Exception("Only 1 Unique value found")
            elif unique_values > 1:
                target = self.target_cols[0]
                kf = model_selection.StratifiedKFold(n_splits=self.num_folds, 
                                                    shuffle=self.shuffle, 
                                                    random_state=self.random_state
                                                )
                for fold_ , (train_idx, val_idx) in enumerate(kf.split(X=self.dataframe, y=self.dataframe[target].values)):
                    self.dataframe.loc[val_idx, 'kfold'] = fold_

        elif self.problem_type in ("single_col_regression", "multi_col_regression"):
            if self.num_targets != 1 and self.problem_type == 'single_col_regression':
                raise Exception(f"Invalid number of targets for {self.problem_type}")
            if self.num_targets < 2 and self.problem_type == 'multi_col_regression':
                raise Exception(f"Invalid number of targets for {self.problem_type}")
            target = self.target_cols[0]
            kf = model_selection.KFold(n_splits=self.num_folds, 
                                                    shuffle=self.shuffle, 
                                                    random_state=self.random_state
                                                )
            for fold_ , (train_idx, val_idx) in enumerate(kf.split(X=self.dataframe)):
                self.dataframe.loc[val_idx, 'kfold'] = fold_

        elif self.problem_type.startswith("holdout_"):
            holdout_percentage = int(self.problem_type.split('_')[1])
            num_holdout_samples = int(len(self.dataframe) * holdout_percentage / 100)
            self.dataframe.loc[:num_holdout_samples, 'kfold'] = 0
            self.dataframe.loc[num_holdout_samples:, 'kfold'] = 1 

        elif self.problem_type == "multilabel_classification":
            if self.num_targets != 1:
                raise Exception(f"Invalid number of targets for {self.problem_type}")
            targets = self.dataframe[self.target_cols[0]].apply(lambda x: len(str(x).split(self.multilabel_delimiter)))
            kf = model_selection.StratifiedKFold(n_splits=self.num_folds, 
                                                    shuffle=self.shuffle, 
                                                    random_state=self.random_state
                                                )
            for fold, (train_idx, val_idx) in enumerate(kf.split(X=self.dataframe, y=targets)):
                self.dataframe.loc[val_idx, 'kfold'] = fold

        else:
            raise Exception("Problem type not understood")

        return self.dataframe

if __name__ == "__main__":
    df = pd.read_csv('input/train_multilabel.csv')
    cv = CrossValidation(
        df=df,
        target_cols=['attribute_ids'],
        problem_type="multilabel_classification", 
        shuffle=True,
        multilabel_delimiter=" "
    )
    df_split = cv.split()
    print(df_split.head())
    print(df_split.kfold.value_counts())
export TRAINING_DATA=input/train_folds.csv
export TEST_DATA=input/test.csv
export MODEL=$1

# to train the network 
FOLD=0 python3 -m src.train
FOLD=1 python3 -m src.train
FOLD=2 python3 -m src.train
FOLD=3 python3 -m src.train
FOLD=4 python3 -m src.train

# To predict the results
#python3 -m src.predict
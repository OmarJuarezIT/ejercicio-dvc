import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
import argparse
import joblib

def train(config_path):
    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)
  
    train_dataset = pd.read_csv(config['data']['trainset_path'])
    y_train = train_dataset.loc[:, 'target'].values.astype('int32')
    X_train = train_dataset.drop('target', axis=1).values.astype('float32')

    logreg = LogisticRegression(**config['train']['clf_params'], random_state = config['base']['random_state'])
    logreg.fit(X_train, y_train)

    joblib.dump(logreg, config['train']['model_path'])

    print('Terminó de entrenarse el modelo de regresión logística')


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    train(config_path=args.config)
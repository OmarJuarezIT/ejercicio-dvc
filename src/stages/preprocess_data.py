import yaml
from sklearn.model_selection import train_test_split
import pandas as pd
import argparse

def data_load(config_path):

    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)

    raw_data_path = config['data']['dataset_csv']
    dataset = pd.read_csv(raw_data_path)
    dataset.columns = [colname.strip(' (cm)').replace(' ', '_') for colname in dataset.columns.tolist()]
    dataset['sepal_length_to_sepal_width'] = dataset['sepal_length'] / dataset['sepal_width']
    dataset['petal_length_to_petal_width'] = dataset['petal_length'] / dataset['petal_width']
    dataset = dataset[[
        'sepal_length', 'sepal_width', 'petal_length', 'petal_width',
    #     'sepal_length_in_square', 'sepal_width_in_square', 'petal_length_in_square', 'petal_width_in_square',
        'sepal_length_to_sepal_width', 'petal_length_to_petal_width',
        'target'
    ]]
    train_dataset, test_dataset = train_test_split(dataset, test_size=config['data_split']['test_size'], random_state=config['base']['random_state'])

    train_dataset.to_csv(config['data']['trainset_path'], index = False)
    test_dataset.to_csv(config['data']['testset_path'], index = False)

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    data_load(config_path=args.config)
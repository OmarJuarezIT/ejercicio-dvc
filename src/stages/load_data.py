import yaml
import argparse
from sklearn.datasets import load_iris



def load_data(config_path):

    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)
        
    data = load_iris(as_frame=True)
    dataset = data.frame
    dataset.to_csv(config['data']['dataset_csv'], index = False)

    

if __name__ == '__main__':
    
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    load_data(config_path=args.config)
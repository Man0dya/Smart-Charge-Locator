import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="EV pipeline CLI")
    parser.add_argument('--step', choices=['prep','train','place','export'], required=True)
    args = parser.parse_args()
    if args.step == 'prep':
        print('Run the notebook 01_data_preprocessing.ipynb for detailed steps.')
    elif args.step == 'train':
        print('Train models via the modeling sections in the notebook.')
    elif args.step == 'place':
        print('Run clustering/ILP sections in the notebook.')
    elif args.step == 'export':
        print('Export artifacts/stations.csv and other files for the app.')


if __name__ == '__main__':
    main()

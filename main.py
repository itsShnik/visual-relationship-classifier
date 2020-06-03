from dataloader import DatasetLoader
from train import train_engine
# write a function to parse the arguments and conifg

def main():

    # first loading the dataset
    dataset = DatasetLoader(config)

    # change config attribute to val
    # TODO: write attr changing code here
    dataset_val = DatasetLoader(config)

    # train the model
    train_engine(config, dataset, dataset_val)


if __name__=='__main__':
    main()

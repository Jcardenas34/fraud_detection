import argparse
from fraud_detection.train_models import train_vae
import pandas as pd

def main(args):
    # Basing batch size on predicted fraud rate

    df = pd.read_hdf(args.data_path, key='fraud_dataset')
    dataset_len = len(df["Channel"])
    batch_size = int(.10*dataset_len)

    print("Choosing batch size to be about 10% of the data, batch_size={}".format(batch_size))

    train_vae(args.data_path, args.model_out, batch_size=batch_size, epochs=30, learning_rate=0.001, num_workers=4, log_interval=100)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", dest="data_path", type=str, default="./dataset/credit_card_fraud.h5")
    parser.add_argument("-m", dest="model_out", type=str, default="./models/vae_model_non_scaled_data.pth")
    parser.add_argument("-b", dest="batch_size", type=int, default=64)
    args = parser.parse_args()

    main(args)



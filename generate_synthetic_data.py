import argparse 
import torch
import pandas as pd
from fraud_detection.helpers import load_scaling_object
from fraud_detection.helpers import load_model, load_trained_vae

def main(args):
    '''
    Will generate synthetic data based on the VAE that was trained 
    '''
    model, _ = load_trained_vae(args.model_path)
    decoder = model.decode



    try:
        z = torch.randn(args.n_samples, 10)  # Sample latent vector

    except TypeError as err:
        print(f"Did not enter a positive integer: {err}")
        return
    
    # Setting to evaluation mode
    # decoder.eval()
    with torch.no_grad():
        synthetic_x = decoder(z)

    # Converting the data to a pandas dataframe to allow
    # For easy conversion to hdf5
    df = pd.DataFrame(synthetic_x.detach().numpy())
    # scalar = load_scaling_object("./scalars/standard_scaler_AE.pkl")
    scalar = load_scaling_object("./scalars/quantle_scaler_VAE.pkl")

    # Inverse transform the synthetic data to get it back to original scale
    df = pd.DataFrame(scalar.inverse_transform(df), columns=df.columns)


    df.to_hdf(args.outfile, key='fraud_dataset')

    for row in df.iterrows():
        print(row)


    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", dest="model_path", type=str, default="./models/vae_model_quantile_scaled_data.pth")
    parser.add_argument("-n_samples", dest="n_samples", type=int, default=100)
    parser.add_argument("-o", dest="outfile", default="dataset/synthetic_dataset_quantile.h5")
    args = parser.parse_args()

    main(args)
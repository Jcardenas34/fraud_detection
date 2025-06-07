import pickle
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler 
from fraud_detection.models import Autoencoder, VariationalAutoencoder

def persistify_scaling_object(data_path:str, scaling_object_name: str) -> None:
    '''
    Will save an instance of the scaling function used to preprocess the data
    '''

    df = pd.read_hdf(data_path, key='fraud_dataset')

    # Creating the scaling object for later transformation of data
    persistified_scaling_object = StandardScaler().fit(df)

    
    filename = "./scalars/"+scaling_object_name

    with open(filename, 'wb') as file:
        pickle.dump(persistified_scaling_object, file)


def load_scaling_object(scalar_path: str):
    '''
    Load up the existing scalar used to train the model
    allows for live preprocessing of incoming data
    '''

    with open(scalar_path, 'rb') as file:
        loaded_scaler = pickle.load(file)

    return loaded_scaler


def load_model(model_path:str, evaluation_arch:str="cpu" ):
    '''
    Entry point for loading the model into memory for evaluation
    '''

    # Load the saved state dictionary
    

    try: 
        # Adjust 'cpu' parameter here if you want to evaluate using a cpu
        state_dict = torch.load(model_path, map_location=torch.device(evaluation_arch))

    except ValueError:
        if evaluation_arch not in ["cpu","gpu"]:
            print("Incorrect architecture selected, use 'cpu' or 'gpu' ")
        return


    if "VAE" in model_path:
        model = VariationalAutoencoder()
    else:
        # Create a model instance
        model = Autoencoder()

    # Load the state dictionary into the model
    model.load_state_dict(state_dict)

    # Need to set into evaluation mode
    model.eval()

    return model

def load_trained_vae(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = VariationalAutoencoder().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model, device

def load_data_subset(data_path:str, n_samples:int = -1):
    '''
    Loads in entire dataset, or loads in just a sub sample of the data
    '''

    df_hdf5 = pd.read_hdf(data_path, key='fraud_dataset')
    if n_samples==-1:
        # loading all samples
        samples = df_hdf5.iloc[:].values

    else:
        # loading in the hdf, to extract a few rows of data
        samples = df_hdf5.iloc[:n_samples].values

    return samples 
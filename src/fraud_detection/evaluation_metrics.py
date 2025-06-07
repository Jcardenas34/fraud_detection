import torch
import pandas as pd
from typing import Tuple
from sklearn.preprocessing import StandardScaler
from fraud_detection.models import Autoencoder, VariationalAutoencoder



# def calculate_zscore( mse:float | list[float], mean:float, std:float ) -> float | list[float] | None : 
def calculate_zscore( mse:float , mean:float, std:float ) -> float : 
    '''
    Will take the 'mse' of a given data point and return
    '''

    zscore = None
    zscore = (mse - mean) / std
    # try:
    #     if isinstance( mse, float ):
    #         zscore = (mse - mean) / std
        
    #     elif isinstance(mse, list):
    #         zscore = []
    #         for err in mse:
    #             zscore.append((err - mean) / std)

    # except ValueError as err:
    #     print(f"Incorrect type passed: {err}")   
        # return None 
    
    return zscore

def extract_mean_std_from_training_data( data_path:str, model_path: str ) -> Tuple[float, float]:
    
    df = pd.read_hdf(data_path, key='fraud_dataset')

    # Providing the same treatment to the data as when it was trained 
    X = StandardScaler().fit_transform(df)
    X_tensor = torch.tensor(X, dtype=torch.float32)

    # Loading the model
    state_dict = torch.load(model_path,  map_location=torch.device('cpu'))


    if "VAE" in model_path:
        loaded_model = VariationalAutoencoder()
    else:
        loaded_model =  Autoencoder()

    loaded_model.load_state_dict(state_dict)
    # Switch the model to evaluation mode
    loaded_model.eval()

    # Calculate reconstruction errors for the entire dataset
    with torch.no_grad():
        reconstructed = loaded_model(X_tensor)
        reconstruction_errors = torch.nn.functional.mse_loss(reconstructed, X_tensor, reduction='none')  # none does not apply an operation to the output
        sample_errors = reconstruction_errors.mean(dim=1)  # Take the mean error of each sample (add the differences of each feature, and divide by the number of features)

    # Convert to numpy for entry into the pandas df
    sample_errors_np = sample_errors.numpy()

    mean = float(sample_errors_np.mean())
    std = float(sample_errors_np.std())
    # print(mean, std)
    # print(type(mean), type(std))

    return mean, std



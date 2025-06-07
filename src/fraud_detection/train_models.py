import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from fraud_detection.models import VariationalAutoencoder
from fraud_detection.loss_functions import loss_function
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from fraud_detection.helpers import persistify_scaling_object

def train_vae(data_path, model_path, batch_size, epochs, learning_rate, num_workers, log_interval):
    """
    Train a Variational Autoencoder (VAE) on the given dataset.
    """

    # Load and preprocess data
    df = pd.read_hdf(data_path, key='fraud_dataset')
    # X = StandardScaler().fit_transform(df)
    X = QuantileTransformer(output_distribution='normal').fit_transform(df) # Better for VAE training and synthetic data generation apparently
    # X = df.values  # Case where data will not be scaled, since model is used to gernerate synthetic data?

    # Persistify the scaling object for later use
    persistify_scaling_object(data_path, "quantle_scaler_VAE.pkl")

    X_tensor = torch.tensor(X, dtype=torch.float32)

    dataset = TensorDataset(X_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # Initialize the VAE model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VariationalAutoencoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    model.train()
    for epoch in range(epochs):
        train_loss = 0
        for batch_idx, (data,) in enumerate(dataloader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_data, mu, logvar = model(data)
            loss = loss_function(recon_data, data, mu, logvar)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # if batch_idx % log_interval == 0:
                # print(f"Epoch {epoch+1}  [{batch_idx * len(data)}/{len(dataloader.dataset)}] Loss: {loss.item():.4f}")

        print(f"Epoch {epoch+1} Average Loss: {train_loss / len(dataloader):.4f}")

    # Save the trained model
    torch.save(model.state_dict(), model_path)

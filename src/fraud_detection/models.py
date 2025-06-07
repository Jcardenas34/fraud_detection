import torch
import torch.nn as nn

# Defining an autoencoder class to make calling the model simpler main function
class Autoencoder(nn.Module):
    '''
    Description:
    ---------------
    A simple auto encoder used to detect instances of fraud in a dataset
    '''
    
    def __init__(self, n_features = 11):
        super(Autoencoder, self).__init__()
        
        self.n_features = n_features

        # Gradually decrease the latent space
        self.layer2_nodes = 100
        self.layer3_nodes = 50
        self.layer4_nodes = 20
        self.latent_nodes = 10
        
        
        # Defining the layers of the encoder
        self.in_layer = torch.nn.Linear(self.n_features,   self.layer2_nodes)
        self.layer1   = torch.nn.Linear(self.layer2_nodes, self.layer2_nodes)
        self.layer2   = torch.nn.Linear(self.layer2_nodes, self.layer2_nodes)
        self.layer3   = torch.nn.Linear(self.layer2_nodes, self.layer3_nodes)
        self.layer4   = torch.nn.Linear(self.layer3_nodes, self.layer4_nodes)
        self.layer5   = torch.nn.Linear(self.layer4_nodes, self.latent_nodes)
    
        # Defining the layers of the decoder
        self.layer6 = torch.nn.Linear(self.latent_nodes, self.layer4_nodes)
        self.layer7 = torch.nn.Linear(self.layer4_nodes, self.layer3_nodes)
        self.layer8 = torch.nn.Linear(self.layer3_nodes, self.layer2_nodes)
        self.layer9 = torch.nn.Linear(self.layer2_nodes, self.layer2_nodes)
        self.layer10 = torch.nn.Linear(self.layer2_nodes, self.layer2_nodes)
        self.out_layer = torch.nn.Linear(self.layer2_nodes, self.n_features)

    

    def encoder(self, x):
        # Constructing the encoder stack
        h = torch.relu(self.in_layer(x))
        h = torch.relu(self.layer1(h))
        h = torch.relu(self.layer2(h))
        h = torch.relu(self.layer3(h))
        h = torch.relu(self.layer4(h))
        h = torch.relu(self.layer5(h))
        return h       

    def decoder(self, x):
        # Constructing the decoder stack
        h = torch.relu(self.layer6(x))
        h = torch.relu(self.layer7(h))
        h = torch.relu(self.layer8(h))
        h = torch.relu(self.layer9(h))
        h = torch.relu(self.layer10(h))
        h = torch.relu(self.out_layer(h))
        return h  

    def forward(self, x):
        # This function will always be called in pytorch
        latent_representation = self.encoder(x)
        return self.decoder(latent_representation)
        

class VariationalAutoencoder(nn.Module):
    """
    Description:
    ------------
    Create a very simple variational autoencoder for detecting fraud transactions in credit card data. 
    """
    
    def __init__(self, input_dim=11, hidden_dim=100, latent_dim=10):
        super(VariationalAutoencoder, self).__init__()

        self.latent_dim = latent_dim

        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)     # Mean
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim) # Log Variance

        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = torch.relu(self.fc3(z))
        h = torch.relu(self.fc4(h))
        return self.fc5(h)  # No activation for final output (regression task)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
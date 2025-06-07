import torch
from fraud_detection.evaluation_metrics import calculate_zscore, extract_mean_std_from_training_data
from fraud_detection.helpers import load_scaling_object, load_model


def detect_fraud(model_path, features, threshold=0.05):
    '''
    Will evaluate instances of fraud based on a threshold z-score
    '''
    data_path = './dataset/credit_card_fraud.h5'

    model = load_model(model_path=model_path)

    mean, std = extract_mean_std_from_training_data(data_path=data_path, model_path=model_path)

    # Preprocess the data before evaluating
    scaling_obj = load_scaling_object("scalars/standard_scaler_AE.pkl")
    # print(features)
    features = features.reshape(1, -1)
    features = scaling_obj.transform(features)[0]
    # print(features[0])

    with torch.no_grad():
        feature_vector = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        reconstructed = model(feature_vector)
        # print(f"feature shape: {feature_vector.shape}, Reco shape: {reconstructed.shape}")
        reconstruction_errors = torch.nn.functional.mse_loss(reconstructed[0], feature_vector[0], reduction='none')  # none does not apply an operation to the output
        # print("Reco Err: ", reconstruction_errors)
        sample_errors = float(reconstruction_errors.mean())
        # print("Sample Err: ", sample_errors)

        sample_zscore = calculate_zscore(sample_errors, mean, std)
        is_fraud =  sample_zscore > threshold
        # return 
        return {
            "fraud": is_fraud,
            "zscore": sample_zscore,
            # "reconstruction_error": reconstruction_errors
        }

from fraud_detection.evaluation_metrics import extract_mean_std_from_training_data


def test_extract_mean_std():
    '''
    Tests weather or not this lone module works
    '''
    data_path = './dataset/credit_card_fraud.h5'
    model_path = "./models/fraud_autoencoder_model.h5"
    mean, std = extract_mean_std_from_training_data(data_path, model_path)

    assert isinstance(mean, float), isinstance(std, float)

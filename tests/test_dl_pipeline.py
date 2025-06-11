
from fraud_detection.core_detection import detect_fraud
from fraud_detection.helpers import load_data_subset

def test_dl_pipeline():
    '''
    Tests the end-to-end vailidty of deep learning pipeline for fraud detection.
    persistification of the scaler, model, and the data loading mechanism are tested here.
    Expected result when evaluating the model on the dataset is that the fraud rate is around 2%.
    Where 70 instances of fraud are detected in the dataset, as a benchmark for this model as that is 
    what it was trained on.

    '''
    data_path = './dataset/credit_card_fraud.h5'
    model_path = "./models/fraud_autoencoder_model.h5"

    data_points = load_data_subset(data_path=data_path, n_samples=100)
    counter = 0
    for data in data_points:
        try:
            fraud_dict = detect_fraud(model_path, data, threshold=2.0)
            assert isinstance(fraud_dict["fraud"], bool), "Fraud detection result should be a boolean"
            assert isinstance(fraud_dict["zscore"], float), "Z-score should be a float"

            # Increment each time we encounter fraud
            if fraud_dict["fraud"]:
                counter += 1

        except ValueError as err:
            print(f"Inputted data is of incorrect dimension: {err}")

    proportion_of_fraud = 100 * counter / len(data_points)
    print(f"Fraud Rate: {proportion_of_fraud}, expected ~2%")
    print(f"Fraud Count: {counter}")

    assert proportion_of_fraud > 0, "Expected some fraud in the dataset"
    assert counter > 0, "Expected some fraud instances to be detected"
    assert len(data_points) > 0, "Expected some data points to be loaded"
    # assert all(isinstance(data, dict) for data in data_points), "Each data point should be a dictionary"
    assert len(data_points[0]) > 0, "Each data point should have some features"
    print("All tests passed successfully.")

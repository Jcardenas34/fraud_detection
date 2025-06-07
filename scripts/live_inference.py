import argparse
import warnings

from fraud_detection.core_detection import detect_fraud
from fraud_detection.helpers import load_data_subset

def main(args):
    '''
    Function that will evaluate a model on a dataset, and print out the fraud rate
    '''

    # Suppress any warnings about the naming of columns not being present at scaling time
    if args.debug is False:
        warnings.simplefilter("ignore", category=UserWarning)

    # Calling the hdf5 dataset something that makes sense
    # data_path = './dataset/credit_card_fraud.h5'
    data_path = "./dataset/synthetic_dataset.h5"

    model_path = './models/fraud_autoencoder_model.h5'

    # Trying to make this work with the variational autoencoder.
    # model_path = './models/fraud_VAE_model.pth' #Work In Progress 
    

    data_points = load_data_subset(data_path=data_path, n_samples=-1)


    # Taking a note of how many instances have a z score larger than 2, 
    counter = 0
    for data in data_points:
        try:
            fraud_dict = detect_fraud(model_path, data,  threshold=2.0)
            print(fraud_dict["fraud"], fraud_dict["zscore"])

            # Increment each time we encounter fraud
            if fraud_dict["fraud"] :
                counter+=1

        except ValueError as err:
            print(f"Inputted data is of incorrect dimension: {err}" )



    proportion_of_fraud = 100* counter / len(data_points)
    print(f"Fraud Rate: {proportion_of_fraud}, expected ~5%")
    print(f"Fraud Count: {counter}")




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-debug", dest = "debug", action="store_true", default=False)
    args = parser.parse_args()

    main(args)
from fraud_detection.helpers import persistify_scaling_object



def test_persistify_scaling_object():
    '''
    Checking to see if this function does return a pkl file
    '''

    scaling_obj_name = 'standard_scaler_AE.pkl'
    data_path = './dataset/credit_card_fraud.h5'

    try:
        persistify_scaling_object(data_path=data_path, scaling_object_name=scaling_obj_name )

    except Exception as err:
        print(f"Logging error here: {err}")


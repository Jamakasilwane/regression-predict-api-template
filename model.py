"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Plase follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.  

"""

# Helper Dependencies
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.


    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    Numpy 2 Dimensional array
        The preprocessed data, ready to be used our model for prediction.

    """
    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])

    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------

    # ----------- Replace this code with your own preprocessing steps --------

    #  Time Columns Types Into Datetime
    for col in list(feature_vector_df):
        if 'Time' in col:
            feature_vector_df[col] = pd.to_datetime(feature_vector_df[col])

    # Making a copy for the dataframe
    dataset_test = feature_vector_df.copy()
    dataset_test = dataset_test.set_index('Order No')

    # Encoding Categoric Data
    # Hours were extracted and added to the dataframe as new columns
    dataset_test['Placement - Hour']=dataset_test['Placement - Time'].dt.hour

    dataset_test['Confirmation - Hour']=dataset_test['Confirmation - Time'].dt.hour

    dataset_test['Arrival at Pickup - Hour']=dataset_test['Arrival at Pickup - Time'].dt.hour

    dataset_test['Pickup - Hour']=dataset_test['Pickup - Time'].dt.hour

    # Minutes were extracted and added to the dataframe as new columns
    dataset_test['Placement - Minutes']=dataset_test['Placement - Time'].dt.minute

    dataset_test['Confirmation - Minutes']=dataset_test['Confirmation - Time'].dt.minute

    dataset_test['Arrival at Pickup - Minutes']=dataset_test['Arrival at Pickup - Time'].dt.minute

    dataset_test['Pickup - Minutes']=dataset_test['Pickup - Time'].dt.minute

    # Seconds were extracted and added to the dataframe as new columns
    dataset_test['Placement - Seconds']=dataset_test['Placement - Time'].dt.second

    dataset_test['Confirmation - Seconds']=dataset_test['Confirmation - Time'].dt.second

    dataset_test['Arrival at Pickup - Seconds']=dataset_test['Arrival at Pickup - Time'].dt.second

    dataset_test['Pickup - Seconds']=dataset_test['Pickup - Time'].dt.second

    # Platform Type
    dataset_test['Platform Type'] = dataset_test['Platform Type'].astype('category')
    dataset_test = pd.concat([dataset_test.drop(columns=['Platform Type']),
                            pd.get_dummies(dataset_test['Platform Type'])], 
                            axis=1)

    # Renaming the 'platform type' columns
    dataset_test.rename(columns={1: "Platform Type 1", 2: "Platform Type 2", 3:
                                "Platform Type 3", 4: "Platform Type 4"},
                        inplace=True)

    # Dummy coding of the 'Personal or Business' column
    dataset_test['Personal or Business'] = dataset_test['Personal or Business'].astype('category')
    dataset_test = pd.concat([dataset_test.drop(columns=['Personal or Business']), 
                            pd.get_dummies(dataset_test['Personal or Business'])],
                            axis=1)

    # Renaming the 'Personal or Business' columns
    dataset_test.rename(columns={0: "Business", 1: "Personal"}, inplace=True)

    # Selecting columns for our test model
    X2 = dataset_test.loc[:,['Order No','Distance (KM)', 'Temperature',
                            'Precipitation in millimeters', 'Platform Type 1', 
                            'Platform Type 2', 'Platform Type 3', 
                            'Platform Type 4', 'Business', 'Personal',
                            'Placement - Hour', 'Confirmation - Hour', 
                            'Arrival at Pickup - Hour', 'Pickup - Hour', 
                            'Placement - Minutes', 'Confirmation - Minutes',
                            'Arrival at Pickup - Minutes', 'Pickup - Minutes',
                            'Placement - Seconds', 'Confirmation - Seconds',
                            'Arrival at Pickup - Seconds', 'Pickup - Seconds']]

    #changing the Index for X_test Dataframes to be Order No and convert the Dataframes to Numpy arrays
    X_test = X2.values

    # Replacing the Nan values with the mean
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer.fit(X_test[:,1:3])
    X_test[:,1:3] = imputer.transform(X_test[:,1:3])

    # Feature Scaling of the training and test data set
    sc = StandardScaler()
    X_test[:,:] = sc.transform(X_test[:,:])

    predict_vector = X_test
    
    # ------------------------------------------------------------------------

    return predict_vector

def load_model(path_to_model:str):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.

    """
    return pickle.load(open(path_to_model, 'rb'))

def make_prediction(data, model):
    """Prepare request data for model prediciton.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    # Format as list for output standerdisation.
    return prediction[0].tolist()

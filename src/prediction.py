import numpy as np

def make_prediction(classifier, scaler, input_data):
    input_data_as_numpy_array = np.asanarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    std_data = scaler.transform(input_data_reshaped)

    prediction = classifier.predict(std_data)
    if (prediction[0] == 0):
        print("This person is not diabetic")
    else:
        print('The person is diabetic')

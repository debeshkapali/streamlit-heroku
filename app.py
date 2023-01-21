import numpy as np
import pickle
import streamlit as st

# loading the saved model using pickle

loaded_model = pickle.load(open('E:/Data Science projects/Model deployment using streamlit library/diabetes_model.sav', 'rb'))

# creating the function for getting the input from the user

def diabetes_prediction(input_data):

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'

def main():
    
    # giving the title for the webpage
    st.title('Diabetes prediction system')

    # getting the input data from the user
    Pregnancies = st.text_input('Number of pregnancies')               
    Glucose = st.text_input('Glucose level')                     
    BloodPressure = st.text_input('Blood Pressure level')              
    SkinThickness = st.text_input('Skin Thickness')              
    Insulin = st.text_input('Insulin level')                    
    BMI = st.text_input('BMI value')                        
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Fucntion value')    
    Age = st.text_input('Age of the patient')

    # Prediction
    diagnosis = ''

    # creating a button for the prediction
    if st.button('Diabetes Test Result'): # name of the button
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])

    st.success(diagnosis)

if __name__ == '__main__': # should run only from command prompt
    main()                     



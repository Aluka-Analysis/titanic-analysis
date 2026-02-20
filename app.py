import streamlit as st
import pandas as pd
import joblib

st.title("Titanic Survival Prediction App")

st.write("Predict whether a passenger would survive")

model = joblib.load(r"C:\Users\DOUBLE J\Documents\data-projects\Titanic-analysis\Outputs\Visualization\models\titanic_Logistic_regression_model.pkl")

scaler = joblib.load(r"C:\Users\DOUBLE J\Documents\data-projects\Titanic-analysis\Outputs\Visualization\models\titanic_scaler.pkl")

# User Inputs

Pclass = st.selectbox("Passenger Class", [1,2,3])

Sex = st.selectbox("Sex", ["Male", "Female"])

Age = st.slider("Age", 1, 80, 25)

Fare = st.slider("Fare", 0, 500, 50)

Embarked = st.selectbox("Embarked", ["Southampton", "Cherbourg", "Queenstown"])

# Converts Inputs

Sex = 0 if Sex == "Male" else 1

Embarked_dict = {
    "Southampton": 2,
    "Cherbourg": 0,
    "Queenstown": 1
}
Embarked = Embarked_dict[Embarked]

sibsp = st.slider("Siblings/Spouses Aboard", 0, 8, 0)

parch = st.slider("Parents/Children Aboard", 0, 6, 0)


# Create dataframe

data = pd.DataFrame({
    "Pclass":[Pclass],
    "Sex":[Sex],
    "Age":[Age],
    "SibSp":[sibsp],
    "Parch":[parch],
    "Fare":[Fare],
    "Embarked":[Embarked]    
})

# Prediction

if st.button("Predict"):

    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)
    

    if prediction == 1:
        st.success("Passenger survived")

    else:
        st.error("Passenger did not survive")
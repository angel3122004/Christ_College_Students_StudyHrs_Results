import streamlit as st
import joblib
model=joblib.load("logistic_regression_studyHrs_model.pkl")
st.title("Student Pass/Fail based on Study Hours")
hours=st.number_input("Enter Study Hours: ",min_value=0.0, max_value= 15.0, value=5.0)
attendance=st.number_input("Enter Attendance: ",min_value=0.0, max_value=100, value=75)

if st.button("Predict"):
  prediction=model.predict([[hours,attendance]])
  pass_probability = model.predict_proba([[hours]])[0][1] * 100
  if prediction[0]==1:
    st.success("Pass")
  else:
    st.error("Fail")
  st.metric(label="Probability of Passing", value=f"{pass_probability:.1f}%")

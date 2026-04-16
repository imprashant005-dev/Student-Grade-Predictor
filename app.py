import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data = {
    "Hours_Studied": [2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Past_Score": [50, 55, 60, 65, 70, 75, 80, 85, 90],
    "Final_Marks": [52, 58, 63, 68, 72, 78, 83, 88, 95],
}

df = pd.DataFrame(data)
X = df[["Hours_Studied", "Past_Score"]]
y = df["Final_Marks"]

model = LinearRegression()
model.fit(X, y)

st.title("Student Grade Predictor 🎓")
st.write(
    "This simple ML project predicts final marks based on study hours and past scores."
)

st.header("Enter Your Details:")
my_hours = st.slider("How many hours do you study daily?", 1, 15, 4)
my_past_score = st.slider("What is your past average score?", 0, 100, 60)

input_data = pd.DataFrame(
    [[my_hours, my_past_score]], columns=["Hours_Studied", "Past_Score"]
)

predicted_mark = model.predict(input_data)[0]

st.header("Prediction Result:")
final_result = min(predicted_mark, 100)
st.success(f"Expected Final Marks: {final_result:.2f} / 100")

st.subheader("Data Trend (Hours vs Final Marks)")
fig, ax = plt.subplots()
ax.scatter(df["Hours_Studied"], df["Final_Marks"], color="blue")
ax.set_xlabel("Hours Studied")
ax.set_ylabel("Final Marks")
st.pyplot(fig)

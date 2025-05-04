import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import messagebox

# Load dataset
data = pd.read_csv("data/student-mat.csv", sep=';')

# One-hot encode categorical variables
data = pd.get_dummies(data, columns=['school', 'sex', 'address', 'famsize', 'Pstatus',
                                     'Mjob', 'Fjob', 'reason', 'guardian', 'schoolsup',
                                     'famsup', 'paid', 'activities', 'nursery', 'higher',
                                     'internet', 'romantic'], drop_first=True)

# Select features and target
features = data.drop(columns=['G3'])
target = data['G3']

# Split data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
model = linear_model.LinearRegression()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
overall_accuracy = r2 + (1 / mse)

# Get coefficients to understand how each parameter affects the grade
coefficients = pd.DataFrame(model.coef_, features.columns, columns=["Coefficient"])
studytime_coeff = coefficients.loc['studytime', 'Coefficient']

# Convert grade to scale (1 to 5)
def grade_to_scale(grade):
    if grade >= 16:
        return 5
    elif grade >= 12:
        return 4
    elif grade >= 8:
        return 3
    elif grade >= 4:
        return 2
    else:
        return 1

# Function to plot graph with actual vs predicted grades
def plot_graph():
    predicted_test_grades = model.predict(X_test)
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, predicted_test_grades, color='blue', label="Actual vs Predicted", alpha=0.6)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label="Ideal Line")
    plt.title("Actual vs Predicted Grades")
    plt.xlabel("Actual Grades")
    plt.ylabel("Predicted Grades")
    plt.legend()
    plt.show()

# GUI App
def launch_gui():
    def on_predict():
        try:
            g1 = int(entry_g1.get())
            g2 = int(entry_g2.get())
            studytime = float(entry_studytime.get())

            if g1 not in range(1, 6) or g2 not in range(1, 6):
                raise ValueError("Grades should be between 1 and 5")

            scaled_g1 = (g1 / 5) * 20
            scaled_g2 = (g2 / 5) * 20

            new_student_data = {
                'G1': [scaled_g1],
                'G2': [scaled_g2],
                'studytime': [studytime],
            }

            new_student_features = pd.DataFrame(new_student_data)
            new_student_features_encoded = pd.get_dummies(new_student_features, drop_first=True)
            new_student_features_encoded = new_student_features_encoded.align(features, join='right', axis=1, fill_value=0)[0]
            predicted_grade = model.predict(new_student_features_encoded)[0]
            grade_scale = grade_to_scale(predicted_grade)
            effect_of_studytime = studytime_coeff * studytime

            result_label.config(text=f"Predicted Final Grade (1-5): {grade_scale}")
            studytime_effect_label.config(text=f"Effect of Study Time: {effect_of_studytime:.2f} points")
            accuracy_label.config(text=f"Model Accuracy (R² Score): {r2:.2f}")

        except ValueError as e:
            messagebox.showerror("Input Error", str(e))

    root = tk.Tk()
    root.title("Student Grade Predictor")
    root.geometry("400x380")

    tk.Label(root, text="Enter G1 (1-5 scale):").pack(pady=5)
    entry_g1 = tk.Entry(root)
    entry_g1.pack()

    tk.Label(root, text="Enter G2 (1-5 scale):").pack(pady=5)
    entry_g2 = tk.Entry(root)
    entry_g2.pack()

    tk.Label(root, text="Enter Study Time (hrs/week):").pack(pady=5)
    entry_studytime = tk.Entry(root)
    entry_studytime.pack()

    predict_button = tk.Button(root, text="Predict Grade", command=on_predict)
    predict_button.pack(pady=10)

    result_label = tk.Label(root, text="")
    result_label.pack(pady=5)

    studytime_effect_label = tk.Label(root, text="")
    studytime_effect_label.pack(pady=5)

    accuracy_label = tk.Label(root, text="")
    accuracy_label.pack(pady=5)

    graph_button = tk.Button(root, text="Show Regression Graph", command=plot_graph)
    graph_button.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    launch_gui()

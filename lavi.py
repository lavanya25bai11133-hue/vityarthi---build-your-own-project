import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from io import StringIO

csv_data = """
Age,Communication_Score,Social_Interaction_Score,Behavioral_Score,Cognitive_Score,Motor_Skills_Score,Symptom_Intensity,ASD_Severity,Autism_Diagnosis,Therapy_Progress
12,7,6,5,6,7,5,6,Yes,Improving
15,5,4,6,5,6,7,8,Yes,Not Improving
18,8,7,7,8,8,4,5,No,Stable
"""
df = pd.read_csv(StringIO(csv_data))

y = df["Therapy_Progress"]
x = df.drop("Therapy_Progress", axis=1)

le_y = LabelEncoder()
y_encoded = le_y.fit_transform(y)

le_autism = LabelEncoder()
df["Autism_Diagnosis"] = le_autism.fit_transform(df["Autism_Diagnosis"])
x["Autism_Diagnosis"] = df["Autism_Diagnosis"]

model = RandomForestRegressor(max_depth=2, random_state=100)
model.fit(x, y_encoded)

root = tk.Tk()
root.title("Autism Therapy Progress Predictor")
root.geometry("850x800")
root.configure(bg="#e8f0fe")

title = tk.Label(root, text="Therapy Progress Prediction System",
                 font=("Arial", 22, "bold"), bg="#e8f0fe", fg="#0b2545")
title.pack(pady=10)

description_frame = tk.LabelFrame(
    root, text="Feature Descriptions", font=("Arial", 14),
    bg="#ffffff", fg="#0b2545", padx=10, pady=10)
description_frame.pack(fill="both", padx=20, pady=10)

descriptions = {
    "Age": "Age of the patient (in years).",
    "Communication_Score": "Ability to communicate (1–10).",
    "Social_Interaction_Score": "Quality of social interaction (1–10).",
    "Behavioral_Score": "Behavior regulation level (1–10).",
    "Cognitive_Score": "Cognitive thinking ability (1–10).",
    "Motor_Skills_Score": "Motor & physical skills (1–10).",
    "Symptom_Intensity": "Intensity of ASD symptoms (1–10).",
    "ASD_Severity": "Overall ASD severity level (1–10).",
    "Autism_Diagnosis": "Is the patient diagnosed with autism? (Yes/No)",
    "Note": "Higher numbers indicate more critical condition."
}

for key, val in descriptions.items():
    tk.Label(
        description_frame,
        text=f"{key}: {val}",
        anchor="w",
        font=("Arial", 11),
        bg="#ffffff",
        fg="#0b2545"
    ).pack(fill="x", pady=2)

input_frame = tk.LabelFrame(
    root, text="Enter Patient Values",
    font=("Arial", 14), bg="#ffffff", fg="#0b2545",
    padx=10, pady=10)
input_frame.pack(fill="both", padx=20, pady=10)

entries = {}

def create_input(label):
    row = tk.Frame(input_frame, bg="#ffffff")
    row.pack(fill="x", pady=5)
    tk.Label(
        row, text=label, width=28, anchor="w",
        font=("Arial", 12), bg="#ffffff", fg="#0b2545"
    ).pack(side="left")
    ent = tk.Entry(
        row, width=20, font=("Arial", 12),
        bg="white", fg="black",
        insertbackground="black"
    )
    ent.pack(side="left")
    entries[label] = ent

numeric_fields = [
    "Age", "Communication_Score", "Social_Interaction_Score",
    "Behavioral_Score", "Cognitive_Score", "Motor_Skills_Score",
    "Symptom_Intensity", "ASD_Severity"
]

for field in numeric_fields:
    create_input(field)

row = tk.Frame(input_frame, bg="#ffffff")
row.pack(fill="x", pady=5)

tk.Label(row, text="Autism_Diagnosis", width=28, anchor="w",
         font=("Arial", 12), bg="#ffffff", fg="#0b2545").pack(side="left")

diagnosis_var = tk.StringVar()
dropdown = ttk.Combobox(
    row, textvariable=diagnosis_var,
    values=["Yes", "No"],
    state="readonly",
    width=18
)
dropdown.pack(side="left")
dropdown.current(0)

def predict():
    try:
        sample = {
            f: float(entries[f].get())
            for f in numeric_fields
        }
        sample["Autism_Diagnosis"] = diagnosis_var.get()
        sample_df = pd.DataFrame([sample])
        sample_df["Autism_Diagnosis"] = le_autism.transform(sample_df["Autism_Diagnosis"])
        pred_encoded = model.predict(sample_df)
        pred_final = le_y.inverse_transform(np.round(pred_encoded).astype(int))[0]
        messages = {
            "Improving": "Yes, the patient is showing improvement in therapy.",
            "Stable": "The patient is stable and maintaining progress.",
            "Not Improving": "The patient is not improving and may need further evaluation."
        }
        messagebox.showinfo("Prediction Result", messages.get(pred_final, pred_final))
    except Exception as e:
        messagebox.showerror("Error", f"Invalid input: {e}")

btn_frame = tk.Frame(root, bg="#e8f0fe")
btn_frame.pack(pady=20)

predict_btn = tk.Button(
    btn_frame,
    text="Proceed / Predict Therapy Progress",
    font=("Arial", 16, "bold"),
    bg="#1b4d89", fg="white",
    padx=20, pady=10,
    command=predict
)
predict_btn.pack()

root.mainloop()

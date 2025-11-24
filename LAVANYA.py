import subprocess
import sys
import importlib
import os
import io

required = ["pandas", "numpy", "scikit-learn"]

def silent_install(pkg):
    try:
        importlib.import_module(pkg.replace("-", ""))
    except ImportError:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", pkg],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

for pkg in required:
    silent_install(pkg)

import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

csv_data = """
Age,Communication_Score,Social_Interaction_Score,Behavioral_Score,Cognitive_Score,Motor_Skills_Score,Symptom_Intensity,ASD_Severity,Autism_Diagnosis,Therapy_Progress
27.0,3.75,1.85,2.62,6.73,5.72,3.94,4.53,No,Stable
9.4,9.51,5.42,2.47,7.97,8.05,4.73,7.24,Yes,Not Improving
35.3,7.32,8.73,9.06,2.5,7.6,8.55,6.87,Yes,Stable
25.7,5.99,7.32,2.5,6.25,1.54,3.4,3.53,No,Improving
8.8,1.56,8.07,2.72,5.72,1.49,8.7,4.96,No,Stable
38.6,1.56,6.59,7.59,8.33,2.68,0.88,3.48,No,Improving
22.2,0.58,6.92,4.5,9.06,3.61,7.77,6.52,Yes,Stable
5.7,8.66,8.49,7.77,0.12,4.08,8.48,6.85,Yes,Stable
26.2,6.01,2.5,0.65,6.74,6.8,1.82,3.13,No,Improving
12.4,7.08,4.89,4.88,0.52,0.57,4.3,3.87,No,Improving
32.7,0.21,2.21,0.34,5.49,0.35,1.65,2.77,No,Improving
33.3,9.7,9.88,0.63,2.88,3.92,7.07,7.03,Yes,Not Improving
39.2,8.32,9.44,9.06,3.07,6.97,5.35,7.32,Yes,Not Improving
21.6,2.12,0.39,1.39,3.53,1.93,6.35,3.04,No,Improving
19.8,1.82,7.06,5.32,6.21,6.42,1.96,5.75,Yes,Stable
30.9,1.83,9.25,4.11,3.34,2.6,2.12,3.39,No,Improving
7.9,3.04,1.81,3.47,7.33,8.86,0.41,4.67,No,Stable
23.2,5.25,5.68,9.0,4.05,8.96,3.22,5.97,Yes,Stable
23.2,4.32,9.15,0.22,0.68,2.97,5.6,4.75,No,Stable
6.3,2.91,0.34,6.64,7.84,2.3,8.58,7.49,Yes,Not Improving
18.5,6.12,6.97,9.63,2.86,4.11,6.67,5.58,Yes,Stable
34.6,1.39,2.97,5.6,4.33,2.41,4.35,3.26,No,Improving
34.7,2.92,9.24,9.37,6.85,6.72,9.53,6.77,Yes,Stable
6.7,3.66,9.71,0.52,3.32,8.26,7.19,7.2,Yes,Not Improving
6.4,4.56,9.44,4.19,0.57,6.73,9.3,8.58,Yes,Not Improving
12.7,7.85,4.74,2.6,3.74,8.24,5.28,7.46,Yes,Not Improving
35.4,2.0,8.62,7.31,9.44,3.97,2.59,6.55,Yes,Stable
7.8,5.14,8.45,9.81,6.42,1.56,0.53,4.88,No,Stable
29.6,5.92,3.19,2.57,6.71,7.38,7.26,5.58,Yes,Stable
6.7,0.46,8.29,6.54,6.32,3.6,1.21,2.49,No,Improving
26.1,6.08,0.37,1.98,1.99,6.71,3.03,5.42,No,Stable
16.2,1.71,5.96,5.65,4.18,2.71,5.32,5.05,No,Stable
32.2,0.65,2.3,4.64,7.51,0.81,5.64,3.27,No,Improving
11.6,9.49,1.21,9.72,1.01,9.93,6.01,7.64,Yes,Not Improving
26.7,9.66,0.77,6.09,2.78,1.56,1.66,2.95,No,Improving
11.7,8.08,6.96,3.5,2.76,9.88,3.8,6.52,Yes,Stable
29.4,3.05,3.4,1.14,4.32,9.77,6.17,4.73,No,Stable
35.3,0.98,7.25,1.51,9.8,7.94,9.7,7.36,Yes,Not Improving
7.7,6.84,0.65,2.25,0.68,6.59,7.28,4.71,No,Stable
35.1,4.4,3.15,2.51,5.19,5.78,9.23,4.41,No,Stable
25.0,1.22,5.39,8.51,1.79,8.66,7.62,5.85,Yes,Stable
7.7,4.95,7.91,5.61,9.71,2.89,5.92,7.52,Yes,Not Improving
18.8,0.34,3.19,5.23,1.13,4.68,1.92,2.54,No,Improving
9.0,9.09,6.26,1.15,4.04,6.19,6.67,7.21,Yes,Not Improving
28.9,2.59,8.86,8.6,7.38,4.11,6.23,5.52,Yes,Stable
31.1,6.63,6.16,7.23,7.05,4.27,6.02,6.54,Yes,Stable
6.9,3.12,2.33,0.68,4.23,3.3,4.9,2.91,No,Improving
21.7,5.2,0.24,7.08,3.47,5.64,5.29,5.25,No,Stable
33.1,5.47,8.7,5.44,3.98,8.51,3.34,6.05,Yes,Stable
37.6,1.85,0.21,0.82,2.64,2.02,5.19,3.22,No,Improving
22.0,9.7,8.75,4.58,2.05,9.34,1.98,7.35,Yes,Not Improving
38.2,7.75,5.29,4.85,4.83,6.89,8.05,7.64,Yes,Not Improving
22.8,9.39,9.39,1.66,2.69,8.23,1.86,6.21,Yes,Stable
17.6,8.95,7.99,9.46,2.87,5.56,0.85,4.92,No,Stable
"""

df = pd.read_csv(io.StringIO(csv_data))

y = df["Therapy_Progress"]
x = df.drop("Therapy_Progress", axis=1)

le_y = LabelEncoder()
y_encoded = le_y.fit_transform(y)

le_autism = LabelEncoder()
df["Autism_Diagnosis"] = le_autism.fit_transform(df["Autism_Diagnosis"])
x["Autism_Diagnosis"] = df["Autism_Diagnosis"]

model = RandomForestRegressor(max_depth=2, random_state=100)
model.fit(x, y_encoded)

# ---------------- GUI ----------------

root = tk.Tk()
root.title("Autism Therapy Progress Predictor")

# Make UI scale better on high-DPI displays and give a bit more height
root.tk.call('tk', 'scaling', 1.25)  # adjust if needed[web:8][web:11]
root.geometry("900x900")             # more room so button is visible on most screens

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
    "Note": "Higher the number means the critical condition"
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


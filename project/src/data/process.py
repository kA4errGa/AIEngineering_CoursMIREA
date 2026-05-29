import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def process_data():
    df = pd.read_csv("./data/RAW/Loan_default.csv")
    if "LoanID" in df.columns:
        df = df.drop("LoanID", axis=1)
    df = pd.get_dummies(df, columns=df.select_dtypes(include='object').columns)
    df[df.select_dtypes(include='bool').columns] = df.select_dtypes(include='bool').astype(np.int64)

    os.makedirs("./artifacts/EDA/figures/hist", exist_ok=True)
    os.makedirs("./artifacts/EDA/figures/boxPlot", exist_ok=True)

    for i in df.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.hist(df[i])
        ax.set_title(i)
        ax.grid(True)
        plt.savefig(f"./artifacts/EDA/figures/hist/hist_{i}.png")
        fig, ax = plt.subplots(figsize=(3, 6))
        ax.boxplot(df[i])
        ax.set_title(i)
        ax.grid(True)
        plt.savefig(f"./artifacts/EDA/figures/boxPlot/bp_{i}.png")

    df.to_csv("./data/PROCESSED/LoanDefaultPredictionDatasetPROC.csv", index=False)
    print("ГОТОВО!")
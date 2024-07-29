import pandas as pd

data = pd.read_csv("Patient_Data_Summary.csv")

data.to_excel("Patient_Data_Summary.xlsx", index=None, header=True)

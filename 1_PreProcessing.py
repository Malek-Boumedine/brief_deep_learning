import pandas as pd
import numpy as np
import missingno as msno

pd.set_option('display.max_columns', 500)

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

msno.matrix(df)

print(df.dtypes)

print(df.select_dtypes(include="object").head())

df["TotalCharges"] = df["TotalCharges"].str.strip().replace(",", ".", regex=True).replace("", np.nan).astype(float)
df["TotalCharges"].isna().sum()

for c in df.select_dtypes(include="object").columns :
    df[c] = df[c].str.strip().replace("", np.nan)
    print(f"valeurs uniques de {c} : {len(df[c].unique())} ")
    
    
for c in df.select_dtypes(include="object").columns :
    print(f"valeurs uniques de {c} : {df[c].unique()} ")
    
    





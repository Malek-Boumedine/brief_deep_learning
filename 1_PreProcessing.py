import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder



########################
# NETTOYAGE DES DONNEES
########################

pd.set_option('display.max_columns', 500)

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
df.columns = df.columns.str.replace(" ", "", regex=False)

print("shape : ", df.shape)

print(df.dtypes)

print(df.select_dtypes(include="object").head())

#  colonne TotalCharges : suppression des espaces, remplacement des vides par des NaN, conversion et type float
df["TotalCharges"] = df["TotalCharges"].str.strip().replace(",", ".", regex=True).replace("", np.nan).astype(float)
df["TotalCharges"].isna().sum()
df.dropna(inplace=True)

print("shape : ", df.shape)

boolean_cols = []
cat_cols = []
numeric_cols = df.select_dtypes(include="number").columns.tolist()
target = "Churn"


# colonnes objet : 
for c in df.select_dtypes(include="object").columns :
    df[c] = df[c].str.strip().replace("", np.nan)
    print(f"# {c} \n valeurs uniques : {len(df[c].unique())}") 
    if len(df[c].unique()) == 2 : 
        boolean_cols.append(c)
    if len(df[c].unique()) > 2 : 
        cat_cols.append(c)
    print(f"nombre de NaN : {df[c].isna().sum()}")
    print(f"valeurs uniques : {df[c].unique()}")
    print()


# colonnes numériques :
for c in df.select_dtypes(exclude="object").columns :
    print(f"# {c} \n valeurs uniques : {len(df[c].unique())}") 
    if len(df[c].unique()) == 2 : 
        boolean_cols.append(c)
    print(f"nombre de NaN : {df[c].isna().sum()}")
    print()    

boolean_cols.remove("Churn")

print("#"*50)
print("colonnes yes no : ", boolean_cols)
print("colonnes numeriques : ", numeric_cols)
print("colonnes 3 valeurs minimum :", cat_cols)
print("target :", target)

#########
#ENCODAGE
#########

X = df.drop(target, axis = 1)
y = df[target]

encoder = OneHotEncoder(drop='first')

bool_encod = encoder.fit_transform(X[boolean_cols])
df_bool_encod = pd.DataFrame(
    bool_encod.toarray(),
    columns=encoder.get_feature_names_out(boolean_cols),
    index=X.index
)


cat_encod = encoder.fit_transform(X[cat_cols])
df_cat_encod = pd.DataFrame(
    cat_encod.toarray(), 
    columns=encoder.get_feature_names_out(cat_cols), 
    index=X.index
)




##################
# MISE A L'ECHELLE
##################












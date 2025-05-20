import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split




########################
# NETTOYAGE DES DONNEES
########################

pd.set_option('display.max_columns', 500)

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
df.drop(["customerID"], axis = 1, inplace=True)
df.columns = df.columns.str.replace(" ", "", regex=False)

print("shape : ", df.shape)

print("DTYPES : \n\n", df.dtypes)
print(df.select_dtypes(include="object").head())

#  colonne TotalCharges : suppression des espaces, remplacement des vides par des NaN, conversion et type float
df["TotalCharges"] = df["TotalCharges"].str.strip().replace(",", ".", regex=True).replace("", np.nan).astype(float)
df["TotalCharges"].isna().sum()
print("nombre de null dans total charges avant traitement : ", df["TotalCharges"].isna().sum())
df = df.dropna(subset=["TotalCharges"])
print("nombre de null dans total charges après traitement : ", df["TotalCharges"].isna().sum())

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

encoder_bool = OneHotEncoder(drop='first')
bool_encod = encoder_bool.fit_transform(X[boolean_cols])
df_bool_encod = pd.DataFrame(
    bool_encod.toarray(),
    columns=encoder_bool.get_feature_names_out(boolean_cols),
    index=X.index
)

encoder_cat = OneHotEncoder(drop='first')
cat_encod = encoder_cat.fit_transform(X[cat_cols])
df_cat_encod = pd.DataFrame(
    cat_encod.toarray(), 
    columns=encoder_cat.get_feature_names_out(cat_cols), 
    index=X.index
)

label_enc = LabelEncoder()
y = label_enc.fit_transform(y)

X = X.drop(boolean_cols, axis=1)
X = X.drop(cat_cols, axis=1)

X = pd.concat([X, df_bool_encod, df_cat_encod], axis = 1)

print(X.shape)

##################
# MISE A L'ECHELLE
##################

X1, X_test, y1, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X1, y1, test_size=0.2, random_state=42, stratify=y1)











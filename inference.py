import numpy as np
import pandas as pd
import tensorflow as tf
import joblib



def load_model_and_transformers():
    """
    Charge le modèle entraîné et les transformations nécessaires
    """
    # chargement du modèle
    try:
        model = tf.keras.models.load_model("saved_model/churn_model")
        print("Modèle chargé avec succès.")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle: {e}")
        return None, None
    
    # chargement des transformations (encodeurs, scalers)
    try:
        transformers = joblib.load("saved_model/transformers.joblib")
        print("Transformations chargées avec succès.")
    except Exception as e:
        print(f"Erreur lors du chargement des transformations: {e}")
        return model, None
    
    return model, transformers

def preprocess_data(data, transformers):
    """
    Applique les mêmes transformations que pendant l'entraînement
    """
    # copie des données pour éviter de modifier l'original
    df = data.copy()
    
    # suppression de customerID si présent
    if "customerID" in df.columns:
        df.drop(["customerID"], axis=1, inplace=True)
    
    # nettoyage des noms de colonnes
    df.columns = df.columns.str.replace(" ", "", regex=False)
    
    # traitement de TotalCharges
    df["TotalCharges"] = df["TotalCharges"].str.strip().replace(",", ".", regex=True).replace("", np.nan).astype(float)
    df = df.dropna(subset=["TotalCharges"])
    
    # nettoyage des colonnes objet
    boolean_cols = transformers['boolean_cols']
    cat_cols = transformers['cat_cols']
    
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].str.strip().replace("", np.nan)
    
    # séparation des features et target
    X = df.drop('Churn', axis=1) if 'Churn' in df.columns else df
    
    # encodage des colonnes booléennes
    bool_encod = transformers['encoder_bool'].transform(X[boolean_cols])
    df_bool_encod = pd.DataFrame(
        bool_encod.toarray(),
        columns=transformers['encoder_bool'].get_feature_names_out(boolean_cols),
        index=X.index
    )
    
    # encodage des colonnes catégorielles
    cat_encod = transformers['encoder_cat'].transform(X[cat_cols])
    df_cat_encod = pd.DataFrame(
        cat_encod.toarray(),
        columns=transformers['encoder_cat'].get_feature_names_out(cat_cols),
        index=X.index
    )
    
    # suppression des colonnes originales
    X = X.drop(boolean_cols, axis=1)
    X = X.drop(cat_cols, axis=1)
    
    # concaténation des données encodées
    X = pd.concat([X, df_bool_encod, df_cat_encod], axis=1)
    
    # standardisation
    X = transformers['standard_scaler'].transform(X)
    
    return X

def predict_churn(model, data, transformers):
    """
    Prédit la probabilité de churn pour de nouvelles données
    """
    # Prétraitement
    X = preprocess_data(data, transformers)
    
    # Prédiction
    predictions = model.predict(X)
    
    # Pour un modèle de classification binaire avec softmax
    churn_probabilities = predictions[:, 1]
    
    # Décision binaire (0 = pas de churn, 1 = churn)
    churn_decisions = (churn_probabilities > 0.5).astype(int)
    
    return churn_probabilities, churn_decisions


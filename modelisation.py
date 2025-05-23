import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from PreProcessing import X_train, X_test, X_val, y_train, y_test, y_val, y
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score, f1_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
from utils import modelisation, results, roc_auc, build_model_random, build_model_grid
import datetime
import subprocess
import joblib
import keras_tuner as kt

# fixer les seeds : 
import random
import tensorflow as tf
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)



# modélisation
#### encodage de la target  avant modélisation

num_classes = len(np.unique(y))

y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)
y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes)
y_test_cat  = tf.keras.utils.to_categorical(y_test,  num_classes)


############################################################
### model baseline
##################

from sklearn.linear_model import LogisticRegression

# Création du modèle baseline - régression logistique

print("#"*50)
print("Création du modèle baseline - régression logistique")
print("#"*50)

baseline_model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',  # Gestion du déséquilibre de classes
    random_state=42
)

# Entraînement du modèle
baseline_model.fit(X_train, y_train)

# Prédictions sur l'ensemble de test
y_pred = baseline_model.predict(X_test)
y_pred_proba = baseline_model.predict_proba(X_test)[:, 1]  # Probabilité de la classe positive (Churn)

# Calcul des métriques demandées dans le brief
baseline_metrics = {
    'roc_auc': roc_auc_score(y_test, y_pred_proba),
    'f1_score': f1_score(y_test, y_pred),
    'recall': recall_score(y_test, y_pred)  # Recall sur la classe Churn=Yes
}

print("Performances du modèle baseline:")
for metric, value in baseline_metrics.items():
    print(f"{metric}: {value:.4f}")

print("#"*100)

############################################################
### models DL
#############

print("#"*50)
print("Création du modèle de deep learning")
print("#"*50)

model = modelisation(X_train, num_classes)

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_val, y_val_cat),
    epochs=40,
    batch_size=10,
    verbose=1,
    callbacks=[early_stop]
)

results(model, X_test, y_test, y_test_cat)
print("#"*100)

############################################################
# ModelCheckpoint
#################

checkpoint_path = "checkpoints/best_model1.keras"
os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
model_ckpt = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_accuracy', 
    save_best_only=True,
    verbose=1
)


############################################################
## gestion du déséquilibre
##########################

print("#"*50)
print("gestion du déséquilibre")
print("#"*50)

# Pour reset :
tf.keras.backend.clear_session()   # libère la mémoire, supprime le graph actuel
model = modelisation(X_train, num_classes)

# y_train doit être un vecteur d'entiers (0 et 1)
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
# Convertir en dictionnaire pour Keras
class_weight_dict = dict(enumerate(class_weights))
print(class_weight_dict)

log_dir = os.path.join(
    "logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
)
tensorboard_cb = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,       # enregistre les histogrammes de poids chaque époque
    write_graph=True,       # sauvegarde le graph du modèle
    write_images=True
)

model.fit(
    X_train, y_train_cat,
    validation_data=(X_val, y_val_cat),
    epochs=40,
    batch_size=10,
    verbose=1,
    callbacks=[early_stop, model_ckpt, tensorboard_cb],
    class_weight=class_weight_dict 
)

print("résultats du model après gestion du déséquilibre :")
results(model, X_test, y_test, y_test_cat)
print("#"*100)

############################################################
### Tensorboard
###############

log_dir = os.path.join(
    "logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
)
tensorboard_cb = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,       # enregistre les histogrammes de poids chaque époque
    write_graph=True,       # sauvegarde le graph du modèle
    write_images=True
)

subprocess.Popen(["tensorboard", "--logdir=logs/fit"])

print("#"*100)

############################################################
### COURBE ROC-AUC

print("#"*50)
print("COURBE ROC-AUC")
print("#"*50)

roc_auc(model, X_test, y_test)

print("#"*100)

############################################################
### fine tuning du model
########################

print("#"*50)
print("fine tuning du model - RandomSearch")
print("#"*50)

# random pour explorer plus de valeurs
random_tuner = kt.RandomSearch(
    lambda hp: build_model_random(X_train, hp, num_classes),
    objective='val_recall',
    max_trials=10,
    project_name='brief_deep_learning_random'
)

# recherche aléatoire
random_tuner.search(
    X_train,
    y_train_cat,
    epochs=20,
    validation_data=(X_val, y_val_cat),
    callbacks=[early_stop],
    class_weight=class_weight_dict,
    verbose=1
)

# résultats du random 
print("résultats du random search")
random_tuner.results_summary()

# meilleurs hyperparamètres du RandomSearch
best_hp = random_tuner.get_best_hyperparameters(1)[0]
print("Meilleurs hyperparamètres trouvés par RandomSearch:")
for param, value in best_hp.values.items():
    print(f"  {param}: {value}")

print("#"*100)

############################################################
# grid search
#############

print("#"*50)
print("fine tuning du model - GridSearch")
print("#"*50)

grid_tuner = kt.GridSearch(
    lambda hp: build_model_grid(hp, best_hp, X_train, num_classes),
    objective='val_recall',
    max_trials=10,
    directory='my_dir',
    project_name='brief_deep_learning_grid'
)

# lancer recherche grid
grid_tuner.search(
    X_train,
    y_train_cat,
    epochs=30,
    validation_data=(X_val, y_val_cat),
    callbacks=[early_stop],
    class_weight=class_weight_dict,
    verbose=1
)

# résultats du grid search 
print("résultats du grid search")
grid_tuner.results_summary()

############################################################
# Récupération du meilleur modele
#################################

# Récupérer le meilleur modèle final
best_model = grid_tuner.get_best_models(1)[0]

### sauvegarde du model final

os.makedirs("saved_model", exist_ok=True)
best_model.save("saved_model/churn_model.keras")
best_model.export("saved_model/churn_model")

print("Sauvegarde du model final réussie !")




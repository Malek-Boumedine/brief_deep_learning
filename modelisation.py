import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from PreProcessing import X_train, X_test, X_val, y_train, y_test, y_val, y
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from utils import modelisation, results
import datetime




# modélisation
#### encodage de la target  avant modélisation

num_classes = len(np.unique(y))

y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)
y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes)
y_test_cat  = tf.keras.utils.to_categorical(y_test,  num_classes)

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


print("########################################")

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

results(model, X_test, y_test, y_test_cat)








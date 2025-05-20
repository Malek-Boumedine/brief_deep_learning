import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from PreProcessing import X_train, X_test, X_val, y_train, y_test, y_val, y
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping


# modélisation
#### encodage de la target  avant modélisation

num_classes = len(np.unique(y))

y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)
y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes)
y_test_cat  = tf.keras.utils.to_categorical(y_test,  num_classes)


def modelisation() : 

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        # optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    return model

model = modelisation()



early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_val, y_val_cat),
    epochs=40,
    batch_size=10,
    verbose=1,
    callbacks=[early_stop]
)


test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"\nAccuracy sur le test set : {test_acc:.4f}")

from sklearn.metrics import classification_report, confusion_matrix


y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
print("\nClassification Report :")
print(classification_report(y_test, y_pred))












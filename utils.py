from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import seaborn as sns
import keras_tuner as kt




def modelisation(X_train, num_classes) : 
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        # optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    return model


def results(model, X_test, y_test, y_test_cat) : 
    test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"\nAccuracy sur le test set : {test_acc:.4f}")
    
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    print("\nClassification Report :")
    # print(classification_report(y_test, y_pred, target_names=iris.target_names))
    print(classification_report(y_test, y_pred))

    print("\nMatrice de Confusion :")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    plt.figure(figsize=(8, 6))
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Matrice de Confusion")
    plt.xlabel("Classe Prédite")
    plt.ylabel("Classe Réelle")
    plt.show()
    

def plot_loss_acc(history, validation=True):
    """
    Trace la loss et l'accuracy du modèle pendant l'entraînement.
    """
    plt.figure(figsize=(12, 4))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    if validation and 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Évolution de la Loss')
    plt.xlabel('Époque')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Acc')
    if validation and 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title("Évolution de l'Accuracy")
    plt.xlabel('Époque')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()
    

def roc_auc(model, X_test, y_test) : 
    
    from sklearn.metrics import roc_curve, roc_auc_score
    import matplotlib.pyplot as plt

    # Après avoir entraîné ton modèle
    y_pred_proba = model.predict(X_test)  # Probabilités prédites

    # Utiliser uniquement les probabilités de la classe positive (indice 1)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])
    auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])

    # Afficher le score AUC
    print(f"ROC-AUC: {auc_score:.4f}")

    # Tracer la courbe ROC
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')  # Ligne diagonale (modèle aléatoire)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()


def build_model_random(X_train, hp, num_classes):
  model = tf.keras.Sequential()

  model.add(tf.keras.layers.Input(shape=(X_train.shape[1],)))

  model.add(tf.keras.layers.Dense(
      hp.Choice('units', [8, 16, 32, 64, 128]),
      activation=hp.Choice('activation', ['relu', 'tanh', 'sigmoid'])))
  
  model.add(tf.keras.layers.Dense(
      hp.Choice('units2', [8, 16, 32, 64, 128]),
      activation=hp.Choice('activation2', ['relu', 'tanh', 'sigmoid'])))
  
  model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

  model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
      loss='categorical_crossentropy',
      metrics=['accuracy', tf.keras.metrics.Recall(name='recall')])
  return model


def build_model_grid(hp, best_hp, X_train, num_classes):
    units_values = [max(8, best_hp.get('units') - 32), best_hp.get('units'), min(128, best_hp.get('units') + 32)]
    units2_values = [max(8, best_hp.get('units2') - 32), best_hp.get('units2'), min(128, best_hp.get('units2') + 32)]
    
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(X_train.shape[1],)))
    model.add(tf.keras.layers.Dense(
        hp.Choice('units', units_values),
        activation=hp.Choice('activation', [best_hp.get('activation')])
    ))
    model.add(tf.keras.layers.Dense(
        hp.Choice('units2', units2_values),
        activation=hp.Choice('activation2', [best_hp.get('activation2')])
    ))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Recall(name='recall')])
    return model











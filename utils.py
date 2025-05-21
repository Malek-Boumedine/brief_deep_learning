from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf



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
    
    
def ModelCheckpoint() : 
    checkpoint_path = "checkpoints/best_model1.keras"
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    model_ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_accuracy',  # sauvegarde le modèle qui maximise l’accuracy de validation
        save_best_only=True,
        verbose=1
    )
    
    














import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

def evaluate_model():
    DATASET_DIR = "dataset/Train"
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    BATCH_SIZE = 32
    MODEL_PATH = "raigad_places_model.keras"

    print("Loading model...")
    model = load_model(MODEL_PATH)

    print("Loading validation dataset...")
    # Must use same seed as training to get the same split
    val_ds = image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        shuffle=True # Match training: Shuffle to get random split across all classes
    )
    
    class_names = val_ds.class_names
    print(f"Classes: {class_names}")

    print("Running predictions...")
    y_true = []
    y_pred = []

    for images, labels in val_ds:
        predictions = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(predictions, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    
    # Ensure target_names matches the classes actually present if possible, 
    # but usually val_ds.class_names covers all *possible* classes.
    # The error might be due to a mismatch in length or indices. 
    # Let's try passing labels parameter explicitly.
    try:
        unique_labels = np.unique(np.concatenate((y_true, y_pred)))
        print(classification_report(y_true, y_pred, labels=unique_labels, target_names=[class_names[i] for i in unique_labels]))
    except Exception as e:
        print(f"Report Generation Failed: {e}")
        print(f"Unique True: {np.unique(y_true)}")
        print(f"Unique Pred: {np.unique(y_pred)}")

    print("\n" + "="*50)
    print("CONFUSION MATRIX")
    print("="*50)
    cm = confusion_matrix(y_true, y_pred)
    
    # Print readable CM
    print(f"{'True / Pred':<30} | ", end="")
    for name in class_names:
        print(f"{name[:10]:<10} | ", end="")
    print("\n" + "-"*150)
    
    for i, row in enumerate(cm):
        print(f"{class_names[i]:<30} | ", end="")
        for val in row:
            print(f"{val:<10} | ", end="")
        print()
    
    # Check specific confusion: Samadhi (True) -> Temple (Pred)
    try:
        idx_samadhi = class_names.index("Raigad Fort Shivaji Maharaj Samadhi")
        idx_temple = class_names.index("Raigad Fort Jagadishwar Temple")
        
        misclassified = cm[idx_samadhi][idx_temple]
        total_samadhi = np.sum(cm[idx_samadhi])
        
        print(f"\nSpecific Analysis:")
        print(f"Samadhi images misclassified as Temple: {misclassified}/{total_samadhi} ({misclassified/total_samadhi*100:.1f}%)")
    except ValueError:
        pass

if __name__ == "__main__":
    evaluate_model()

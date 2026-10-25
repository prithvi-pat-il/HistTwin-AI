import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input, RandomFlip, RandomRotation, RandomZoom
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

def train_raigad_model():
    # Configuration
    DATASET_DIR = "dataset/Train"
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    BATCH_SIZE = 32
    EPOCHS = 25  # Increased epochs for better convergence
    MODEL_SAVE_PATH = "raigad_places_model.keras"

    if not os.path.exists(DATASET_DIR):
        print(f"Error: Dataset directory '{DATASET_DIR}' not found. Please run dataset_builder.py first.")
        return

    print("Loading dataset...")
    # Split into training and validation sets
    train_ds = image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )

    val_ds = image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )

    class_names = train_ds.class_names
    print(f"Found {len(class_names)} classes: {class_names}")

    # Calculate Class Weights
    y_train = []
    # Extract labels from the dataset to compute weights
    # Note: This iterates the dataset once.
    print("Computing class weights...")
    for _, label in train_ds.unbatch():
        y_train.append(label.numpy())
    
    y_train = np.array(y_train)
    
    class_weights_list = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = dict(enumerate(class_weights_list))
    print(f"Class Weights: {class_weights}")

    # Performance optimization
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Data Augmentation (New)
    data_augmentation = Sequential([
        RandomFlip("horizontal"),
        RandomRotation(0.2),  # Increased rotation
        RandomZoom(0.2),      # Increased zoom
    ])

    # --- STAGE 1: Train head with frozen base ---
    print("\n--- STAGE 1: Training top layers (frozen base) ---")
    
    # Build the Model using ResNet50
    # Base model with pre-trained weights
    base_model = ResNet50(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), include_top=False, weights='imagenet')
    base_model.trainable = False # Freeze base model

    inputs = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = data_augmentation(inputs)  # Apply augmentation
    x = preprocess_input(x) 
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(len(class_names), activation='softmax')(x)
    
    model = Model(inputs, outputs)

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    model.summary()

    # Train for a few epochs to warm up the head
    initial_epochs = 10
    history_warmup = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=initial_epochs,
        class_weight=class_weights
    )

    # --- STAGE 2: Fine-tuning ---
    print("\n--- STAGE 2: Fine-tuning (unfrozen base) ---")
    
    # Unfreeze the base model
    base_model.trainable = True

    # Fine-tune from this layer onwards
    # ResNet50 has 175 layers. We unfreeze the top ~75 layers.
    # ResNet50 has 175 layers. We unfreeze the top ~125 layers (starts at 0).
    fine_tune_at = 50
    
    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    # Recompile with a lower learning rate
    item_lr = 1e-5
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  optimizer=Adam(item_lr),
                  metrics=['accuracy'])

    model.summary()

    # Callbacks
    # Adjusted patience for EarlyStopping to 5
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-7, verbose=1)
    checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)

    # Train end-to-end
    total_epochs = EPOCHS
    
    history_fine = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=total_epochs,
        initial_epoch=history_warmup.epoch[-1] + 1,
        callbacks=[early_stop, reduce_lr, checkpoint],
        class_weight=class_weights
    )

    # Save class names structure
    with open("class_names.txt", "w") as f:
        for name in class_names:
            f.write(name + "\n")
    print("Class names saved to class_names.txt")

    # Plot results (combine histories)
    acc = history_warmup.history['accuracy'] + history_fine.history['accuracy']
    val_acc = history_warmup.history['val_accuracy'] + history_fine.history['val_accuracy']
    loss = history_warmup.history['loss'] + history_fine.history['loss']
    val_loss = history_warmup.history['val_loss'] + history_fine.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.ylim([0.4, 1])
    plt.plot([initial_epochs-1,initial_epochs-1],
              plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.ylim([0, 2.0])
    plt.plot([initial_epochs-1,initial_epochs-1],
              plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.savefig('training_results_augmented.png')
    print("Training results saved to training_results_augmented.png")

    # --- Evaluation & Confusion Matrix ---
    print("\nGenerating Confusion Matrix...")
    
    # Get true labels and predictions
    y_true = []
    y_pred = []
    
    for images, labels in val_ds:
        predictions = model.predict(images)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(predictions, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("Confusion Matrix saved to confusion_matrix.png")
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

if __name__ == "__main__":
    train_raigad_model()

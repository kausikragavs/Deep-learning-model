#Deep learning model for differentiating between a drone and a bird
# For good accuracy set epochs as 15 or above and set img_size as 180,180
#Done by S.Kausik Ragav
import os
import zipfile
import numpy as np
#use Python 3.12 as tensorflow is updated only till there
import tensorflow as tf
from tensorflow.keras import layers, models
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

#SETTINGS
ZIP_PATH = "dataset-dronevsbird.zip"
EXTRACT_DIR = "./data"
IMG_SIZE = (180, 180)
BATCH_SIZE = 32
EPOCHS = 15
#DATA PIPELINE
if not os.path.exists(EXTRACT_DIR):
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)
dataset_path=os.path.join(EXTRACT_DIR,"dataset")
# Note: Keras loads folders alphabetically: 'bird' = 0, 'drone' = 1
train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path, validation_split=0.2, subset="training", seed=42,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path, validation_split=0.2, subset="validation", seed=42,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE
)
print("Class names found:", train_ds.class_names)

# Optimization for speed
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

#THE CNN MODEL
model = models.Sequential([
    layers.Input(shape=IMG_SIZE +(3,)),layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),layers.RandomZoom(0.1),
    layers.Rescaling(1./255),
    
    # Feature Extraction
    layers.Conv2D(32, 3, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    
    layers.Conv2D(64, 3, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    
    layers.Conv2D(128, 3, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    
    # Classification Head
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(1, activation='sigmoid') # 0=Bird, 1=Drone
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#W&B INITIALIZATION
wandb.init(project="drone-vs-bird-cnn", config={"epochs": EPOCHS, "batch_size": BATCH_SIZE})

#TRAINING (clean output)
print("\n Training started. Check your W&B dashboard for real-time graphs!\n")

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=2, # This stops the line-by-line flooding
    callbacks=[
        WandbMetricsLogger(),
        WandbModelCheckpoint("best_drone_model.keras")
    ]
)

#INFERENCE FUNCTION
def predict_image(img_path):
    """Predicts if image is a drone or a bird."""
    img = tf.keras.utils.load_img(img_path, target_size=IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create batch axis

    prediction = model.predict(img_array, verbose=0)[0][0]
    
    label = "DRONE" if prediction > 0.5 else "BIRD"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    
    print(f"\n🔍 Analysis for {os.path.basename(img_path)}:")
    print(f"Result: {label} ({confidence*100:.2f}% Confidence)")

predict_image("test_drone.jpg")

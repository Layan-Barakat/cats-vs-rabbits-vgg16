# cats_vs_rabbits_vgg16.py
import os, json
import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras import layers, models
from keras.optimizers import Adam

# ==== paths ====
DATA_DIR   = r"C:\Users\layan\OneDrive\Documents\cat-vs-rabbit"
TRAIN_DIR  = os.path.join(DATA_DIR, "train-cat-rabbit")
VAL_DIR    = os.path.join(DATA_DIR, "val-cat-rabbit")

# ==== hyperparams ====
IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
EPOCHS     = 5
FT_EPOCHS  = 3
SEED       = 42
LR         = 1e-4
FT_LR      = 1e-5
OUT_DIR    = "./outputs"

os.makedirs(OUT_DIR, exist_ok=True)

# ==== datasets ====
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="int",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    labels="inferred",
    label_mode="int",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

class_names = train_ds.class_names
num_classes = len(class_names)
print("Classes:", class_names)

# cache + prefetch for speed
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(AUTOTUNE)
val_ds   = val_ds.cache().prefetch(AUTOTUNE)

# ==== model ====
data_aug = tf.keras.Sequential([
    layers.Rescaling(1./255),
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
], name="data_augmentation")

base = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base.trainable = False  # freeze for transfer learning

inputs = layers.Input(shape=(224, 224, 3))
x = data_aug(inputs)
x = base(x, training=False)
x = layers.Flatten()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)
model = models.Model(inputs, outputs)

model.compile(
    optimizer=Adam(LR),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ==== callbacks ====
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=3, restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(OUT_DIR, "best.keras"),
        monitor="val_accuracy",
        save_best_only=True
    ),
]

# ==== train classifier head ====
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)

# ==== fine-tune last few conv layers ====
for layer in base.layers[-4:]:
    layer.trainable = True

model.compile(
    optimizer=Adam(FT_LR),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

ft_history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FT_EPOCHS,
    callbacks=callbacks,
)

# ==== evaluate + save ====
val_loss, val_acc = model.evaluate(val_ds, verbose=0)
print(f"✅ Final Validation Accuracy: {val_acc:.4f}")

# save model + class names
model.save(os.path.join(OUT_DIR, "final.keras"))
with open(os.path.join(OUT_DIR, "labels.json"), "w") as f:
    json.dump(class_names, f)
print("Model and labels saved in the 'outputs' folder!")

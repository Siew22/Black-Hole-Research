# src/cnn_image_extractor.py (完整修复版)
import numpy as np
import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Flatten, Dense, Dropout, Input, BatchNormalization, Lambda
from keras.layers import Layer

# --- 修复 ImportError ---
# 在 Keras 3+ 中, ImageDataGenerator 被移到了 legacy 路径下
# 使用 try-except 块以兼容不同版本
try:
    from keras.src.legacy.preprocessing.image import ImageDataGenerator
except ImportError:
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

from keras.applications import ResNet50
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
import os
from src.utils import plot_history, plot_confusion_matrix
from keras import regularizers
import wandb
from wandb.integration.keras import WandbMetricsLogger # 移除 WandbModelCheckpoint
from tensorflow.keras.callbacks import ModelCheckpoint # 导入原生

# Import configuration
from config import GlobalConfig

class IdentityLayer(Layer):
    def call(self, inputs):
        return inputs

    def get_config(self):
        config = super().get_config()
        return config

def build_image_feature_extractor(img_size, num_classes):
    """
    Builds an image feature extractor and classifier using ResNet50 as a base.
    V2: Replaces Lambda with a custom serializable IdentityLayer.
    """
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
    base_model.trainable = False

    inputs = Input(shape=(img_size, img_size, 3))
    x = base_model(inputs, training=False)
    x = Flatten()(x)
    x = Dense(512, activation='relu', name='image_features_dense', kernel_regularizer=regularizers.l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    # 使用自定义的、可序列化的 IdentityLayer
    image_features = IdentityLayer(name='image_features_output')(x)

    outputs = Dense(num_classes, activation='softmax', name='classification_output', dtype='float32')(image_features)
    model = Model(inputs, outputs, name='image_classifier_and_feature_extractor_v2')
    return model

def run_cnn_image_extractor(X_images, y_labels, img_size=GlobalConfig.IMG_SIZE,
                            epochs=GlobalConfig.FEAT_EXTRACTOR_EPOCHS, batch_size=GlobalConfig.FEAT_EXTRACTOR_BATCH_SIZE,
                            wandb_run=None):
    print("\n--- Running CNN Image Feature Extractor Training (ResNet50) ---")
    num_classes = len(np.unique(y_labels))
    GlobalConfig.NUM_CLASSES = num_classes
    print(f"Image Extractor: Detected {num_classes} classes.")
    if num_classes <= 1:
        print("Error: Number of unique classes is <= 1. Cannot train classifier.")
        return

    X_images_norm = X_images.astype('float32') / 255.0
    y_categorical = tf.keras.utils.to_categorical(y_labels, num_classes=num_classes)

    X_train, X_test, y_train_cat, y_test_cat, y_train_orig, y_test_orig = train_test_split(
        X_images_norm, y_categorical, y_labels, test_size=0.2, random_state=GlobalConfig.RANDOM_SEED, stratify=y_labels
    )

    class_weights_val = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_orig), y=y_train_orig)
    class_weights_dict = {i : w for i, w in enumerate(class_weights_val)}
    print(f"Calculated Class Weights for Image Extractor: {class_weights_dict}")

    datagen = ImageDataGenerator(
        rotation_range=20, width_shift_range=0.1, height_shift_range=0.1,
        shear_range=0.1, zoom_range=0.1, horizontal_flip=True, fill_mode='nearest'
    )

    model = build_image_feature_extractor(img_size, num_classes)
    learning_rate = 1e-4
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=GlobalConfig.HP_TUNING_PATIENCE, restore_best_weights=True)
    ]
    if wandb_run:
        wandb_config = {"modality": "Image", "architecture": "ResNet50_Transfer", "input_shape": model.input_shape, "num_classes": num_classes, "learning_rate": learning_rate, "epochs": epochs, "batch_size": batch_size, "class_weights": class_weights_dict}
        wandb_run.config.update({"image_extractor": wandb_config})
        callbacks.append(WandbMetricsLogger(log_freq="epoch"))
        if GlobalConfig.WANDB_LOG_MODELS:
            # 使用原生 ModelCheckpoint
            checkpoint_filepath = os.path.join(GlobalConfig.MODELS_DIR, f"cnn_image_extractor_best_{wandb_run.id}.keras")
            # 确保目录存在
            os.makedirs(os.path.dirname(checkpoint_filepath), exist_ok=True) 
            callbacks.append(ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_loss', mode='min', save_best_only=True, save_weights_only=False))

    print("Training Image Extractor (using GPU if available)...")
    history = model.fit(
        datagen.flow(X_train, y_train_cat, batch_size=batch_size),
        epochs=epochs, validation_data=(X_test, y_test_cat),
        callbacks=callbacks,
        class_weight=class_weights_dict
    )
    
    plot_history(history, filename=os.path.join(GlobalConfig.RESULTS_DIR, 'cnn_image_extractor_history.png'))
    loss, accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"\nImage Extractor Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

    y_pred_probs = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    target_names_cls = [GlobalConfig.LABELS_MAP.get(i, f'Class {i}') for i in range(num_classes)]
    report = classification_report(y_test_orig, y_pred_classes, target_names=target_names_cls, labels=np.arange(num_classes), zero_division=0)
    cm = confusion_matrix(y_test_orig, y_pred_classes, labels=np.arange(num_classes))
    print("\nClassification Report (Image Extractor):\n", report)
    print("\nConfusion Matrix (Image Extractor):\n", cm)

    if not os.path.exists(GlobalConfig.RESULTS_DIR): os.makedirs(GlobalConfig.RESULTS_DIR)
    with open(os.path.join(GlobalConfig.RESULTS_DIR, 'cnn_image_extractor_metrics.txt'), 'w') as f:
        f.write(f"CNN Image Extractor Results:\nTest Loss: {loss:.4f}\nAccuracy: {accuracy:.4f}\n{report}\n\nCM:\n{str(cm)}")
    plot_confusion_matrix(cm, classes=target_names_cls, filename=os.path.join(GlobalConfig.RESULTS_DIR, 'cnn_image_extractor_confusion_matrix.png'))

    feature_extractor_model = Model(inputs=model.input, outputs=model.get_layer('image_features_output').output)
    feature_extractor_filepath = os.path.join(GlobalConfig.MODELS_DIR, 'image_feature_extractor.keras')
    if not os.path.exists(GlobalConfig.MODELS_DIR): os.makedirs(GlobalConfig.MODELS_DIR)
    feature_extractor_model.save(feature_extractor_filepath)
    print(f"Image Feature Extractor (ResNet50 based) saved to {feature_extractor_filepath}")

    full_classifier_model_path = os.path.join(GlobalConfig.MODELS_DIR, 'cnn_full_classifier_model.keras')
    model.save(full_classifier_model_path)
    print(f"Full CNN classifier model saved to {full_classifier_model_path}")
    
    print("CNN Image Feature Extractor training finished.")
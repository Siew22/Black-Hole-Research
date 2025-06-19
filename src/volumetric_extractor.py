# src/volumetric_extractor.py (完整文件)
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
import os
from src.utils import plot_history, plot_confusion_matrix, augment_volumetric_data
from tensorflow.keras.utils import Sequence
import wandb
from wandb.integration.keras import WandbMetricsLogger
from tensorflow.keras.callbacks import ModelCheckpoint
from config import GlobalConfig

# --- Data Generator for on-the-fly augmentation ---
class VolumetricDataGenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size, augment=False):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.augment = augment
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch_size))

    def __getitem__(self, idx):
        indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        X_batch_raw = self.x[indices]
        y_batch = self.y[indices]

        if self.augment:
            X_batch_aug = np.array([augment_volumetric_data(vol) for vol in X_batch_raw])
        else:
            X_batch_aug = X_batch_raw

        # Preprocessing is done here, inside the generator
        X_processed = X_batch_aug.astype('float32') / 255.0
        if X_processed.ndim == 4:
            X_processed = np.expand_dims(X_processed, axis=-1)
            
        return X_processed, y_batch
    
    def on_epoch_end(self):
        self.indices = np.arange(len(self.x))
        np.random.shuffle(self.indices)

# --- build_volumetric_feature_extractor (V4 - Increased Regularization) ---
def build_volumetric_feature_extractor(input_shape, num_classes):
    """ V5: Increased regularization with L2 and adjusted dropout. """
    l2_rate = 1e-5 # A small L2 regularization rate
    inputs = Input(shape=input_shape, name='volumetric_input')
    
    x = Conv3D(32, (3, 3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_rate))(inputs)
    x = BatchNormalization()(x)
    x = Conv3D(32, (3, 3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_rate))(x)
    x = BatchNormalization()(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = Dropout(0.35)(x) # Slightly increased

    x = Conv3D(64, (3, 3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_rate))(x)
    x = BatchNormalization()(x)
    x = Conv3D(64, (3, 3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_rate))(x)
    x = BatchNormalization()(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = Dropout(0.45)(x) # Slightly increased

    x = Conv3D(128, (3, 3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_rate))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x) 

    x = Flatten()(x)
    x = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(l2_rate))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    volumetric_features = Dense(256, activation='relu', name='volumetric_features_output', kernel_regularizer=regularizers.l2(l2_rate))(x)
    
    outputs = Dense(num_classes, activation='softmax', name='classification_output', dtype='float32')(volumetric_features)
    model = Model(inputs=inputs, outputs=outputs, name='volumetric_classifier_v5_l2_reg')
    return model

def run_volumetric_extractor(X_volumetric, y_labels,
                            epochs=GlobalConfig.FEAT_EXTRACTOR_EPOCHS, 
                            batch_size=GlobalConfig.FEAT_EXTRACTOR_BATCH_SIZE,
                            wandb_run=None):
    print("\n--- Running 3D Volumetric Feature Extractor Training (3D CNN v4 with Augmentation) ---")
    num_classes = len(np.unique(y_labels))
    GlobalConfig.NUM_CLASSES = num_classes
    print(f"Volumetric Extractor: Detected {num_classes} classes.")
    if num_classes <= 1:
        print("Error: Num classes for Volumetric <= 1. Cannot train classifier.")
        return

    y_categorical = tf.keras.utils.to_categorical(y_labels, num_classes=num_classes)
    
    X_train_raw, X_test_raw, y_train_cat, y_test_cat, y_train_orig, y_test_orig = train_test_split(
        X_volumetric, y_categorical, y_labels, test_size=0.2, random_state=GlobalConfig.RANDOM_SEED, stratify=y_labels
    )
    
    class_weights_val = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_orig), y=y_train_orig)
    class_weights_dict = {i : w for i, w in enumerate(class_weights_val)}
    print(f"Calculated Class Weights for Volumetric Extractor: {class_weights_dict}")

    # Create data generators
    train_generator = VolumetricDataGenerator(X_train_raw, y_train_cat, batch_size, augment=True)
    # Validation generator does not augment data but handles preprocessing
    val_generator = VolumetricDataGenerator(X_test_raw, y_test_cat, batch_size, augment=False)
    
    # Determine input shape from raw data before generator processing
    input_shape_for_model = X_train_raw.shape[1:] + (1,)
    model = build_volumetric_feature_extractor(input_shape=input_shape_for_model, num_classes=num_classes)
    
    learning_rate = 1e-4
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)]
    if wandb_run:
        wandb_config = {"modality": "Volumetric", "architecture": "3D CNN v4 Augmentation", "input_shape": model.input_shape, "num_classes": num_classes, "learning_rate": learning_rate, "epochs": epochs, "batch_size": batch_size, "class_weights": class_weights_dict}
        wandb_run.config.update({"volumetric_extractor": wandb_config})
        callbacks.append(WandbMetricsLogger(log_freq="epoch"))
        if GlobalConfig.WANDB_LOG_MODELS:
            # 使用原生 ModelCheckpoint
            checkpoint_filepath = os.path.join(GlobalConfig.MODELS_DIR, f"volumetric_extractor_best_{wandb_run.id}.keras")
            # 确保目录存在
            os.makedirs(os.path.dirname(checkpoint_filepath), exist_ok=True)
            callbacks.append(ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_loss', mode='min', save_best_only=True, save_weights_only=False))
    print("Training 3D Volumetric Extractor with data augmentation...")
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=callbacks,
        class_weight=class_weights_dict
    )
    
    # Evaluation
    plot_history(history, filename=os.path.join(GlobalConfig.RESULTS_DIR, 'volumetric_extractor_history.png'))
    loss, accuracy = model.evaluate(val_generator, verbose=0)
    print(f"\nVolumetric Extractor Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
    
    y_pred_probs = model.predict(val_generator)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    target_names_cls = [GlobalConfig.LABELS_MAP.get(i, f'Class {i}') for i in range(num_classes)]
    report = classification_report(y_test_orig, y_pred_classes, target_names=target_names_cls, labels=np.arange(num_classes), zero_division=0)
    cm = confusion_matrix(y_test_orig, y_pred_classes, labels=np.arange(num_classes))
    print("\nClassification Report (Volumetric Extractor):\n", report)
    print("\nConfusion Matrix (Volumetric Extractor):\n", cm)

    if not os.path.exists(GlobalConfig.RESULTS_DIR): os.makedirs(GlobalConfig.RESULTS_DIR)
    with open(os.path.join(GlobalConfig.RESULTS_DIR, 'volumetric_extractor_metrics.txt'), 'w') as f:
        f.write(f"Volumetric Extractor Results:\nTest Loss: {loss:.4f}\nAccuracy: {accuracy:.4f}\n{report}\n\nCM:\n{str(cm)}")
    plot_confusion_matrix(cm, classes=target_names_cls, filename=os.path.join(GlobalConfig.RESULTS_DIR, 'volumetric_extractor_cm.png'))

    # --- KEY FIX: Saving in .keras format ---
    feature_extractor_model = Model(inputs=model.input, outputs=model.get_layer('volumetric_features_output').output)
    feature_extractor_filepath = os.path.join(GlobalConfig.MODELS_DIR, 'volumetric_feature_extractor.keras')
    if not os.path.exists(GlobalConfig.MODELS_DIR): os.makedirs(GlobalConfig.MODELS_DIR)
    feature_extractor_model.save(feature_extractor_filepath)
    print(f"Volumetric Feature Extractor saved to {feature_extractor_filepath}")
    
    print("3D Volumetric Feature Extractor training finished.")
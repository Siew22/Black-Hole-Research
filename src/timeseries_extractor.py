# src/timeseries_extractor.py (最终优化版)
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Conv1D, MaxPooling1D, Bidirectional, LSTM, Attention, Flatten
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
import os
from src.utils import plot_history, plot_confusion_matrix
import wandb
from wandb.integration.keras import WandbMetricsLogger
from tensorflow.keras.callbacks import ModelCheckpoint
from config import GlobalConfig

def build_timeseries_feature_extractor(input_shape, num_classes):
    """
    V4: Adds L2 regularization to combat overfitting observed in GW model.
    """
    l2_rate = 1e-5
    inputs = Input(shape=input_shape, name='timeseries_input')

    # CNN Block
    x = Conv1D(filters=64, kernel_size=7, padding='causal', activation='relu', kernel_regularizer=regularizers.l2(l2_rate))(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.25)(x)

    x = Conv1D(filters=128, kernel_size=5, padding='causal', activation='relu', kernel_regularizer=regularizers.l2(l2_rate))(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.25)(x)

    # Bi-LSTM Block
    lstm_out = Bidirectional(LSTM(units=80, return_sequences=True, kernel_regularizer=regularizers.l2(l2_rate)))(x)
    lstm_out = Dropout(0.35)(lstm_out) 

    # Attention Block
    attention_out = Attention()([lstm_out, lstm_out])
    attention_out = BatchNormalization()(attention_out)

    # Final feature representation
    final_features = Flatten()(attention_out)
    timeseries_features = Dense(128, activation='relu', name='timeseries_features_output', kernel_regularizer=regularizers.l2(l2_rate))(final_features)
    timeseries_features = Dropout(0.4)(timeseries_features)

    # Classification head
    outputs = Dense(num_classes, activation='softmax', name='classification_output')(timeseries_features)

    model_name_prefix = "timeseries"
    if hasattr(inputs, 'name') and inputs.name:
        name_part = inputs.name.split('_')[0]
        if name_part: model_name_prefix = name_part

    model = Model(inputs=inputs, outputs=outputs, name=f'{model_name_prefix}_cnnlstm_attention_v4_l2')
    return model

def run_timeseries_extractor(X_timeseries, y_labels, modality_name="TimeSeries",
                            epochs=GlobalConfig.FEAT_EXTRACTOR_EPOCHS,
                            batch_size=GlobalConfig.FEAT_EXTRACTOR_BATCH_SIZE,
                            wandb_run=None):
    print(f"\n--- Running {modality_name} Feature Extractor Training (CNN-BiLSTM-Attention V3) ---")
    num_classes = GlobalConfig.NUM_CLASSES
    print(f"{modality_name} Extractor: Detected {num_classes} classes.")
    if num_classes <= 1:
        print(f"Error: Num classes for {modality_name} <= 1. Cannot train classifier.")
        return

    scaler = StandardScaler()
    original_shape = X_timeseries.shape
    X_timeseries_2d = X_timeseries.reshape(-1, original_shape[-1]) if X_timeseries.ndim > 1 else X_timeseries.reshape(-1, 1)
    X_timeseries_scaled_2d = scaler.fit_transform(X_timeseries_2d)
    X_timeseries_scaled = X_timeseries_scaled_2d.reshape(original_shape)
    X_timeseries_reshaped = np.expand_dims(X_timeseries_scaled, axis=-1) if X_timeseries_scaled.ndim == 2 else X_timeseries_scaled

    y_categorical = tf.keras.utils.to_categorical(y_labels, num_classes=num_classes)
    X_train, X_test, y_train_cat, y_test_cat, y_train_orig, y_test_orig = train_test_split(
        X_timeseries_reshaped, y_categorical, y_labels, test_size=0.2, random_state=GlobalConfig.RANDOM_SEED, stratify=y_labels
    )

    class_weights_val = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_orig), y=y_train_orig)
    class_weights_dict = {i : w for i, w in enumerate(class_weights_val)}
    print(f"Calculated Class Weights for {modality_name} Extractor: {class_weights_dict}")

    model = build_timeseries_feature_extractor(input_shape=(X_train.shape[1], X_train.shape[2]), num_classes=num_classes)
    
    learning_rate = 1e-4
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)]
    if wandb_run:
        wandb_config = {"modality": modality_name, "architecture": "CNN-BiLSTM-Attention-V3", "input_shape": model.input_shape, "num_classes": num_classes, "learning_rate": learning_rate, "epochs": epochs, "batch_size": batch_size, "class_weights": class_weights_dict}
        wandb_run.config.update({f"{modality_name.lower()}_extractor": wandb_config})
        callbacks.append(WandbMetricsLogger(log_freq="epoch"))
        if GlobalConfig.WANDB_LOG_MODELS:
            # 使用原生 ModelCheckpoint
            checkpoint_filepath = os.path.join(GlobalConfig.MODELS_DIR, f"{modality_name.lower()}_extractor_best_{wandb_run.id}.keras")
            # 确保目录存在
            os.makedirs(os.path.dirname(checkpoint_filepath), exist_ok=True)
            callbacks.append(ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_loss', mode='min', save_best_only=True, save_weights_only=False))

    print(f"Training {modality_name} Extractor (using GPU if available)...")
    history = model.fit(X_train, y_train_cat, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test_cat), callbacks=callbacks, class_weight=class_weights_dict)

    # Evaluation and saving logic (unchanged)
    plot_history(history, filename=os.path.join(GlobalConfig.RESULTS_DIR, f'{modality_name.lower()}_extractor_history.png'))
    loss, accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"\n{modality_name} Extractor Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

    y_pred_probs = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    target_names_cls = [GlobalConfig.LABELS_MAP.get(i, f'Class {i}') for i in range(num_classes)]
    report = classification_report(y_test_orig, y_pred_classes, target_names=target_names_cls, labels=np.arange(num_classes), zero_division=0)
    cm = confusion_matrix(y_test_orig, y_pred_classes, labels=np.arange(num_classes))
    print(f"\nClassification Report ({modality_name} Extractor):\n", report)
    print(f"\nConfusion Matrix ({modality_name} Extractor):\n", cm)

    if not os.path.exists(GlobalConfig.RESULTS_DIR): os.makedirs(GlobalConfig.RESULTS_DIR)
    with open(os.path.join(GlobalConfig.RESULTS_DIR, f'{modality_name.lower()}_extractor_metrics.txt'), 'w') as f:
        f.write(f"{modality_name} Extractor Results:\nTest Loss: {loss:.4f}\nAccuracy: {accuracy:.4f}\n{report}\n\nCM:\n{str(cm)}")
    
    plot_confusion_matrix(cm, classes=target_names_cls, filename=os.path.join(GlobalConfig.RESULTS_DIR, f'{modality_name.lower()}_extractor_cm.png'))

    feature_extractor_model = Model(inputs=model.input, outputs=model.get_layer('timeseries_features_output').output)
    feature_extractor_filepath = os.path.join(GlobalConfig.MODELS_DIR, f'{modality_name.lower()}_feature_extractor.keras')
    if not os.path.exists(GlobalConfig.MODELS_DIR): os.makedirs(GlobalConfig.MODELS_DIR)
    feature_extractor_model.save(feature_extractor_filepath)

    print(f"{modality_name} Feature Extractor saved to {feature_extractor_filepath}")
    print(f"{modality_name} Feature Extractor training finished.")
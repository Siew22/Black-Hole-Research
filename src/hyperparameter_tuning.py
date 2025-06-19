# src/hyperparameter_tuning.py (V3 - Hierarchical Fusion)
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, concatenate, Dropout, BatchNormalization, Attention
from tensorflow.keras import regularizers
import keras_tuner as kt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import class_weight
import os
import wandb
import gc
try:
    from wandb.integration.keras import WandbMetricsLogger
except ImportError:
    WandbMetricsLogger = None
    print("WandbMetricsLogger not found.")

from config import GlobalConfig
from src.multimodal_fusion_system import create_weighted_categorical_crossentropy, custom_gaussian_nll_loss_stacked, CustomProbabilisticRegression
from src.multimodal_fusion_system import IdentityLayer

# --- V3: build_model_for_tuner with Hierarchical Fusion ---
def build_model_for_tuner(hp, image_feature_dim, tabular_feature_dim, gw_feature_dim, xray_feature_dim, volumetric_feature_dim, num_classes, class_weights_dict=None):
    # Hyperparameters
    hp_fused_dense1_units = hp.Int('fused_dense1_units', min_value=256, max_value=768, step=128)
    hp_fused_dense2_units = hp.Int('fused_dense2_units', min_value=128, max_value=512, step=128)
    hp_dropout_shared1 = hp.Float('dropout_shared1', min_value=0.3, max_value=0.6, step=0.1)
    hp_dropout_shared2 = hp.Float('dropout_shared2', min_value=0.2, max_value=0.5, step=0.1)
    hp_reg_dense1_units = hp.Int('reg_dense1_units', min_value=64, max_value=256, step=64)
    hp_reg_dense2_units = hp.Int('reg_dense2_units', min_value=32, max_value=128, step=32)
    hp_reg_dropout1 = hp.Float('reg_dropout1', min_value=0.2, max_value=0.5, step=0.1)
    hp_reg_dropout2 = hp.Float('reg_dropout2', min_value=0.2, max_value=0.5, step=0.1)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-3, 5e-4, 1e-4, 5e-5])
    hp_l2_reg = hp.Choice('l2_reg', values=[0.001, 0.0001, 0.0])
    hp_regression_loss_weight = hp.Float('regression_loss_weight', min_value=0.01, max_value=0.2, step=0.02)
    hp_weak_fusion_units = hp.Int('weak_fusion_units', min_value=64, max_value=256, step=64)
    hp_weak_fusion_dropout = hp.Float('weak_fusion_dropout', 0.2, 0.5, step=0.1)

    # --- Hierarchical Architecture ---
    image_input = Input(shape=(image_feature_dim,), name='image_features_input')
    tabular_input = Input(shape=(tabular_feature_dim,), name='tabular_features_input')
    gw_input = Input(shape=(gw_feature_dim,), name='gw_features_input')
    xray_input = Input(shape=(xray_feature_dim,), name='xray_features_input')
    volumetric_input = Input(shape=(volumetric_feature_dim,), name='volumetric_features_input')

    weak_modalities_features = concatenate([gw_input, volumetric_input])
    weak_fused_representation = Dense(hp_weak_fusion_units, activation='relu')(weak_modalities_features)
    weak_fused_representation = BatchNormalization()(weak_fused_representation)
    weak_fused_representation = Dropout(hp_weak_fusion_dropout)(weak_fused_representation)
    
    final_fusion_input = concatenate([image_input, tabular_input, xray_input, weak_fused_representation, gw_input, volumetric_input])
    
    attention_scores = Dense(int(final_fusion_input.shape[-1] * 0.5), activation='relu')(final_fusion_input)
    attention_scores = Dense(final_fusion_input.shape[-1], activation='sigmoid')(attention_scores)
    fused_features_attn = tf.keras.layers.multiply([final_fusion_input, attention_scores])

    x_shared = Dense(hp_fused_dense1_units, activation='relu', kernel_regularizer=regularizers.l2(hp_l2_reg))(fused_features_attn)
    x_shared = BatchNormalization()(x_shared)
    x_shared = Dropout(hp_dropout_shared1)(x_shared, training=True)
    x_shared = Dense(hp_fused_dense2_units, activation='relu', kernel_regularizer=regularizers.l2(hp_l2_reg))(x_shared)
    x_shared = BatchNormalization()(x_shared)
    x_shared_output = Dropout(hp_dropout_shared2)(x_shared, training=True)

    classification_output = Dense(num_classes, activation='softmax', name='classification_output', dtype='float32')(x_shared_output)

    regression_branch = Dense(hp_reg_dense1_units, activation='relu', kernel_regularizer=regularizers.l2(hp_l2_reg))(x_shared_output)
    regression_branch = BatchNormalization()(regression_branch)
    regression_branch = Dropout(hp_reg_dropout1)(regression_branch, training=True)
    regression_branch = Dense(hp_reg_dense2_units, activation='relu', kernel_regularizer=regularizers.l2(hp_l2_reg))(regression_branch)
    regression_params_hidden = BatchNormalization()(regression_branch)
    regression_params_hidden = Dropout(hp_reg_dropout2)(regression_params_hidden, training=True)
    params_for_regression = Dense(2, activation=None, name='params_for_regression_tensor', dtype='float32')(regression_params_hidden)
    regression_output = CustomProbabilisticRegression(name='regression_output')(params_for_regression)

    model = Model(inputs=[image_input, tabular_input, gw_input, xray_input, volumetric_input],
                  outputs=[classification_output, regression_output],
                  name='tunable_hierarchical_model')
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=hp_learning_rate)
    weighted_classification_loss = create_weighted_categorical_crossentropy(class_weights_dict)

    model.compile(
        optimizer=optimizer,
        loss={'classification_output': weighted_classification_loss, 'regression_output': custom_gaussian_nll_loss_stacked},
        loss_weights={'classification_output': 1.0, 'regression_output': hp_regression_loss_weight},
        metrics={'classification_output': 'accuracy', 'regression_output': 'mae'}
    )
    return model

def run_hyperparameter_tuning(X_images_initial, X_tabular_initial, X_gw_initial, X_xray_initial, X_volumetric_initial, y_labels_initial, y_regression_initial,
                              image_extractor_path=None, tabular_extractor_path=None, gw_extractor_path=None,
                              xray_extractor_path=None, vol_extractor_path=None,
                              max_trials=GlobalConfig.HP_TUNING_MAX_TRIALS,
                              max_epochs=GlobalConfig.HP_TUNING_MAX_EPOCHS_PER_TRIAL,
                              patience=GlobalConfig.HP_TUNING_PATIENCE, wandb_run=None):
    print("\n--- Running Hyperparameter Tuning for Multimodal Fusion Model (Hierarchical) ---")
    
    image_extractor_path = image_extractor_path or os.path.join(GlobalConfig.MODELS_DIR, 'image_feature_extractor.keras')
    tabular_extractor_path = tabular_extractor_path or os.path.join(GlobalConfig.MODELS_DIR, 'tabular_feature_extractor.keras')
    gw_extractor_path = gw_extractor_path or os.path.join(GlobalConfig.MODELS_DIR, 'gw_feature_extractor.keras')
    xray_extractor_path = xray_extractor_path or os.path.join(GlobalConfig.MODELS_DIR, 'xray_feature_extractor.keras')
    vol_extractor_path = vol_extractor_path or os.path.join(GlobalConfig.MODELS_DIR, 'volumetric_feature_extractor.keras')

    try:
        custom_objects = {'IdentityLayer': IdentityLayer}
        image_feature_extractor = tf.keras.models.load_model(image_extractor_path, custom_objects=custom_objects, compile=False)
        tabular_feature_extractor = tf.keras.models.load_model(tabular_extractor_path, compile=False)
        gw_timeseries_extractor = tf.keras.models.load_model(gw_extractor_path, compile=False)
        xray_timeseries_extractor = tf.keras.models.load_model(xray_extractor_path, compile=False)
        volumetric_feature_extractor = tf.keras.models.load_model(vol_extractor_path, compile=False)
    except Exception as e:
        print(f"Error loading feature extractors for Keras Tuner: {e}. Ensure all extractors are trained."); return None

    print("Extracting features for HP tuning...")
    X_images_processed = X_images_initial.astype('float32') / 255.0
    img_feats = image_feature_extractor.predict(X_images_processed, verbose=0, batch_size=32)
    del X_images_processed; gc.collect()

    scaler_tabular_tune = StandardScaler().fit(X_tabular_initial)
    X_tabular_scaled = scaler_tabular_tune.transform(X_tabular_initial)
    tab_feats = tabular_feature_extractor.predict(X_tabular_scaled, verbose=0, batch_size=32)
    del X_tabular_scaled; gc.collect()

    scaler_gw_tune = StandardScaler().fit(X_gw_initial.reshape(-1, 1))
    X_gw_scaled = scaler_gw_tune.transform(X_gw_initial.reshape(-1, 1)).reshape(X_gw_initial.shape)
    X_gw_reshaped = np.expand_dims(X_gw_scaled, axis=-1)
    gw_feats = gw_timeseries_extractor.predict(X_gw_reshaped, verbose=0, batch_size=32)
    del X_gw_reshaped, X_gw_scaled; gc.collect()

    scaler_xray_tune = StandardScaler().fit(X_xray_initial.reshape(-1, 1))
    X_xray_scaled = scaler_xray_tune.transform(X_xray_initial.reshape(-1, 1)).reshape(X_xray_initial.shape)
    X_xray_reshaped = np.expand_dims(X_xray_scaled, axis=-1)
    xray_feats = xray_timeseries_extractor.predict(X_xray_reshaped, verbose=0, batch_size=32)
    del X_xray_reshaped, X_xray_scaled; gc.collect()

    X_vol_processed = X_volumetric_initial.astype('float32') / 255.0
    if X_vol_processed.ndim == 4: X_vol_processed = np.expand_dims(X_vol_processed, axis=-1)
    vol_feats = volumetric_feature_extractor.predict(X_vol_processed, verbose=0, batch_size=32)
    del X_vol_processed; gc.collect()
    print("Feature extraction complete. Original data released from memory.")
    
    image_dim, tab_dim, gw_dim, xray_dim, vol_dim, num_classes = \
        img_feats.shape[1], tab_feats.shape[1], gw_feats.shape[1], xray_feats.shape[1], vol_feats.shape[1], len(np.unique(y_labels_initial))
    print(f"Hyperparameter Tuning: Detected {num_classes} classes.")

    y_regression_log = np.log1p(np.maximum(0, y_regression_initial))
    scaler_regression_target_tune = MinMaxScaler().fit(y_regression_log.reshape(-1, 1))
    y_regression_scaled = scaler_regression_target_tune.transform(y_regression_log.reshape(-1, 1))
    
    (img_feats_train, img_feats_val, tab_feats_train, tab_feats_val, gw_feats_train, gw_feats_val,
     xray_feats_train, xray_feats_val, vol_feats_train, vol_feats_val, y_train_cls, y_val_cls,
     y_train_reg, y_val_reg) = train_test_split(
        img_feats, tab_feats, gw_feats, xray_feats, vol_feats, y_labels_initial, y_regression_scaled,
        test_size=0.2, random_state=GlobalConfig.RANDOM_SEED, stratify=y_labels_initial
    )

    y_train_cls_cat = tf.keras.utils.to_categorical(y_train_cls, num_classes=num_classes)
    y_val_cls_cat = tf.keras.utils.to_categorical(y_val_cls, num_classes=num_classes)
    
    class_weights_vals = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_cls), y=y_train_cls)
    class_weights_dict = {i: w for i, w in enumerate(class_weights_vals)}

    tuner = kt.RandomSearch(
        hypermodel=lambda hp: build_model_for_tuner(hp, image_dim, tab_dim, gw_dim, xray_dim, vol_dim, num_classes, class_weights_dict),
        objective=kt.Objective("val_classification_output_accuracy", direction="max"),
        max_trials=GlobalConfig.HP_TUNING_MAX_TRIALS,  # This now becomes the main controller
        executions_per_trial=1, # Run each trial only once
        directory=GlobalConfig.KERAS_TUNER_DIR,
        project_name='multimodal_hierarchical_tuning_random', # New name to avoid conflicts
        overwrite=True
    )
    
    # The search_space_summary and subsequent code remains the same...
    tuner.search_space_summary()
    print("Starting hyperparameter search...")
    callbacks_tuner = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)]
    if wandb_run and WandbMetricsLogger:
        callbacks_tuner.append(WandbMetricsLogger(log_freq="epoch"))

    X_train_dict = {'image_features_input': img_feats_train, 'tabular_features_input': tab_feats_train, 
                    'gw_features_input': gw_feats_train, 'xray_features_input': xray_feats_train, 
                    'volumetric_features_input': vol_feats_train}
    y_train_dict = {'classification_output': y_train_cls_cat, 'regression_output': y_train_reg}
    X_val_dict = {'image_features_input': img_feats_val, 'tabular_features_input': tab_feats_val,
                  'gw_features_input': gw_feats_val, 'xray_features_input': xray_feats_val,
                  'volumetric_features_input': vol_feats_val}
    y_val_dict = {'classification_output': y_val_cls_cat, 'regression_output': y_val_reg}

    tuner.search(X_train_dict, y_train_dict,
                 epochs=max_epochs,
                 validation_data=(X_val_dict, y_val_dict),
                 callbacks=callbacks_tuner)

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("\n--- Best Hyperparameters Found ---")
    best_hps_values = {}
    for param, value in best_hps.values.items():
        print(f"{param}: {value}")
        best_hps_values[f"best_hp/{param}"] = value
    if wandb_run:
        wandb_run.config.update(best_hps_values)

    best_model = build_model_for_tuner(best_hps, image_dim, tab_dim, gw_dim, xray_dim, vol_dim, num_classes, class_weights_dict)
    
    if not os.path.exists(GlobalConfig.MODELS_DIR): os.makedirs(GlobalConfig.MODELS_DIR)
    best_model_path = os.path.join(GlobalConfig.MODELS_DIR, 'multimodal_fusion_model_tuned.keras')
    best_model.save(best_model_path)
    print(f"Best tuned model saved to {best_model_path}")
    
    if wandb_run and GlobalConfig.WANDB_LOG_MODELS:
        try:
            artifact = wandb.Artifact('tuned_fusion_model', type='model', metadata=best_hps.values)
            artifact.add_file(best_model_path)
            wandb_run.log_artifact(artifact)
        except Exception as e: print(f"Error logging artifact: {e}")
    
    print("Hyperparameter tuning finished.")
    return best_model_path
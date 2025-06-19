# src/multimodal_fusion_system.py (V3 - Hierarchical Fusion)
import numpy as np
import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Input, Dense, concatenate, Dropout, BatchNormalization, Attention, Layer
from keras import regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
from sklearn.utils import class_weight
import os
from src.utils import plot_history, plot_confusion_matrix
import wandb
from wandb.integration.keras import WandbMetricsLogger
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from config import GlobalConfig

# --- Custom Layers and Losses (Unchanged but necessary) ---
@keras.saving.register_keras_serializable()
class IdentityLayer(Layer):
    def call(self, inputs):
        return inputs
    def get_config(self):
        config = super().get_config()
        return config

class CustomProbabilisticRegression(keras.layers.Layer):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        if 'dtype' not in kwargs:
            if hasattr(self, 'dtype_policy') and self.dtype_policy.name == 'mixed_float16': kwargs['dtype'] = tf.float16
            else: kwargs['dtype'] = tf.float32
    def call(self, inputs):
        mean_param = inputs[..., :1]; variance_param_raw = inputs[..., 1:]
        processed_variance = keras.activations.softplus(tf.cast(variance_param_raw, self.compute_dtype)) + keras.backend.epsilon()
        return tf.concat([tf.cast(mean_param, self.compute_dtype), tf.cast(processed_variance, self.compute_dtype)], axis=-1)
    def compute_output_shape(self, input_shape): return (input_shape[0], 2)
    def compute_output_signature(self, input_signature):
        return tf.TensorSpec(shape=(input_signature.shape[0], 2), dtype=self.compute_dtype if hasattr(self, 'compute_dtype') else self.dtype, name='regression_params_mean_var')
    def get_config(self): return super().get_config()

def create_weighted_categorical_crossentropy(class_weights_dict):
    def weighted_loss(y_true, y_pred):
        if not class_weights_dict:
            return keras.losses.categorical_crossentropy(y_true, y_pred)
        class_indices = tf.argmax(y_true, axis=-1)
        weights_values = list(class_weights_dict.values())
        weights = tf.constant(weights_values, dtype=tf.float32)
        sample_weights = tf.gather(weights, class_indices)
        unweighted_loss = keras.losses.categorical_crossentropy(y_true, y_pred)
        weighted_loss = unweighted_loss * sample_weights
        return tf.reduce_mean(weighted_loss)
    return weighted_loss

def custom_gaussian_nll_loss_stacked(y_true, y_pred_stacked):
    y_true_casted = tf.cast(y_true, dtype=y_pred_stacked.dtype)
    y_pred_mean = y_pred_stacked[..., :1]; y_pred_variance = y_pred_stacked[..., 1:]
    safe_variance = y_pred_variance + keras.backend.epsilon()
    log_two_pi = tf.cast(tf.math.log(2. * np.pi), dtype=safe_variance.dtype)
    log_likelihood = -0.5 * (log_two_pi + tf.math.log(safe_variance) + tf.square(y_true_casted - y_pred_mean) / safe_variance)
    return -tf.reduce_mean(log_likelihood)

def build_tabular_feature_extractor(input_dim):
    tabular_input = Input(shape=(input_dim,), name='tabular_input')
    x = Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(tabular_input)
    x = BatchNormalization()(x); x = Dropout(0.3)(x)
    tabular_features = Dense(32, activation='relu', name='tabular_features_output', kernel_regularizer=keras.regularizers.l2(0.001))(x)
    return Model(inputs=tabular_input, outputs=tabular_features, name='tabular_feature_extractor')

# --- V3: Hierarchical Fusion Model ---
def build_multimodal_fusion_model_probabilistic_uq(image_feature_dim, tabular_feature_dim, gw_feature_dim, xray_feature_dim, volumetric_feature_dim, num_classes, l2_reg=1e-4):
    image_input = Input(shape=(image_feature_dim,), name='image_features_input')
    tabular_input = Input(shape=(tabular_feature_dim,), name='tabular_features_input')
    gw_input = Input(shape=(gw_feature_dim,), name='gw_features_input')
    xray_input = Input(shape=(xray_feature_dim,), name='xray_features_input')
    volumetric_input = Input(shape=(volumetric_feature_dim,), name='volumetric_features_input')

    weak_modalities_features = concatenate([gw_input, volumetric_input], name='weak_modalities_concat')
    weak_fused_representation = Dense(128, activation='relu', name='weak_pre_fusion_dense')(weak_modalities_features)
    weak_fused_representation = BatchNormalization(name='weak_pre_fusion_bn')(weak_fused_representation)
    weak_fused_representation = Dropout(0.4, name='weak_pre_fusion_dropout')(weak_fused_representation)

    final_fusion_input = concatenate([
        image_input, tabular_input, xray_input,
        weak_fused_representation, gw_input, volumetric_input
    ], name='hierarchical_fusion_concat')

    attention_dense_units = int(final_fusion_input.shape[-1] * 0.5); attention_dense_units = max(1, attention_dense_units)
    attention_scores = Dense(attention_dense_units, activation='relu', name='attention_relu', kernel_regularizer=regularizers.l2(l2_reg))(final_fusion_input)
    attention_scores = Dense(final_fusion_input.shape[-1], activation='sigmoid', name='attention_weights')(attention_scores)
    fused_features_attn = keras.layers.multiply([final_fusion_input, attention_scores], name='fused_features_with_attention')

    x_shared = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(fused_features_attn)
    x_shared = BatchNormalization()(x_shared)
    x_shared = Dropout(0.5)(x_shared)
    x_shared = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(x_shared)
    x_shared = BatchNormalization()(x_shared)
    x_shared_output = Dropout(0.4)(x_shared)

    classification_output_tensor = Dense(num_classes, activation='softmax', name='classification_output', dtype='float32')(x_shared_output)
    
    regression_branch = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(x_shared_output)
    regression_branch = BatchNormalization()(regression_branch)
    regression_branch = Dropout(0.3)(regression_branch)
    regression_branch = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(regression_branch)
    regression_branch = BatchNormalization()(regression_branch)
    regression_params_hidden = Dropout(0.3)(regression_branch)
    params_for_regression = Dense(2, activation=None, name='params_for_regression_tensor', dtype='float32')(regression_params_hidden)
    regression_output_tensor = CustomProbabilisticRegression(name='regression_output')(params_for_regression)
    
    model = Model(
        inputs=[image_input, tabular_input, gw_input, xray_input, volumetric_input], 
        outputs=[classification_output_tensor, regression_output_tensor], 
        name='hierarchical_multimodal_regressor'
    )
    return model

class MultimodalFusionSystem:
    def __init__(self, num_classes, fusion_model_path=None,
                 image_extractor_path=None, tabular_feature_extractor_model=None,
                 gw_timeseries_extractor_path=None, xray_timeseries_extractor_path=None,
                 volumetric_extractor_path=None,
                 use_probabilistic_uq=False,
                 strategy=None,
                 use_log_transform_regression=False):
        self.num_classes = num_classes
        self.use_log_transform_regression = use_log_transform_regression
        self.fusion_model_path = fusion_model_path if fusion_model_path else os.path.join(GlobalConfig.MODELS_DIR, 'multimodal_fusion_model.keras')
        if not self.fusion_model_path.endswith(".keras"): self.fusion_model_path = os.path.splitext(self.fusion_model_path)[0] + ".keras"
        self.use_probabilistic_uq = use_probabilistic_uq
        self.strategy = strategy if strategy else tf.distribute.get_strategy()
        self.fusion_model = None
        self.image_extractor = None
        self.tabular_extractor = None
        self.gw_timeseries_extractor = None
        self.xray_timeseries_extractor = None
        self.volumetric_extractor = None
        self.model_input_dims_cache = {} 
        self._load_extractors(image_extractor_path, tabular_feature_extractor_model, gw_timeseries_extractor_path, xray_timeseries_extractor_path, volumetric_extractor_path)
        self.scaler_tabular = StandardScaler()
        self.scaler_gw = StandardScaler()
        self.scaler_xray = StandardScaler()
        self.scaler_regression_target = MinMaxScaler()
        self.scaler_anomaly_features = MinMaxScaler()
        self.is_scaler_fitted = False
        self.replay_buffer = {k: [] for k in ['images', 'tabular', 'gw', 'xray', 'volumetric', 'labels', 'regression_targets']}
        self.max_buffer_size = GlobalConfig.REPLAY_BUFFER_MAX_SIZE
        self.active_learning_uncertainty_threshold = GlobalConfig.ACTIVE_LEARNING_UNCERTAINTY_THRESHOLD
        self.active_learning_selection_ratio = GlobalConfig.ACTIVE_LEARNING_SELECTION_RATIO
        self.anomaly_detector = None
        self.anomaly_threshold = 0.05
        self.model_input_dims_cache = {}
        self.input_names, self.output_names = [], []
        self.class_weights = None

    def _fit_scalers(self, tabular_data, gw_data, xray_data, regression_targets_raw):
        if tabular_data is not None and tabular_data.shape[0] > 0: self.scaler_tabular.fit(tabular_data)
        if gw_data is not None and gw_data.shape[0] > 0: self.scaler_gw.fit(gw_data.reshape(-1, 1))
        if xray_data is not None and xray_data.shape[0] > 0: self.scaler_xray.fit(xray_data.reshape(-1, 1))
        if regression_targets_raw is not None and regression_targets_raw.shape[0] > 0:
            targets_to_scale = np.array(regression_targets_raw).reshape(-1, 1)
            if self.use_log_transform_regression:
                print("Applying log1p transform to regression target for scaling.")
                targets_to_scale = np.log1p(np.maximum(0, targets_to_scale))
            if not np.all(np.isfinite(targets_to_scale)):
                max_finite = np.max(targets_to_scale[np.isfinite(targets_to_scale)], initial=-np.inf)
                min_finite = np.min(targets_to_scale[np.isfinite(targets_to_scale)], initial=np.inf)
                if max_finite == -np.inf: max_finite = 1.0
                if min_finite == np.inf: min_finite = 0.0
                targets_to_scale = np.nan_to_num(targets_to_scale, nan=min_finite, posinf=max_finite, neginf=min_finite)
                print(f"Handled non-finite values in regression target. Min/Max after handling: {min_finite}/{max_finite}")
            self.scaler_regression_target.fit(targets_to_scale)
        self.is_scaler_fitted = True
        print("All data scalers have been fitted.")
    
    def _load_extractors(self, img_ep, tab_ep, gw_ep, xray_ep, vol_ep):
        try:
            custom_objects = {'IdentityLayer': IdentityLayer, 'CustomProbabilisticRegression': CustomProbabilisticRegression}
            self.image_extractor = keras.saving.load_model(img_ep, custom_objects=custom_objects, compile=False) if img_ep and os.path.exists(img_ep) else None
            self.tabular_extractor = keras.saving.load_model(tab_ep, compile=False) if isinstance(tab_ep, str) and os.path.exists(tab_ep) else (tab_ep if isinstance(tab_ep, keras.Model) else None)
            self.gw_timeseries_extractor = keras.saving.load_model(gw_ep, compile=False) if gw_ep and os.path.exists(gw_ep) else None
            self.xray_timeseries_extractor = keras.saving.load_model(xray_ep, compile=False) if xray_ep and os.path.exists(xray_ep) else None
            self.volumetric_extractor = keras.saving.load_model(vol_ep, compile=False) if vol_ep and os.path.exists(vol_ep) else None
            print("Feature extractors loaded/initialized.")
            for name, extractor in [("Image", self.image_extractor), ("Tabular", self.tabular_extractor), ("GW", self.gw_timeseries_extractor), ("XRAY", self.xray_timeseries_extractor), ("Volumetric", self.volumetric_extractor)]:
                print(f"  {name} Extractor: {'Loaded' if extractor else 'Not Loaded/Path Invalid'}")
                if extractor:
                    extractor.trainable = False
                    self.model_input_dims_cache[f'{name.lower()}_extractor_native_dim'] = extractor.output_shape[-1]
        except Exception as e:
            print(f"Warning: Error loading some feature extractors: {e}")

    def _extract_features(self, images, tabular, gw, xray, volumetric, batch_size_pred=32):
        if not self.is_scaler_fitted: self._fit_scalers(tabular, gw, xray, np.zeros((1,)))
        num_samples = next((src.shape[0] for src in [images, tabular, gw, xray, volumetric] if src is not None and hasattr(src, 'shape') and src.shape[0] > 0), 0)
        if num_samples == 0: return (np.array([]),)*5
        def get_feats(extractor, data, scaler, feat_dim_key, default_dim, preprocess_fn=None, reshape_fn=None):
            feat_dim = self.model_input_dims_cache.get(feat_dim_key, default_dim)
            if data is None or data.shape[0] == 0 or extractor is None: return np.zeros((num_samples, feat_dim), dtype=np.float32)
            processed = preprocess_fn(data) if preprocess_fn else data
            if scaler and hasattr(scaler, 'n_features_in_'):
                shape = processed.shape
                reshaped = processed.reshape(-1, 1) if processed.ndim == 1 else processed.reshape(-1, shape[-1])
                if reshaped.shape[-1] == scaler.n_features_in_: processed = scaler.transform(reshaped).reshape(shape)
                else: print(f"Warning: Scaler mismatch for {feat_dim_key}.")
            processed = reshape_fn(processed) if reshape_fn else processed
            try: return extractor.predict(processed, verbose=0, batch_size=batch_size_pred)
            except Exception as e: print(f"Error predicting {feat_dim_key}: {e}"); return np.zeros((num_samples, feat_dim), dtype=np.float32)
        
        img_feats = get_feats(self.image_extractor, images, None, 'image_extractor_native_dim', GlobalConfig.DUMMY_IMAGE_FEATURE_DIM, lambda x: x.astype('float32')/255.0)
        tab_feats = get_feats(self.tabular_extractor, tabular, self.scaler_tabular, 'tabular_extractor_native_dim', GlobalConfig.DUMMY_TABULAR_FEATURE_DIM)
        gw_feats = get_feats(self.gw_timeseries_extractor, gw, self.scaler_gw, 'gw_extractor_native_dim', GlobalConfig.DUMMY_GW_FEATURE_DIM, reshape_fn=lambda x: np.expand_dims(x, -1) if x.ndim==2 else x)
        xray_feats = get_feats(self.xray_timeseries_extractor, xray, self.scaler_xray, 'xray_extractor_native_dim', GlobalConfig.DUMMY_XRAY_FEATURE_DIM, reshape_fn=lambda x: np.expand_dims(x, -1) if x.ndim==2 else x)
        vol_feats = get_feats(self.volumetric_extractor, volumetric, None, 'volumetric_extractor_native_dim', GlobalConfig.DUMMY_VOLUMETRIC_FEATURE_DIM, lambda x: x.astype('float32')/255.0, lambda x: np.expand_dims(x,-1) if x.ndim==4 else x)
        return img_feats, tab_feats, gw_feats, xray_feats, vol_feats

    def _get_or_create_fusion_model(self):
        if self.fusion_model:
            self._compile_fusion_model(self.fusion_model, class_weights_dict=self.class_weights)
            return
        with self.strategy.scope():
            if os.path.exists(self.fusion_model_path):
                print(f"Loading existing fusion model from {self.fusion_model_path}...")
                custom_objects = {'CustomProbabilisticRegression': CustomProbabilisticRegression, 'custom_gaussian_nll_loss_stacked': custom_gaussian_nll_loss_stacked, 'weighted_loss':create_weighted_categorical_crossentropy(self.class_weights)}
                self.fusion_model = keras.saving.load_model(self.fusion_model_path, custom_objects=custom_objects, compile=False)
            else:
                print("Building new hierarchical fusion model from scratch...")
                dims = {k: self.model_input_dims_cache.get(f'{k}_extractor_native_dim', getattr(GlobalConfig, f'DUMMY_{k.upper()}_FEATURE_DIM')) for k in ['image', 'tabular', 'gw', 'xray', 'volumetric']}
                self.fusion_model = build_multimodal_fusion_model_probabilistic_uq(dims['image'], dims['tabular'], dims['gw'], dims['xray'], dims['volumetric'], self.num_classes)
        
        self.input_names = [layer.name for layer in self.fusion_model.inputs]
        self.output_names = self.fusion_model.output_names
        self._compile_fusion_model(self.fusion_model, learning_rate=5e-5, class_weights_dict=self.class_weights)
        print(f"Fusion model is ready. Input names: {self.input_names}, Output names: {self.output_names}")

    def _compile_fusion_model(self, model_to_compile, learning_rate=5e-5, class_weights_dict=None):
        with self.strategy.scope():
            classification_output_name, regression_output_name = 'classification_output', 'regression_output'
            loss_weights_dict = {classification_output_name: 1.0, regression_output_name: 0.03}
            classification_loss = create_weighted_categorical_crossentropy(class_weights_dict)
            if self.use_probabilistic_uq:
                losses_dict = {classification_output_name: classification_loss, regression_output_name: custom_gaussian_nll_loss_stacked}
                def mae_on_mean_from_stacked(y_true, y_pred_stacked): return keras.metrics.mean_absolute_error(tf.cast(y_true, dtype=y_pred_stacked.dtype), y_pred_stacked[..., :1])
                metrics_dict = {classification_output_name: ['accuracy'], regression_output_name: [mae_on_mean_from_stacked]}
            else:
                losses_dict = {classification_output_name: classification_loss, regression_output_name: 'mse'}
                metrics_dict = {classification_output_name: ['accuracy'], regression_output_name: ['mse', 'mae']}
            model_to_compile.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss=losses_dict, loss_weights=loss_weights_dict, metrics=metrics_dict)
            print(f"Fusion model compiled successfully with custom weighted loss and LR={learning_rate}.")

    def run_initial_fusion_training(self, X_images_initial, X_tabular_initial, X_gw_initial,
                                    X_xray_initial, X_volumetric_initial, y_labels_initial,
                                    y_regression_initial,
                                    epochs=GlobalConfig.INITIAL_TRAINING_EPOCHS,
                                    batch_size_per_replica=GlobalConfig.INITIAL_TRAINING_BATCH_SIZE,
                                    wandb_run=None):
        print("\n--- Initial Training of Hierarchical Fusion Model ---")
        if y_labels_initial is None or len(y_labels_initial) == 0:
            print("No initial labels provided. Cannot train."); return
        if not self.is_scaler_fitted:
            self._fit_scalers(X_tabular_initial, X_gw_initial, X_xray_initial, y_regression_initial)
        unique_labels = np.unique(y_labels_initial)
        class_weights_vals = class_weight.compute_class_weight('balanced', classes=unique_labels, y=y_labels_initial)
        self.class_weights = {i: float(weight) for i, weight in enumerate(class_weights_vals)}
        print(f"Calculated class weights for fusion model: {self.class_weights}")
        self._get_or_create_fusion_model()
        self.fusion_model.summary(line_length=120)
        
        img_feats, tab_feats, gw_feats, xray_feats, vol_feats = self._extract_features(
            X_images_initial, X_tabular_initial, X_gw_initial, X_xray_initial, X_volumetric_initial)
        
        y_reg_to_scale = y_regression_initial.copy().reshape(-1,1)
        if self.use_log_transform_regression:
             y_reg_to_scale = np.log1p(np.maximum(0, y_reg_to_scale))
        y_regression_initial_scaled = self.scaler_regression_target.transform(y_reg_to_scale)

        (X_img_train, X_img_test, X_tab_train, X_tab_test, X_gw_train, X_gw_test, X_xray_train, X_xray_test, 
         X_vol_train, X_vol_test, y_train_cls_labels, y_test_cls_labels, 
         y_train_reg_scaled, y_test_reg_scaled) = train_test_split(
            img_feats, tab_feats, gw_feats, xray_feats, vol_feats, y_labels_initial, y_regression_initial_scaled,
            test_size=0.2, random_state=GlobalConfig.RANDOM_SEED, stratify=y_labels_initial)

        y_train_cls_cat = keras.utils.to_categorical(y_train_cls_labels, num_classes=self.num_classes)
        y_test_cls_cat = keras.utils.to_categorical(y_test_cls_labels, num_classes=self.num_classes)
        
        train_inputs_dict = {
            self.input_names[0]: X_img_train, self.input_names[1]: X_tab_train,
            self.input_names[2]: X_gw_train, self.input_names[3]: X_xray_train,
            self.input_names[4]: X_vol_train
        }
        val_inputs_dict = {
            self.input_names[0]: X_img_test, self.input_names[1]: X_tab_test,
            self.input_names[2]: X_gw_test, self.input_names[3]: X_xray_test,
            self.input_names[4]: X_vol_test
        }
        train_outputs_dict = {self.output_names[0]: y_train_cls_cat, self.output_names[1]: y_train_reg_scaled}
        val_outputs_dict = {self.output_names[0]: y_test_cls_cat, self.output_names[1]: y_test_reg_scaled}
        
        callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)]
        if wandb_run:
            callbacks.append(WandbMetricsLogger(log_freq="epoch"))
            if GlobalConfig.WANDB_LOG_MODELS:
                ckpt_path = os.path.join(GlobalConfig.MODELS_DIR, f"fusion_model_initial_best_{wandb_run.id}.keras")
                callbacks.append(ModelCheckpoint(filepath=ckpt_path, monitor='val_loss', mode='min', save_best_only=True))
        
        print("Starting model fitting...")
        history = self.fusion_model.fit(
            train_inputs_dict, train_outputs_dict,
            epochs=epochs,
            batch_size=batch_size_per_replica * self.strategy.num_replicas_in_sync,
            validation_data=(val_inputs_dict, val_outputs_dict),
            callbacks=callbacks
        )
        print("Model fitting finished.")
        if history and hasattr(history, 'history') and history.history:
            plot_history(history, os.path.join(GlobalConfig.RESULTS_DIR, 'multimodal_fusion_initial_history.png'))
        print("Evaluating fusion model on validation set...")
        eval_res = self.fusion_model.evaluate(val_inputs_dict, val_outputs_dict, verbose=1, return_dict=True)
        print("Fusion Model Validation Metrics:", eval_res)
        if wandb_run:
            for metric_name, metric_value in eval_res.items(): wandb_run.summary[f"fusion_val_{metric_name}"] = metric_value
        y_pred_outputs_list = self.fusion_model.predict(val_inputs_dict, batch_size=batch_size_per_replica * self.strategy.num_replicas_in_sync, verbose=0)
        y_pred_cls_probs = y_pred_outputs_list[0]; y_pred_reg_output_tensor = y_pred_outputs_list[1]
        if self.use_probabilistic_uq: y_pred_reg_scaled_mean = y_pred_reg_output_tensor[:, 0:1]
        else: y_pred_reg_scaled_mean = y_pred_reg_output_tensor
        y_pred_cls = np.argmax(y_pred_cls_probs, axis=1)
        target_names_report = [GlobalConfig.LABELS_MAP.get(i,f'C{i}') for i in range(self.num_classes)]
        print("\nClassification Report:\n", classification_report(y_test_cls_labels, y_pred_cls, target_names=target_names_report, labels=np.arange(self.num_classes), zero_division=0))
        plot_confusion_matrix(confusion_matrix(y_test_cls_labels, y_pred_cls, labels=np.arange(self.num_classes)), classes=target_names_report, filename=os.path.join(GlobalConfig.RESULTS_DIR,'multimodal_fusion_initial_cm.png'))
        y_pred_reg_orig = self._safe_inverse_transform_regression(y_pred_reg_scaled_mean); y_test_reg_orig = self._safe_inverse_transform_regression(y_test_reg_scaled.reshape(-1,1))
        rmse = np.sqrt(mean_squared_error(y_test_reg_orig, y_pred_reg_orig)); r2 = r2_score(y_test_reg_orig, y_pred_reg_orig)
        print(f"\nRegression Report: RMSE: {rmse:.4f}, R2: {r2:.4f}")
        if wandb_run: wandb_run.summary.update({"fusion_val_rmse_reg_on_mean": rmse, "fusion_val_r2_reg_on_mean": r2})
        self.fusion_model.save(self.fusion_model_path)
        print(f"Initial Multimodal Fusion Model saved to {self.fusion_model_path}")

    def _safe_inverse_transform_regression(self, y_scaled_values, is_test_data=False, max_exp_input_log_domain=35.0):
        if not hasattr(self, 'scaler_regression_target') or not hasattr(self.scaler_regression_target, 'scale_') or self.scaler_regression_target.scale_ is None:
            return y_scaled_values.flatten()
        y_unscaled_from_scaler = self.scaler_regression_target.inverse_transform(y_scaled_values.reshape(-1, 1))
        if self.use_log_transform_regression:
            clipped_values = np.clip(y_unscaled_from_scaler, None, max_exp_input_log_domain)
            y_orig_scale = np.expm1(clipped_values)
        else:
            y_orig_scale = y_unscaled_from_scaler
        if not np.all(np.isfinite(y_orig_scale)):
             min_val = np.min(y_orig_scale[np.isfinite(y_orig_scale)]) if np.any(np.isfinite(y_orig_scale)) else 0.0
             max_val = np.max(y_orig_scale[np.isfinite(y_orig_scale)]) if np.any(np.isfinite(y_orig_scale)) else min_val + 1.0
             y_orig_scale = np.nan_to_num(y_orig_scale, nan=min_val, posinf=max_val, neginf=min_val)
        return y_orig_scale.flatten()

    def predict_batch(self, images, tabular, gw, xray, volumetric, num_mc_samples=10):
        if self.fusion_model is None: self._get_or_create_fusion_model()
        num_s = images.shape[0] if images is not None else 0
        if num_s == 0: return None, None, None, None, None
        
        img_f, tab_f, gw_f, xray_f, vol_f = self._extract_features(images, tabular, gw, xray, volumetric, batch_size_pred=num_s)
        if any(f is None or f.shape[0] != num_s for f in [img_f, tab_f, gw_f, xray_f, vol_f]):
            return None, None, None, None, None
            
        inputs_dict_for_predict = {
            self.input_names[0]: img_f, self.input_names[1]: tab_f, self.input_names[2]: gw_f,
            self.input_names[3]: xray_f, self.input_names[4]: vol_f
        }
        
        mean_cls_probs_out, mean_reg_pred_scaled_out, epistemic_cls_uncertainty_out = None, None, None
        aleatoric_reg_uncertainty_out, epistemic_reg_uncertainty_out = np.zeros(num_s), np.zeros(num_s)
        if self.use_probabilistic_uq:
            cls_mc_preds, reg_stacked_mc_outputs = [], []
            for _ in range(num_mc_samples):
                model_outputs_mc_list = self.fusion_model(inputs_dict_for_predict, training=True)
                cls_mc_preds.append(model_outputs_mc_list[0].numpy()); reg_stacked_mc_outputs.append(model_outputs_mc_list[1].numpy())
            mean_cls_probs_out = np.mean(cls_mc_preds, axis=0)
            epistemic_cls_uncertainty_out = np.mean(np.var(cls_mc_preds, axis=0), axis=1)
            reg_stacked_mc_outputs_np = np.array(reg_stacked_mc_outputs)
            mean_reg_pred_scaled_out = np.mean(reg_stacked_mc_outputs_np[..., 0], axis=0).reshape(-1, 1)
            aleatoric_reg_uncertainty_out = np.sqrt(np.mean(reg_stacked_mc_outputs_np[..., 1], axis=0)).flatten()
            epistemic_reg_uncertainty_out = np.std(reg_stacked_mc_outputs_np[..., 0], axis=0).flatten()
        else:
            # This path would also need updating if used, but is currently not.
            pass
        return (mean_cls_probs_out, mean_reg_pred_scaled_out, epistemic_cls_uncertainty_out, aleatoric_reg_uncertainty_out, epistemic_reg_uncertainty_out)
        
    def build_anomaly_detector(self, input_dim):
        encoder_input = Input(shape=(input_dim,), name=f"ae_input_{input_dim}")
        encoded = Dense(512, activation='relu')(encoder_input); encoded = BatchNormalization()(encoded); encoded = Dropout(0.2)(encoded)
        encoded = Dense(256, activation='relu')(encoded); encoded = BatchNormalization()(encoded); encoded = Dropout(0.2)(encoded)
        encoded = Dense(128, activation='relu')(encoded); encoded = BatchNormalization()(encoded); encoded = Dropout(0.1)(encoded)
        bottleneck = Dense(64, activation='relu', name='bottleneck')(encoded)
        decoded = Dense(128, activation='relu')(bottleneck); decoded = BatchNormalization()(decoded); decoded = Dropout(0.1)(decoded)
        decoded = Dense(256, activation='relu')(decoded); decoded = BatchNormalization()(decoded); decoded = Dropout(0.2)(decoded)
        decoded = Dense(512, activation='relu')(decoded); decoded = BatchNormalization()(decoded); decoded = Dropout(0.2)(decoded)
        decoder_output = Dense(input_dim, activation='sigmoid')(decoded)
        autoencoder = Model(encoder_input, decoder_output, name='anomaly_detector_model')
        autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return autoencoder

    def train_anomaly_detector_model(self, X_feats_ae, wandb_run=None, epochs=50, batch_size=32):
        if X_feats_ae is None or X_feats_ae.shape[0]==0: return
        if not os.path.exists(GlobalConfig.MODELS_DIR): os.makedirs(GlobalConfig.MODELS_DIR)
        print(f"Train AE: {X_feats_ae.shape[0]} samples, feature_dim={X_feats_ae.shape[1]}...")
        self.scaler_anomaly_features.fit(X_feats_ae)
        X_s = self.scaler_anomaly_features.transform(X_feats_ae)
        self.anomaly_detector = self.build_anomaly_detector(X_s.shape[1])
        cb_ae=[keras.callbacks.EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True)]
        if wandb_run and WandbMetricsLogger is not None: cb_ae.append(WandbMetricsLogger(log_freq="epoch"))
        ae_batch_size = min(batch_size, X_s.shape[0]); ae_batch_size = max(1, ae_batch_size)
        self.anomaly_detector.fit(X_s,X_s,epochs=epochs,batch_size=ae_batch_size,verbose=1,validation_split=0.1,callbacks=cb_ae)
        rec=self.anomaly_detector.predict(X_s,verbose=0,batch_size=ae_batch_size); errs=np.mean(np.square(X_s-rec),axis=1)
        self.anomaly_threshold=np.percentile(errs,98); print(f"AE thresh: {self.anomaly_threshold:.6f}")
        ae_path=os.path.join(GlobalConfig.MODELS_DIR,'anomaly_detector.keras'); keras.saving.save_model(self.anomaly_detector,ae_path)
        print(f"AE saved: {ae_path}")

    def update_model_online(self, epochs=1, batch_size_online_per_replica=16, wandb_run=None, current_stream_batch=0):
        buf_len = len(self.replay_buffer.get('labels',[]));
        min_s_needed = batch_size_online_per_replica * self.strategy.num_replicas_in_sync if self.strategy.num_replicas_in_sync > 0 else 1
        if self.fusion_model is None or buf_len < min_s_needed: return
        print(f"Starting online update with {buf_len} samples from replay buffer...")
        r_data = {key: np.array(val_list) for key, val_list in self.replay_buffer.items() if val_list}
        if not r_data or 'labels' not in r_data or r_data['labels'].size == 0: return
        act_buf_len = r_data['labels'].shape[0]
        if act_buf_len == 0: return
        
        img_r, tab_r, gw_r, xray_r, vol_r = self._extract_features(r_data.get('images'), r_data.get('tabular'), r_data.get('gw'),r_data.get('xray'), r_data.get('volumetric'))
        if any(f is None or f.shape[0] != act_buf_len for f in [img_r, tab_r, gw_r, xray_r, vol_r]): return
        
        lbl_cat_r = keras.utils.to_categorical(r_data['labels'], self.num_classes)
        y_reg_online_raw = r_data['regression_targets']
        y_reg_online_to_scale = np.log1p(np.maximum(0, y_reg_online_raw.copy())) if self.use_log_transform_regression else y_reg_online_raw.copy()
        y_reg_online_to_scale = np.nan_to_num(y_reg_online_to_scale, nan=0.0, posinf=np.max(y_reg_online_to_scale[np.isfinite(y_reg_online_to_scale)]) if np.isfinite(y_reg_online_to_scale).any() else 1.0, neginf=0.0)
        reg_s_r = self.scaler_regression_target.transform(y_reg_online_to_scale.reshape(-1,1)) if hasattr(self.scaler_regression_target, 'scale_') else y_reg_online_to_scale.reshape(-1,1)

        online_inputs = {
            self.input_names[0]: img_r, self.input_names[1]: tab_r, self.input_names[2]: gw_r,
            self.input_names[3]: xray_r, self.input_names[4]: vol_r
        }
        online_outputs = {self.output_names[0]: lbl_cat_r, self.output_names[1]: reg_s_r}

        with self.strategy.scope():
            if hasattr(self.fusion_model.optimizer, 'learning_rate'):
                try:
                    lr_var = self.fusion_model.optimizer.learning_rate
                    if isinstance(lr_var, tf.Variable): lr_var.assign(max(lr_var.numpy() * 0.9, 1e-7))
                except Exception as e: print(f"Could not modify LR: {e}")
            callbacks = [WandbMetricsLogger(log_freq="epoch")] if wandb_run and WandbMetricsLogger else []
            self.fusion_model.fit(online_inputs, online_outputs, epochs=epochs, batch_size=batch_size_online_per_replica*self.strategy.num_replicas_in_sync, verbose=0, callbacks=callbacks)
            self.fusion_model.save(self.fusion_model_path)
            print(f"Online model update completed. Model saved.")

    def detect_anomalies(self, feats_batch):
        s=self.scaler_anomaly_features
        if self.anomaly_detector is None or not hasattr(s,'scale_') or s.scale_ is None: return np.zeros(len(feats_batch) if feats_batch is not None else 0,dtype=bool), np.zeros(len(feats_batch) if feats_batch is not None else 0,dtype=float)
        if feats_batch is None or feats_batch.shape[0]==0: return np.array([],dtype=bool),np.array([],dtype=float)
        if feats_batch.shape[1]!=s.n_features_in_: return np.zeros(feats_batch.shape[0],dtype=bool), np.zeros(feats_batch.shape[0],dtype=float)
        X_s_batch=s.transform(feats_batch)
        reconstructions_batch=self.anomaly_detector.predict(X_s_batch,verbose=0,batch_size=max(1, min(32, feats_batch.shape[0])))
        errors_batch=np.mean(np.square(X_s_batch-reconstructions_batch),axis=1)
        return errors_batch > self.anomaly_threshold, errors_batch

    def select_uncertain_samples(self, batch_data, cls_probs, unc_scores_cls, sel_ratio_override=None):
        if cls_probs is None or cls_probs.shape[0]==0: return {},[]
        num_samples_in_batch = cls_probs.shape[0]; selection_ratio = sel_ratio_override if sel_ratio_override is not None else self.active_learning_selection_ratio
        if unc_scores_cls is None or len(unc_scores_cls) != num_samples_in_batch: uncertainty_scores_to_sort = 1.0 - np.max(cls_probs, axis=1)
        else: uncertainty_scores_to_sort = unc_scores_cls
        num_to_select = max(0, min(int(num_samples_in_batch * selection_ratio), num_samples_in_batch))
        if num_to_select == 0: return {}, []
        selected_indices_in_batch = np.argsort(uncertainty_scores_to_sort)[-num_to_select:]
        if len(selected_indices_in_batch) == 0: return {}, []
        selected_data_dict = {}
        for key_rb in self.replay_buffer.keys():
            if key_rb in batch_data and batch_data[key_rb] is not None:
                data_array_from_batch = np.array(batch_data[key_rb])
                if data_array_from_batch.ndim > 0 and data_array_from_batch.shape[0] == num_samples_in_batch: selected_data_dict[key_rb] = data_array_from_batch[selected_indices_in_batch]
                elif data_array_from_batch.ndim == 0 and num_samples_in_batch == 1 and len(selected_indices_in_batch) > 0 and selected_indices_in_batch[0] == 0: selected_data_dict[key_rb] = np.array([data_array_from_batch.item()])
                else: selected_data_dict[key_rb] = np.array([])
            else: selected_data_dict[key_rb] = np.array([])
        if 'labels' not in selected_data_dict or len(selected_data_dict['labels']) == 0: return {}, []
        return selected_data_dict, list(selected_indices_in_batch)

    def add_to_replay_buffer(self, data_dict_for_replay):
        if not data_dict_for_replay or 'labels' not in data_dict_for_replay or len(data_dict_for_replay['labels'])==0: return
        num_added_samples = len(data_dict_for_replay['labels'])
        if num_added_samples == 0: return
        for key_rb in self.replay_buffer.keys():
            if key_rb in data_dict_for_replay and data_dict_for_replay[key_rb] is not None:
                data_to_add = data_dict_for_replay[key_rb]
                if hasattr(data_to_add, '__len__') and len(data_to_add) == num_added_samples: self.replay_buffer[key_rb].extend(list(data_to_add))
        current_buffer_length = len(self.replay_buffer['labels'])
        if current_buffer_length > self.max_buffer_size:
            num_to_remove = current_buffer_length - self.max_buffer_size
            for key_rb_trim in self.replay_buffer.keys(): self.replay_buffer[key_rb_trim] = self.replay_buffer[key_rb_trim][num_to_remove:]

    def generate_recommendations(self, cls_p_batch, reg_p_orig_b, cls_u_b, ale_reg_u_b, epi_reg_u_b, anom_f_b, b_data_d):
        recommendations_list = []
        if cls_p_batch is None or cls_p_batch.shape[0] == 0: return recommendations_list
        num_samples_rec = cls_p_batch.shape[0]
        for i in range(num_samples_rec):
            class_probs_sample=cls_p_batch[i]; predicted_class_index=np.argmax(class_probs_sample); predicted_class_label=GlobalConfig.LABELS_MAP.get(predicted_class_index,f"Class_{predicted_class_index}"); confidence=class_probs_sample[predicted_class_index]
            reg_pred_original_sample = 0.0
            if reg_p_orig_b is not None and i < len(reg_p_orig_b):
                 current_reg_val = reg_p_orig_b[i,0] if reg_p_orig_b.ndim == 2 and reg_p_orig_b.shape[1] > 0 else reg_p_orig_b[i]
                 if isinstance(current_reg_val, (np.ndarray, list)):
                     reg_pred_original_sample = current_reg_val[0] if len(current_reg_val) > 0 else 0.0
                 else:
                     reg_pred_original_sample = current_reg_val
            cls_uncertainty_sample=cls_u_b[i] if cls_u_b is not None and i < len(cls_u_b) else 0.0
            aleatoric_reg_unc_sample=ale_reg_u_b[i] if ale_reg_u_b is not None and i < len(ale_reg_u_b) else 0.0
            epistemic_reg_unc_sample=epi_reg_u_b[i] if epi_reg_u_b is not None and i < len(epi_reg_u_b) else 0.0
            total_reg_unc_sample = np.sqrt(aleatoric_reg_unc_sample**2 + epistemic_reg_unc_sample**2) if aleatoric_reg_unc_sample is not None and epistemic_reg_unc_sample is not None else (aleatoric_reg_unc_sample or epistemic_reg_unc_sample or 0.0)
            if isinstance(total_reg_unc_sample, np.ndarray): total_reg_unc_sample = total_reg_unc_sample.item()
            is_anomaly_sample = anom_f_b[i] if anom_f_b is not None and i < len(anom_f_b) else False
            tabular_sample_data = b_data_d.get('tabular'); params_string = "N/A"
            if tabular_sample_data is not None and i < len(tabular_sample_data) and hasattr(tabular_sample_data[i], '__len__') and len(tabular_sample_data[i]) >= 2: params_string = ",".join([f"{p_val:.1f}" for p_val in tabular_sample_data[i][:2]])
            risk_flags = []
            if reg_pred_original_sample > GlobalConfig.REGRESSION_RISK_THRESHOLD_HIGH_ENERGY: risk_flags.append(f"HighEnergyRisk({reg_pred_original_sample:.1f})")
            if predicted_class_label in GlobalConfig.CLASSIFICATION_RISK_LABELS: risk_flags.append(f"CriticalClass:{predicted_class_label}")
            if cls_uncertainty_sample > GlobalConfig.CLASSIFICATION_UNCERTAINTY_THRESHOLD: risk_flags.append(f"HighClsUncertainty({cls_uncertainty_sample:.2f})")
            if total_reg_unc_sample > (GlobalConfig.REGRESSION_UNCERTAINTY_RELATIVE_FACTOR * abs(reg_pred_original_sample) + GlobalConfig.REGRESSION_UNCERTAINTY_ABSOLUTE_OFFSET): risk_flags.append(f"HighRegUncertainty({total_reg_unc_sample:.2f})")
            if is_anomaly_sample: risk_flags.append("AnomalyDetected")
            if risk_flags: recommendations_list.append(f"Sample {i}: PredClass={predicted_class_label}(Conf={confidence:.2f}), PredReg={reg_pred_original_sample:.1f}. RISKS: {', '.join(risk_flags)}. Params: {params_string}. ACTION: Review Recommended.")
        return recommendations_list
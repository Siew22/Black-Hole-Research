# main.py (最终优化版 - 包含内存清理)

import matplotlib
matplotlib.use('Agg')
import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import wandb
import random # For seeding
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
from keras.optimizers import Adam

# Keras 3.x 推荐的后端和配置导入方式
from tensorflow import keras
from keras import backend as K

# 启用这个配置以兼容使用自定义层的旧模型保存格式
keras.config.enable_unsafe_deserialization()

# Import configuration
from config import GlobalConfig
# *** FIX: ADDED IMPORTS FOR FINAL EVALUATION ***
from sklearn.metrics import classification_report, mean_squared_error, r2_score, accuracy_score, f1_score

# Import your modules
from src.ml_experiment import run_ml_experiment
from src.cnn_image_extractor import run_cnn_image_extractor
from src.timeseries_extractor import run_timeseries_extractor
from src.volumetric_extractor import run_volumetric_extractor
from src.multimodal_fusion_system import MultimodalFusionSystem, build_tabular_feature_extractor, build_flat_fusion_model, create_weighted_categorical_crossentropy, custom_gaussian_nll_loss_stacked
from src.nlp_experiment import run_nlp_experiment
from src.xai_analysis import run_xai_analysis
from src.generative_model_demo import (
    build_multimodal_generator, build_multimodal_discriminator, train_gan_multimodal,
    DEFAULT_LR_G, DEFAULT_LR_D, DEFAULT_GP_LAMBDA, DEFAULT_N_CRITIC,
    DEFAULT_PHYSICS_LOSS_WEIGHT, DEFAULT_CONSISTENCY_LOSS_WEIGHT
)
from src.hyperparameter_tuning import run_hyperparameter_tuning
from src.sim_training_experiment import run_sim_training_experiment
from src.utils import (
    generate_paired_multimodal_bh_data, generate_simulated_data_stream,
    plot_sample_multimodal_event, plot_3d_data_interactive, plot_uncertainty_selection,
    generate_synthetic_physics_sim_data
)

# Attempt to import W&B Keras integration callbacks
try:
    from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint
    WANDB_KERAS_INTEGRATION_AVAILABLE = True
except ImportError:
    WandbMetricsLogger = None
    WandbModelCheckpoint = None
    WANDB_KERAS_INTEGRATION_AVAILABLE = False
    print("Warning: wandb.integration.keras callbacks not found. W&B Keras logging might be limited.")


# --- 内存清理辅助函数 ---
def clear_memory_and_session():
    """
    一个可复用的内存和显存清理函数。
    在大型任务切换时调用，以释放资源。
    """
    print("\n--- Clearing memory and Keras session for the next stage ---")
    import gc # 导入垃圾回收模块
    gc.collect() # 运行 Python 垃圾回收
    K.clear_session() # 清理 Keras 后端会话，这是释放 GPU 显存的关键！
    print("--- Memory and session cleared. ---\n")


# Placeholder for AccumulativeModel (code from previous response)
class AccumulativeModel(tf.keras.Model):
    def __init__(self, model, accum_steps=4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.accum_steps = tf.constant(accum_steps, dtype=tf.int32)
        self.accum_gradient = None
        self.current_step = tf.Variable(0, dtype=tf.int32, trainable=False)

    def _initialize_gradients(self):
        if self.accum_gradient is None and self.model.trainable_variables:
            self.accum_gradient = [tf.Variable(tf.zeros_like(tv), trainable=False) for tv in self.model.trainable_variables]

    def compile(self, optimizer, loss=None, metrics=None, **kwargs):
        super().compile(optimizer=optimizer, loss=loss, metrics=metrics, **kwargs)
        self.acc_optimizer = optimizer

    @tf.function
    def train_step(self, data):
        self._initialize_gradients()
        self.current_step.assign_add(1)
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self.model(x, training=True)
            loss = self.model.compiled_loss(y, y_pred, regularization_losses=self.model.losses)

        gradients = tape.gradient(loss, self.model.trainable_variables)

        for i in range(len(self.accum_gradient)):
            if gradients[i] is not None:
                self.accum_gradient[i].assign_add(gradients[i])

        tf.cond(tf.equal(self.current_step % self.accum_steps, 0),
                self._apply_and_reset_gradients,
                lambda: tf.constant(False))

        self.model.compiled_metrics.update_state(y, y_pred)
        metrics_results = {m.name: m.result() for m in self.model.metrics}
        metrics_results["loss"] = loss
        return metrics_results


    def _apply_and_reset_gradients(self):
        self.acc_optimizer.apply_gradients(zip(self.accum_gradient, self.model.trainable_variables))
        for grad_var in self.accum_gradient:
            grad_var.assign(tf.zeros_like(grad_var))
        return tf.constant(True)

    def call(self, inputs, training=False):
        return self.model(inputs, training=training)

    def save_weights(self, filepath, overwrite=True, save_format=None, options=None):
        self.model.save_weights(filepath, overwrite, save_format, options)

    def load_weights(self, filepath, by_name=False, skip_mismatch=False, options=None):
        self.model.load_weights(filepath, by_name, skip_mismatch, options)

    def save(self, filepath, overwrite=True, include_optimizer=True, save_format=None, signatures=None, options=None, save_traces=True):
        self.model.save(filepath, overwrite, include_optimizer, save_format, signatures, options, save_traces)


def main():
    # --- STAGE 0: SETUP ---
    np.random.seed(GlobalConfig.RANDOM_SEED)
    tf.random.set_seed(GlobalConfig.RANDOM_SEED)
    random.seed(GlobalConfig.RANDOM_SEED)
    os.environ['PYTHONHASHSEED'] = str(GlobalConfig.RANDOM_SEED)

    for dir_name in [GlobalConfig.DATA_DIR, GlobalConfig.MODELS_DIR, GlobalConfig.RESULTS_DIR,
                     GlobalConfig.RESEARCH_PAPERS_DIR, GlobalConfig.KERAS_TUNER_DIR]:
        os.makedirs(dir_name, exist_ok=True)
    print("--- Starting AI Research Framework ---")

    strategy = tf.distribute.get_strategy()
    gpu_devices = tf.config.list_physical_devices('GPU')
    if gpu_devices:
        print(f"TensorFlow detected GPU(s): {gpu_devices}")
        try:
            for gpu in gpu_devices: tf.config.experimental.set_memory_growth(gpu, True)
            if len(gpu_devices) > 1:
                strategy = tf.distribute.MirroredStrategy()
                print(f"Using MirroredStrategy with {strategy.num_replicas_in_sync} replicas.")
        except RuntimeError as e: print(f"GPU setup error: {e}")
    else:
        print("TensorFlow did NOT detect GPU. Running on CPU.")

    if gpu_devices:
        try:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            print("Mixed precision 'mixed_float16' enabled globally.")
        except Exception as e:
            print(f"Could not enable mixed precision: {e}")

    config_for_wandb = {attr: getattr(GlobalConfig, attr) for attr in dir(GlobalConfig) if not callable(getattr(GlobalConfig, attr)) and not attr.startswith("__")}
    config_for_wandb["num_gpus"] = len(gpu_devices)
    config_for_wandb["distribute_strategy"] = strategy.__class__.__name__

    # --- STAGE 1: DATA GENERATION & PREPARATION ---
    img_size = GlobalConfig.IMG_SIZE; gw_timesteps = GlobalConfig.GW_TIMESTEPS
    x_ray_timesteps = GlobalConfig.X_RAY_TIMESTEPS; volumetric_size = GlobalConfig.VOLUMETRIC_SIZE
    num_initial_samples = GlobalConfig.NUM_INITIAL_SAMPLES

    X_images_initial, X_tabular_initial, X_gw_initial, X_xray_initial, X_volumetric_initial, y_labels_initial, y_regression_initial = \
        generate_paired_multimodal_bh_data(
            n_samples=num_initial_samples, img_size=img_size, gw_timesteps=gw_timesteps,
            x_ray_timesteps=x_ray_timesteps, volumetric_size=volumetric_size
        )
    if X_images_initial.shape[0] == 0:
        print("No initial data generated. Exiting.")
        return
    num_classes_initial = len(np.unique(y_labels_initial))
    GlobalConfig.NUM_CLASSES = num_classes_initial

    plot_sample_multimodal_event(
        X_images_initial[0], X_tabular_initial[0], X_gw_initial[0],
        X_xray_initial[0], X_volumetric_initial[0], y_labels_initial[0],
        regression_target=y_regression_initial[0],
        filename=os.path.join(GlobalConfig.RESULTS_DIR, 'sample_multimodal_event_initial.png')
    )
    print(f"Initial sample multimodal event plot saved to: {os.path.join(GlobalConfig.RESULTS_DIR, 'sample_multimodal_event_initial.png')}")


    # --- (OPTIONAL) STAGE 1.5: GAN DEMO ---
    run_gan_demo = True
    generator_model_for_final_plot = None
    if run_gan_demo:
        wandb_gan_run = wandb.init(project=GlobalConfig.WANDB_PROJECT, entity=GlobalConfig.WANDB_ENTITY,
                                   job_type="gan_training_multimodal", config=config_for_wandb,
                                   name="GAN_Multimodal_Physics_Consistency_Run", reinit=True)

        latent_dim = GlobalConfig.GAN_LATENT_DIM
        img_shape_gan = (img_size, img_size, 3)
        volumetric_shape_gan = (volumetric_size, volumetric_size, volumetric_size)
        tabular_input_dim_gan = X_tabular_initial.shape[1]

        with strategy.scope():
            generator = build_multimodal_generator(
                latent_dim, img_shape_gan, tabular_input_dim_gan,
                gw_timesteps, x_ray_timesteps, volumetric_shape_gan
            )
            discriminator = build_multimodal_discriminator(
                img_shape_gan, tabular_input_dim_gan,
                gw_timesteps, x_ray_timesteps, volumetric_shape_gan, is_wgan_gp=True
            )
        generator_model_for_final_plot = generator

        scaler_tabular_gan = MinMaxScaler().fit(X_tabular_initial)
        X_tabular_initial_gan_scaled = scaler_tabular_gan.transform(X_tabular_initial)
        all_real_data_tuple_for_gan = (X_images_initial, X_gw_initial, X_xray_initial, X_volumetric_initial)

        gan_lr_g = getattr(wandb_gan_run.config, "GAN_LEARNING_RATE_G", DEFAULT_LR_G)
        gan_lr_d = getattr(wandb_gan_run.config, "GAN_LEARNING_RATE_D", DEFAULT_LR_D)
        gan_gp_lambda = getattr(wandb_gan_run.config, "GAN_GP_LAMBDA", DEFAULT_GP_LAMBDA)
        gan_n_critic = getattr(wandb_gan_run.config, "GAN_N_CRITIC", DEFAULT_N_CRITIC)
        gan_batch_size = getattr(wandb_gan_run.config, "GAN_BATCH_SIZE_SWEEP", GlobalConfig.GAN_BATCH_SIZE)
        physics_loss_weight = getattr(wandb_gan_run.config, "GAN_PHYSICS_LOSS_WEIGHT", DEFAULT_PHYSICS_LOSS_WEIGHT)
        consistency_loss_weight = getattr(wandb_gan_run.config, "GAN_CONSISTENCY_LOSS_WEIGHT", DEFAULT_CONSISTENCY_LOSS_WEIGHT)

        gan_batch_size_per_replica = gan_batch_size // strategy.num_replicas_in_sync if strategy.num_replicas_in_sync > 0 else gan_batch_size
        gan_batch_size_per_replica = max(1, gan_batch_size_per_replica)

        train_gan_multimodal(
            generator, discriminator, all_real_data_tuple_for_gan, X_tabular_initial_gan_scaled,
            latent_dim, epochs=GlobalConfig.GAN_EPOCHS, batch_size=gan_batch_size_per_replica,
            wandb_run=wandb_gan_run,
            lr_g_sweep=gan_lr_g, lr_d_sweep=gan_lr_d, gp_lambda_sweep=gan_gp_lambda, n_critic_sweep=gan_n_critic,
            physics_loss_weight_sweep=physics_loss_weight,
            consistency_loss_weight_sweep=consistency_loss_weight
        )
        wandb_gan_run.finish()

    # ===> 清理点 #1: 在 GAN (如果运行了) 和 特征提取器之间
    clear_memory_and_session()


    # --- STAGE 2: FEATURE EXTRACTOR TRAINING ---
    wandb_extractor_run = wandb.init(project=GlobalConfig.WANDB_PROJECT, entity=GlobalConfig.WANDB_ENTITY,
                                     job_type="feature_extractor_training", config=config_for_wandb,
                                     name="Feature_Extractor_Training_Run", reinit=True)
    
    print("\n--- Training Feature Extractors ---")
    run_cnn_image_extractor(X_images_initial, y_labels_initial, img_size=img_size, wandb_run=wandb_extractor_run)
    run_timeseries_extractor(X_gw_initial, y_labels_initial, modality_name="GW", wandb_run=wandb_extractor_run)
    run_timeseries_extractor(X_xray_initial, y_labels_initial, modality_name="XRAY", wandb_run=wandb_extractor_run)
    run_volumetric_extractor(X_volumetric_initial, y_labels_initial, wandb_run=wandb_extractor_run)

    print("\n--- Training Tabular Feature Extractor ---")
    with strategy.scope():
        tabular_feature_extractor_inst = build_tabular_feature_extractor(X_tabular_initial.shape[1])
        dummy_tabular_classifier = tf.keras.Sequential([
            tabular_feature_extractor_inst,
            tf.keras.layers.Dense(num_classes_initial, activation='softmax')
        ])
        dummy_tabular_classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    scaler_tab_feat_ext = StandardScaler().fit(X_tabular_initial)
    X_tabular_initial_scaled_fe = scaler_tab_feat_ext.transform(X_tabular_initial)

    tab_callbacks = [WandbMetricsLogger(log_freq="epoch")] if wandb_extractor_run and WANDB_KERAS_INTEGRATION_AVAILABLE else []
    
    tab_ext_batch_size = GlobalConfig.FEAT_EXTRACTOR_BATCH_SIZE
    dummy_tabular_classifier.fit(X_tabular_initial_scaled_fe, y_labels_initial,
                                 epochs=GlobalConfig.FEAT_EXTRACTOR_EPOCHS,
                                 batch_size=tab_ext_batch_size,
                                 verbose=1, callbacks=tab_callbacks)
    
    tabular_feature_extractor_inst.save(os.path.join(GlobalConfig.MODELS_DIR, 'tabular_feature_extractor.keras'))
    print("Tabular Feature Extractor trained and saved as .keras file.")
    if wandb_extractor_run: wandb_extractor_run.finish()

    # ===> 清理点 #2: 在特征提取器 和 融合模型之间
    try:
        del dummy_tabular_classifier
        del tabular_feature_extractor_inst
    except NameError:
        pass
    clear_memory_and_session()


    # --- STAGE 3: FUSION MODEL PREPARATION & TRAINING ---
    wandb_fusion_run = wandb.init(project=GlobalConfig.WANDB_PROJECT, entity=GlobalConfig.WANDB_ENTITY,
                                  job_type="fusion_model_pipeline", config=config_for_wandb,
                                  name="Fusion_Model_Run_UQ", reinit=True)
    
    # 3.1 (Optional) Hyperparameter Tuning
    print("\n--- Hyperparameter Tuning Stage ---")
    run_hp_tuning = True
    best_model_path_from_tuning = None
    if run_hp_tuning and X_images_initial.shape[0] > 0:
        best_model_path_from_tuning = run_hyperparameter_tuning(
            X_images_initial, X_tabular_initial, X_gw_initial, X_xray_initial, X_volumetric_initial,
            y_labels_initial, y_regression_initial,
            wandb_run=wandb_fusion_run
        )
    else:
        print("Skipping hyperparameter tuning.")

    # 3.2 Define Final Model Path and Instantiate System
    print("\n--- Multimodal Fusion System Preparation Stage ---")
    if best_model_path_from_tuning and os.path.exists(best_model_path_from_tuning):
        print(f"Using best model from Hyperparameter Tuning: {best_model_path_from_tuning}")
        fusion_model_path_to_use = best_model_path_from_tuning
    else:
        print("No tuned model found. Using default path for training/loading.")
        fusion_model_path_to_use = os.path.join(GlobalConfig.MODELS_DIR, 'multimodal_fusion_model_main.keras')

    # 3.3 Instantiate the System ONCE
    multimodal_system = MultimodalFusionSystem(
        num_classes=num_classes_initial,
        fusion_model_path=fusion_model_path_to_use,
        image_extractor_path=os.path.join(GlobalConfig.MODELS_DIR, 'image_feature_extractor.keras'),
        tabular_feature_extractor_model=os.path.join(GlobalConfig.MODELS_DIR, 'tabular_feature_extractor.keras'),
        gw_timeseries_extractor_path=os.path.join(GlobalConfig.MODELS_DIR, 'gw_feature_extractor.keras'),
        xray_timeseries_extractor_path=os.path.join(GlobalConfig.MODELS_DIR, 'xray_feature_extractor.keras'),
        volumetric_extractor_path=os.path.join(GlobalConfig.MODELS_DIR, 'volumetric_feature_extractor.keras'),
        use_probabilistic_uq=True,
        strategy=strategy,
        use_log_transform_regression=True
    )

    # 3.4 Train or Load the Fusion Model
    if not os.path.exists(multimodal_system.fusion_model_path):
        print(f"No fusion model found at {multimodal_system.fusion_model_path}. Starting initial training.")
        batch_size = GlobalConfig.INITIAL_TRAINING_BATCH_SIZE
        batch_size_per_replica = batch_size // strategy.num_replicas_in_sync if strategy.num_replicas_in_sync > 0 else batch_size
        
        multimodal_system.run_initial_fusion_training(
            X_images_initial, X_tabular_initial, X_gw_initial, X_xray_initial,
            X_volumetric_initial, y_labels_initial, y_regression_initial,
            epochs=GlobalConfig.INITIAL_TRAINING_EPOCHS,
            batch_size_per_replica=max(1, batch_size_per_replica),
            wandb_run=wandb_fusion_run
        )
    else:
        print(f"Fusion model found at {multimodal_system.fusion_model_path}. Loading...")
        multimodal_system._get_or_create_fusion_model()
        if not multimodal_system.is_scaler_fitted:
            multimodal_system._fit_scalers(X_tabular_initial, X_gw_initial, X_xray_initial, y_regression_initial)
            
    # 3.5 Train Anomaly Detector
    print("\n--- Anomaly Detector Training ---")
    normal_indices_ae = np.where((y_labels_initial == 0) | (y_labels_initial == 1))[0]
    if len(normal_indices_ae) > 0 and X_images_initial[normal_indices_ae].shape[0] > 0:
        img_f_ae, tab_f_ae, gw_f_ae, xray_f_ae, vol_f_ae = multimodal_system._extract_features(
            X_images_initial[normal_indices_ae], X_tabular_initial[normal_indices_ae],
            X_gw_initial[normal_indices_ae], X_xray_initial[normal_indices_ae],
            X_volumetric_initial[normal_indices_ae]
        )
        if all(f is not None for f in [img_f_ae, tab_f_ae, gw_f_ae, xray_f_ae, vol_f_ae]):
            ts_comb_ae = np.concatenate([gw_f_ae, xray_f_ae], axis=-1)
            fused_features_for_ae = np.concatenate([img_f_ae, tab_f_ae, ts_comb_ae, vol_f_ae], axis=-1)
            if fused_features_for_ae.shape[0] > 0:
                 multimodal_system.train_anomaly_detector_model(fused_features_for_ae, wandb_run=wandb_fusion_run)
            else:
                print("No fused features for AE training (possibly due to small normal_indices sample or feature extraction issue).")
        else:
            print("Could not extract all necessary features for AE training.")
    else:
        print("Not enough 'normal' samples or data to train anomaly detector.")

    # --- STAGE 4: ONLINE LEARNING & SIMULATION ---
    print("\n--- Simulating Real-time Stream and Online Learning ---")
    use_probabilistic_uq_fusion = True
    stream_samples_total_per_batch = GlobalConfig.SAMPLES_PER_BATCH_STREAM * (strategy.num_replicas_in_sync if strategy.num_replicas_in_sync > 0 else 1)
    
    stream_generator = generate_simulated_data_stream(
        num_batches=GlobalConfig.NUM_STREAM_BATCHES,
        samples_per_batch=max(1, stream_samples_total_per_batch),
        img_size=img_size, gw_timesteps=gw_timesteps,
        x_ray_timesteps=x_ray_timesteps, volumetric_size=volumetric_size
    )

    online_update_batch_size = GlobalConfig.ONLINE_LEARNING_BATCH_SIZE
    online_update_batch_size_per_replica = online_update_batch_size // strategy.num_replicas_in_sync if strategy.num_replicas_in_sync > 0 else online_update_batch_size

    for i, batch_data_full in enumerate(stream_generator):
        print(f"\n--- Processing Stream Batch {i+1}/{GlobalConfig.NUM_STREAM_BATCHES} ---")
        if batch_data_full['images'].shape[0] == 0:
            print("  Stream batch is empty, skipping.")
            continue

        pred_outputs = multimodal_system.predict_batch(
            batch_data_full['images'], batch_data_full['tabular'], batch_data_full['gw'],
            batch_data_full['xray'], batch_data_full['volumetric'],
            num_mc_samples=10
        )

        if pred_outputs[0] is None:
            print("  Prediction failed. Skipping batch.")
            continue
        mean_cls_probs, mean_reg_scaled, cls_unc, ale_reg_unc, epi_reg_unc = pred_outputs
        
        mean_reg_original = multimodal_system._safe_inverse_transform_regression(mean_reg_scaled)

        img_f_b, tab_f_b, gw_f_b, xray_f_b, vol_f_b = multimodal_system._extract_features(
             batch_data_full['images'], batch_data_full['tabular'], batch_data_full['gw'],
             batch_data_full['xray'], batch_data_full['volumetric']
        )
        is_anomaly_flags = np.zeros(batch_data_full['labels'].shape[0], dtype=bool)
        if all(f is not None for f in [img_f_b, tab_f_b, gw_f_b, xray_f_b, vol_f_b]):
            ts_comb_b = np.concatenate([gw_f_b, xray_f_b], axis=-1)
            fused_features_batch = np.concatenate([img_f_b, tab_f_b, ts_comb_b, vol_f_b], axis=-1)
            if fused_features_batch.shape[0] > 0 :
                 is_anomaly_flags, anomaly_errors = multimodal_system.detect_anomalies(fused_features_batch)
        else:
            print("  Skipping anomaly detection for batch due to missing features.")

        selected_data, selected_indices = multimodal_system.select_uncertain_samples(
            batch_data_full, mean_cls_probs, cls_unc
        )
        if selected_data and len(selected_data.get('labels', [])) > 0:
            multimodal_system.add_to_replay_buffer(selected_data)
            plot_uncertainty_selection(
                batch_data_full['images'], mean_cls_probs, selected_indices,
                filename=os.path.join(GlobalConfig.RESULTS_DIR, f'uncertainty_selection_batch_{i}.png')
            )

            multimodal_system.update_model_online(
                epochs=GlobalConfig.ONLINE_LEARNING_EPOCHS,
                batch_size_online_per_replica=max(1, online_update_batch_size_per_replica),
                wandb_run=wandb_fusion_run, current_stream_batch=i
            )

        recommendations = multimodal_system.generate_recommendations(
            mean_cls_probs, mean_reg_original.reshape(-1, 1), cls_unc,
            ale_reg_unc, epi_reg_unc,
            is_anomaly_flags, batch_data_full
        )
        if recommendations:
            print(f"  Recommendations for stream batch {i+1}:")
            for rec in recommendations[:3]: print(f"    - {rec.strip()}")

        if wandb_fusion_run:
            log_stream_metrics = {
                f"stream_batch_{i}/avg_cls_uncertainty": np.mean(cls_unc) if cls_unc is not None and len(cls_unc)>0 else 0,
                f"stream_batch_{i}/avg_aleatoric_reg_unc": np.mean(ale_reg_unc) if ale_reg_unc is not None and len(ale_reg_unc)>0 else 0,
                f"stream_batch_{i}/avg_epistemic_reg_unc": np.mean(epi_reg_unc) if epi_reg_unc is not None and len(epi_reg_unc)>0 else 0,
                f"stream_batch_{i}/num_anomalies": np.sum(is_anomaly_flags),
                f"stream_batch_{i}/num_selected_active_learning": len(selected_indices) if selected_indices is not None else 0
            }
            wandb_fusion_run.log(log_stream_metrics)
    
    if wandb_fusion_run: wandb_fusion_run.finish()

    # ===> 清理点 #3: 在融合模型 和 其他独立实验之间
    # XAI 和 Final Evaluation 仍然需要 multimodal_system，所以我们不清空所有东西，
    # 但清理会话仍然有助于释放一些资源。
    clear_memory_and_session()


    # --- STAGE 5: OTHER EXPERIMENTS & FINAL ANALYSIS ---
    wandb_other_exp_run = wandb.init(project=GlobalConfig.WANDB_PROJECT, entity=GlobalConfig.WANDB_ENTITY,
                                     job_type="other_experiments", config=config_for_wandb,
                                     name="Other_Experiments_Run", reinit=True)
    run_ml_experiment()
    run_nlp_experiment(wandb_run=wandb_other_exp_run)
    run_sim_training_experiment(wandb_run=wandb_other_exp_run)

    # 5.1 XAI Analysis
    print("\n--- XAI Analysis Stage ---")
    run_xai = True
    if run_xai and multimodal_system.fusion_model is not None:
        num_xai_samples = min(GlobalConfig.XAI_NUM_SAMPLES, X_images_initial.shape[0])
        run_xai_analysis(
            X_images_initial[:num_xai_samples],
            X_tabular_initial[:num_xai_samples],
            X_gw_initial[:num_xai_samples],
            X_xray_initial[:num_xai_samples],
            X_volumetric_initial[:num_xai_samples],
            y_labels_initial[:num_xai_samples],
            y_regression_initial[:num_xai_samples],
            multimodal_system,
            wandb_run=wandb_other_exp_run
        )
    else:
        print("Skipping XAI analysis as fusion model is not available.")
    
    # --- NEW: STAGE 5.2: BASELINE FLAT FUSION MODEL TRAINING & EVALUATION ---
    print("\n--- Baseline Flat Fusion Model Stage ---")
    
    # We will reuse the data and splits from the main experiment for a fair comparison
    # First, let's re-create the data splits to ensure they are available
    print("Preparing data for baseline model...")
    img_feats, tab_feats, gw_feats, xray_feats, vol_feats = multimodal_system._extract_features(
        X_images_initial, X_tabular_initial, X_gw_initial, X_xray_initial, X_volumetric_initial
    )
    y_reg_to_scale = y_regression_initial.copy().reshape(-1, 1)
    if multimodal_system.use_log_transform_regression:
        y_reg_to_scale = np.log1p(np.maximum(0, y_reg_to_scale))
    y_regression_initial_scaled = multimodal_system.scaler_regression_target.transform(y_reg_to_scale)

    (X_img_train, X_img_test, X_tab_train, X_tab_test, X_gw_train, X_gw_test, X_xray_train, X_xray_test, 
     X_vol_train, X_vol_test, y_train_cls_labels, y_test_cls_labels, 
     y_train_reg_scaled, y_test_reg_scaled) = train_test_split(
        img_feats, tab_feats, gw_feats, xray_feats, vol_feats, y_labels_initial, y_regression_initial_scaled,
        test_size=0.2, random_state=GlobalConfig.RANDOM_SEED, stratify=y_labels_initial
    )

    y_train_cls_cat = keras.utils.to_categorical(y_train_cls_labels, num_classes=num_classes_initial)
    y_test_cls_cat = keras.utils.to_categorical(y_test_cls_labels, num_classes=num_classes_initial)
    
    train_inputs_dict = {'image_features_input_flat': X_img_train, 'tabular_features_input_flat': X_tab_train,
                         'gw_features_input_flat': X_gw_train, 'xray_features_input_flat': X_xray_train,
                         'volumetric_features_input_flat': X_vol_train}
    val_inputs_dict = {'image_features_input_flat': X_img_test, 'tabular_features_input_flat': X_tab_test,
                       'gw_features_input_flat': X_gw_test, 'xray_features_input_flat': X_xray_test,
                       'volumetric_features_input_flat': X_vol_test}

    train_outputs_dict = {'classification_output': y_train_cls_cat, 'regression_output': y_train_reg_scaled}
    val_outputs_dict = {'classification_output': y_test_cls_cat, 'regression_output': y_test_reg_scaled}

    with strategy.scope():
        print("Building and compiling Flat Fusion baseline model...")
        flat_model = build_flat_fusion_model(
            img_feats.shape[1], tab_feats.shape[1], gw_feats.shape[1],
            xray_feats.shape[1], vol_feats.shape[1], num_classes_initial
        )
        # Use a reasonable learning rate and the same weighted loss
        class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_cls_labels), y=y_train_cls_labels)
        class_weights_dict = {i: w for i, w in enumerate(class_weights)}
        weighted_loss = create_weighted_categorical_crossentropy(class_weights_dict)
        
        flat_model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss={'classification_output': weighted_loss, 'regression_output': custom_gaussian_nll_loss_stacked},
            loss_weights={'classification_output': 1.0, 'regression_output': 0.05}
        )
    
    print("Training Flat Fusion baseline model...")
    flat_model.fit(
        train_inputs_dict, train_outputs_dict,
        epochs=GlobalConfig.INITIAL_TRAINING_EPOCHS,
        batch_size=GlobalConfig.INITIAL_TRAINING_BATCH_SIZE,
        validation_data=(val_inputs_dict, val_outputs_dict),
        callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
        verbose=1
    )

    print("\n--- Evaluating Flat Fusion Baseline ---")
    # Evaluate on the full test set (created in STAGE 6)
    # Note: Ensure STAGE 6 code runs before this, or redefine test sets here. For simplicity, we redefine.
    _, test_indices = train_test_split(np.arange(X_images_initial.shape[0]), test_size=0.15, random_state=GlobalConfig.RANDOM_SEED, stratify=y_labels_initial)
    
    img_feats_test, tab_feats_test, gw_feats_test, xray_feats_test, vol_feats_test = multimodal_system._extract_features(
        X_images_initial[test_indices], X_tabular_initial[test_indices], X_gw_initial[test_indices],
        X_xray_initial[test_indices], X_volumetric_initial[test_indices]
    )
    
    # 1. Full Test Set Evaluation
    print("\n--- on Full Test Set ---")
    flat_preds_full = flat_model.predict([img_feats_test, tab_feats_test, gw_feats_test, xray_feats_test, vol_feats_test])
    flat_pred_cls_full = np.argmax(flat_preds_full[0], axis=1)
    flat_accuracy_full = accuracy_score(y_labels_initial[test_indices], flat_pred_cls_full)
    print(f"Flat Fusion Accuracy (Full Test Set): {flat_accuracy_full:.4f}")

    # 2. Degraded Test Set Evaluation
    print("\n--- on Degraded Test Set (50% X-ray loss) ---")
    xray_feats_degraded = xray_feats_test.copy()
    num_to_degrade = len(xray_feats_degraded) // 2
    degrade_indices = np.random.choice(len(xray_feats_degraded), num_to_degrade, replace=False)
    xray_feats_degraded[degrade_indices] = 0
    
    flat_preds_degraded = flat_model.predict([img_feats_test, tab_feats_test, gw_feats_test, xray_feats_degraded, vol_feats_test])
    flat_pred_cls_degraded = np.argmax(flat_preds_degraded[0], axis=1)
    flat_accuracy_degraded = accuracy_score(y_labels_initial[test_indices], flat_pred_cls_degraded)
    print(f"Flat Fusion Accuracy (Degraded Test Set): {flat_accuracy_degraded:.4f}")

    if wandb_other_exp_run:
        wandb_other_exp_run.summary["baseline_accuracy_full"] = flat_accuracy_full
        wandb_other_exp_run.summary["baseline_accuracy_degraded"] = flat_accuracy_degraded
    
    # *** NEW: STAGE 6: FINAL MODEL EVALUATION ***
    print("\n--- Final Model Evaluation Stage ---")
    if multimodal_system.fusion_model:
        # We need a hold-out test set for final evaluation.
        # Let's create one from the initial data.
        # NOTE: In a real-world scenario, you would have a completely separate test dataset.
        from sklearn.model_selection import train_test_split
        
        # Split the initial data into a main set (for training/tuning) and a final test set
        indices = np.arange(X_images_initial.shape[0])
        # Use stratification to ensure test set has representative class distribution
        train_indices, test_indices = train_test_split(
            indices, test_size=0.15, random_state=GlobalConfig.RANDOM_SEED, stratify=y_labels_initial
        )

        # Prepare the test data
        X_img_test_final = X_images_initial[test_indices]
        X_tab_test_final = X_tabular_initial[test_indices]
        X_gw_test_final = X_gw_initial[test_indices]
        X_xray_test_final = X_xray_initial[test_indices]
        X_vol_test_final = X_volumetric_initial[test_indices]
        y_cls_test_final = y_labels_initial[test_indices]
        y_reg_test_final = y_regression_initial[test_indices]

        print(f"Evaluating final model on a hold-out test set of {len(y_cls_test_final)} samples.")

        # Get predictions using the system's predict_batch method
        final_preds = multimodal_system.predict_batch(
            X_img_test_final, X_tab_test_final, X_gw_test_final,
            X_xray_test_final, X_vol_test_final,
            num_mc_samples=30 # Use more samples for a stable uncertainty estimate
        )

        if final_preds and final_preds[0] is not None:
            mean_cls_probs, mean_reg_scaled, _, _, _ = final_preds
            
            # --- Classification Evaluation ---
            y_pred_cls = np.argmax(mean_cls_probs, axis=1)
            class_names_final = [GlobalConfig.LABELS_MAP.get(i, f'Class {i}') for i in range(multimodal_system.num_classes)]
            
            print("\n--- Final Classification Report ---")
            final_cls_report = classification_report(y_cls_test_final, y_pred_cls, target_names=class_names_final, zero_division=0)
            print(final_cls_report)

            # --- Regression Evaluation ---
            y_pred_reg_orig = multimodal_system._safe_inverse_transform_regression(mean_reg_scaled)
            
            print("\n--- Final Regression Report ---")
            final_rmse = np.sqrt(mean_squared_error(y_reg_test_final, y_pred_reg_orig))
            final_r2 = r2_score(y_reg_test_final, y_pred_reg_orig)
            print(f"Final Test Set RMSE: {final_rmse:.4f}")
            print(f"Final Test Set R2 Score: {final_r2:.4f}")

            # Log final metrics to W&B
            if wandb_other_exp_run:
                wandb_other_exp_run.summary["final_test_accuracy"] = accuracy_score(y_cls_test_final, y_pred_cls)
                wandb_other_exp_run.summary["final_test_f1_macro"] = f1_score(y_cls_test_final, y_pred_cls, average='macro')
                wandb_other_exp_run.summary["final_test_rmse"] = final_rmse
                wandb_other_exp_run.summary["final_test_r2"] = final_r2
        else:
            print("Could not get predictions for the final evaluation set.")
    else:
        print("Final model not available for evaluation.")
    
    if wandb_other_exp_run: wandb_other_exp_run.finish()

    # --- (OPTIONAL) STAGE 7: FINAL GAN PLOT ---
    if run_gan_demo and generator_model_for_final_plot is not None and X_tabular_initial.shape[0] > 0:
        print("\n--- Generating Final Multimodal Sample from Trained GAN ---")
        current_latent_dim = GlobalConfig.GAN_LATENT_DIM
        if 'X_tabular_initial_gan_scaled' not in locals() or X_tabular_initial_gan_scaled is None:
            if 'scaler_tabular_gan' not in locals() or scaler_tabular_gan is None:
                 scaler_tabular_gan_final = MinMaxScaler().fit(X_tabular_initial)
            else:
                 scaler_tabular_gan_final = scaler_tabular_gan
            X_tabular_for_final_gan_scaled = scaler_tabular_gan_final.transform(X_tabular_initial)
        else:
            X_tabular_for_final_gan_scaled = X_tabular_initial_gan_scaled

        if X_tabular_for_final_gan_scaled.shape[0] > 0:
            num_final_gan_samples_to_plot = 1
            final_noise = np.random.normal(0, 1, (num_final_gan_samples_to_plot, current_latent_dim))
            final_tab_cond_idx = np.random.choice(X_tabular_for_final_gan_scaled.shape[0])
            final_tab_cond_scaled_main = X_tabular_for_final_gan_scaled[final_tab_cond_idx:final_tab_cond_idx+1]
            final_tab_cond_original_main = X_tabular_initial[final_tab_cond_idx]
            final_gen_outputs_main = generator_model_for_final_plot.predict([final_noise, final_tab_cond_scaled_main], verbose=0)
            gen_img_final_main = (final_gen_outputs_main[0][0] * 127.5 + 127.5).astype(np.uint8)

            gw_data_orig_main = X_gw_initial
            gw_min_main, gw_max_main = np.min(gw_data_orig_main), np.max(gw_data_orig_main)
            gen_gw_from_gan = final_gen_outputs_main[1][0]
            if (gw_max_main - gw_min_main) > 1e-7:
                 gen_gw_final_unnorm_main = (gen_gw_from_gan + 1.0) / 2.0 * (gw_max_main - gw_min_main) + gw_min_main
            else:
                 gen_gw_final_unnorm_main = np.full_like(gen_gw_from_gan, gw_min_main)

            xray_data_orig_main = X_xray_initial
            xray_min_main_norm, xray_max_main_norm = np.min(xray_data_orig_main), np.max(xray_data_orig_main)
            gen_xray_from_gan = final_gen_outputs_main[2][0]
            if (xray_max_main_norm - xray_min_main_norm) > 1e-7:
                gen_xray_final_unnorm_main = gen_xray_from_gan * (xray_max_main_norm - xray_min_main_norm) + xray_min_main_norm
            else:
                gen_xray_final_unnorm_main = np.full_like(gen_xray_from_gan, xray_min_main_norm)

            gen_vol_final_unnorm_main = (final_gen_outputs_main[3][0] * 255.0)
            if gen_vol_final_unnorm_main.ndim == 4 and gen_vol_final_unnorm_main.shape[-1] == 1:
                gen_vol_final_unnorm_main_slice = gen_vol_final_unnorm_main[:,:,:,0]
            else:
                gen_vol_final_unnorm_main_slice = gen_vol_final_unnorm_main

            pseudo_label_main = GlobalConfig.LABELS_MAP.get(1, "Black Hole")
            pseudo_reg_target_main = np.random.uniform(5, 50)
            final_event_filename = os.path.join(GlobalConfig.RESULTS_DIR, 'sample_multimodal_event_gan_generated_END_OF_RUN.png')
            plot_sample_multimodal_event(
                gen_img_final_main, final_tab_cond_original_main, gen_gw_final_unnorm_main.flatten(),
                gen_xray_final_unnorm_main.flatten(), gen_vol_final_unnorm_main_slice, 1,
                regression_target=pseudo_reg_target_main, filename=final_event_filename
            )
            print(f"Final GAN generated multimodal sample event saved to {final_event_filename}")
            if wandb.run is not None and os.path.exists(final_event_filename): # Check if a wandb run is active
                wandb.log({"final_gan_multimodal_sample_event_end_of_run": wandb.Image(final_event_filename)})
        else:
            print("Skipping final GAN sample event generation: No tabular conditions available for GAN.")

    print("\n--- All experiments completed. Check 'results/' and 'models/' directories. ---")

if __name__ == "__main__":
    main()

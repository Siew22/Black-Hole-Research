# src/generative_model_demo.py
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Reshape, Flatten, Dropout, BatchNormalization, Concatenate,
    LeakyReLU, Conv2DTranspose, Conv2D, Conv1D, UpSampling1D,
    Conv3DTranspose, UpSampling3D, Layer, Conv3D
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
import matplotlib.pyplot as plt
from src.utils import plot_gan_generated_samples, plot_3d_data_slices
import wandb
from tqdm import tqdm

from config import GlobalConfig
from tensorflow.keras import regularizers

# Default hyperparameters (no changes)
DEFAULT_LR_G = GlobalConfig.GAN_LEARNING_RATE_G_DEFAULT
DEFAULT_LR_D = GlobalConfig.GAN_LEARNING_RATE_D_DEFAULT
DEFAULT_GP_LAMBDA = GlobalConfig.GAN_GP_LAMBDA_DEFAULT
DEFAULT_N_CRITIC = GlobalConfig.GAN_N_CRITIC_DEFAULT
DEFAULT_PHYSICS_LOSS_WEIGHT = GlobalConfig.GAN_PHYSICS_LOSS_WEIGHT_DEFAULT
DEFAULT_CONSISTENCY_LOSS_WEIGHT = GlobalConfig.GAN_CONSISTENCY_LOSS_WEIGHT_DEFAULT


# --- MODIFIED: build_multimodal_generator (V3 - Shape Fix) ---
def build_multimodal_generator(latent_dim, img_shape, tabular_cond_dim,
                               gw_timesteps, xray_timesteps, volumetric_shape):
    noise_input = Input(shape=(latent_dim,), name='generator_noise_input')
    tabular_condition_input = Input(shape=(tabular_cond_dim,), name='generator_tabular_condition_input')

    cond_dense = Dense(128)(tabular_condition_input)
    cond_dense = LeakyReLU(0.2)(cond_dense)
    merged_input = Concatenate()([noise_input, cond_dense])
    
    # --- 1. Image Generation Head (Unchanged) ---
    img_base = Dense(4 * 4 * 256, use_bias=False)(merged_input)
    img_base = BatchNormalization()(img_base)
    img_base = LeakyReLU(0.2)(img_base)
    img_base_reshaped = Reshape((4, 4, 256))(img_base)

    img_head = Conv2DTranspose(128, kernel_size=5, strides=2, padding='same', use_bias=False)(img_base_reshaped)
    img_head = BatchNormalization()(img_head)
    img_head = LeakyReLU(0.2)(img_head)
    
    img_head = Conv2DTranspose(64, kernel_size=5, strides=2, padding='same', use_bias=False)(img_head)
    img_head = BatchNormalization()(img_head)
    img_head = LeakyReLU(0.2)(img_head)

    img_head = Conv2DTranspose(32, kernel_size=5, strides=2, padding='same', use_bias=False)(img_head)
    img_head = BatchNormalization()(img_head)
    img_head = LeakyReLU(0.2)(img_head)
    
    # Assumes img_shape[0] is 64 (4 -> 8 -> 16 -> 32 -> 64)
    generated_image = Conv2DTranspose(img_shape[-1], kernel_size=5, strides=2, padding='same', activation='tanh', name='generated_image')(img_head)

    # --- 2. GW Time-series Generation Head ---
    gw_base = Dense((gw_timesteps // 4) * 32, use_bias=False)(merged_input)
    gw_base = BatchNormalization()(gw_base)
    gw_base = LeakyReLU(0.2)(gw_base)
    gw_base_reshaped = Reshape((gw_timesteps // 4, 32))(gw_base)
    
    gw_path = UpSampling1D(size=2)(gw_base_reshaped)
    gw_path = Conv1D(filters=16, kernel_size=5, padding='same', use_bias=False)(gw_path)
    gw_path = BatchNormalization()(gw_path)
    gw_path = LeakyReLU(0.2)(gw_path)
    
    gw_path = UpSampling1D(size=2)(gw_path)
    gw_path = Conv1D(filters=8, kernel_size=5, padding='same', use_bias=False)(gw_path)
    gw_path = BatchNormalization()(gw_path)
    gw_path = LeakyReLU(0.2)(gw_path)
    generated_gw = Conv1D(filters=1, kernel_size=7, padding='same', activation='tanh', name='generated_gw')(gw_path)

    # --- 3. X-ray Time-series Generation Head (FIXED) ---
    # Create a separate base for xray with the correct dimensions
    xray_initial_timesteps = xray_timesteps // 4 # e.g. 100 -> 25
    xray_base = Dense(xray_initial_timesteps * 32, use_bias=False)(merged_input)
    xray_base = BatchNormalization()(xray_base)
    xray_base = LeakyReLU(0.2)(xray_base)
    xray_base_reshaped = Reshape((xray_initial_timesteps, 32))(xray_base)

    xray_path = UpSampling1D(size=2)(xray_base_reshaped) # 25 -> 50
    xray_path = Conv1D(filters=16, kernel_size=5, padding='same', use_bias=False)(xray_path)
    xray_path = BatchNormalization()(xray_path)
    xray_path = LeakyReLU(0.2)(xray_path)
    
    xray_path = UpSampling1D(size=2)(xray_path) # 50 -> 100
    xray_path = Conv1D(filters=8, kernel_size=5, padding='same', use_bias=False)(xray_path)
    xray_path = BatchNormalization()(xray_path)
    xray_path = LeakyReLU(0.2)(xray_path)
    generated_xray = Conv1D(filters=1, kernel_size=7, padding='same', activation='sigmoid', name='generated_xray')(xray_path)

    # --- 4. 3D Volumetric Data Generation Head (Unchanged) ---
    vol_start_dim = volumetric_shape[0] // 4
    vol_base = Dense(vol_start_dim * vol_start_dim * vol_start_dim * 128, use_bias=False)(merged_input)
    vol_base = BatchNormalization()(vol_base)
    vol_base = LeakyReLU(0.2)(vol_base)
    vol_base_reshaped = Reshape((vol_start_dim, vol_start_dim, vol_start_dim, 128))(vol_base)

    vol_head = Conv3DTranspose(64, kernel_size=4, strides=2, padding='same', use_bias=False)(vol_base_reshaped)
    vol_head = BatchNormalization()(vol_head)
    vol_head = LeakyReLU(0.2)(vol_head)
    
    generated_volumetric = Conv3DTranspose(1, kernel_size=4, strides=2, padding='same', activation='sigmoid', name='generated_volumetric')(vol_head)

    model = Model(inputs=[noise_input, tabular_condition_input],
                  outputs=[generated_image, generated_gw, generated_xray, generated_volumetric],
                  name='multimodal_generator_v3_shapefix')
    return model


# The rest of the file (build_multimodal_discriminator and all training functions)
# remains exactly the same as in my previous V2 response.
def build_multimodal_discriminator(img_shape, tabular_cond_dim,
                                   gw_timesteps, xray_timesteps, volumetric_shape, is_wgan_gp=True):
    # This version uses Dropout and careful architecture instead of Spectral Norm
    
    img_input = Input(shape=img_shape, name='disc_img_input')
    gw_input = Input(shape=(gw_timesteps, 1), name='disc_gw_input')
    xray_input = Input(shape=(xray_timesteps, 1), name='disc_xray_input')
    
    volumetric_input_shape = list(volumetric_shape)
    if len(volumetric_input_shape) == 3: volumetric_input_shape.append(1)
    volumetric_input = Input(shape=tuple(volumetric_input_shape), name='disc_vol_input')
    
    tabular_condition_input = Input(shape=(tabular_cond_dim,), name='disc_tabular_condition_input')

    # Image path
    img_path = Conv2D(64, (5, 5), strides=(2, 2), padding='same')(img_input)
    img_path = LeakyReLU(negative_slope=0.2)(img_path)
    img_path = Dropout(0.3)(img_path)
    
    img_path = Conv2D(128, (5, 5), strides=(2, 2), padding='same')(img_path)
    img_path = LeakyReLU(negative_slope=0.2)(img_path)
    img_path = Dropout(0.3)(img_path)
    
    img_path = Conv2D(256, (5, 5), strides=(2, 2), padding='same')(img_path)
    img_path = LeakyReLU(negative_slope=0.2)(img_path)
    img_path = Dropout(0.3)(img_path)
    img_features = Flatten(name='disc_img_features')(img_path)

    # GW path
    gw_path = Conv1D(64, kernel_size=5, strides=2, padding="same")(gw_input)
    gw_path = LeakyReLU(negative_slope=0.2)(gw_path)
    gw_path = Dropout(0.2)(gw_path)
    gw_path = Conv1D(128, kernel_size=5, strides=2, padding="same")(gw_path)
    gw_path = LeakyReLU(negative_slope=0.2)(gw_path)
    gw_path = Dropout(0.2)(gw_path)
    gw_features = Flatten(name='disc_gw_features')(gw_path)

    # X-ray path
    xray_path = Conv1D(64, kernel_size=5, strides=2, padding="same")(xray_input)
    xray_path = LeakyReLU(negative_slope=0.2)(xray_path)
    xray_path = Dropout(0.2)(xray_path)
    xray_path = Conv1D(128, kernel_size=5, strides=2, padding="same")(xray_path)
    xray_path = LeakyReLU(negative_slope=0.2)(xray_path)
    xray_path = Dropout(0.2)(xray_path)
    xray_features = Flatten(name='disc_xray_features')(xray_path)

    # Volumetric path
    vol_path = Conv3D(32, (3,3,3), strides=(2,2,2), padding='same')(volumetric_input)
    vol_path = LeakyReLU(negative_slope=0.2)(vol_path)
    vol_path = Dropout(0.2)(vol_path)
    vol_path = Conv3D(64, (3,3,3), strides=(2,2,2), padding='same')(vol_path)
    vol_path = LeakyReLU(negative_slope=0.2)(vol_path)
    vol_path = Dropout(0.2)(vol_path)
    vol_features = Flatten(name='disc_vol_features')(vol_path)

    # Condition path
    cond_dense_disc = Dense(128)(tabular_condition_input)
    cond_dense_disc = LeakyReLU(negative_slope=0.2)(cond_dense_disc)

    merged_features = Concatenate()([img_features, gw_features, xray_features, vol_features, cond_dense_disc])
    
    x = Dense(512)(merged_features)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = Dropout(0.4)(x)
    
    validity = Dense(1, name='discriminator_output', dtype='float32')(x)

    model = Model(inputs=[img_input, gw_input, xray_input, volumetric_input, tabular_condition_input],
                  outputs=validity,
                  name='multimodal_discriminator_v3_no_addons')
    return model

@tf.function
def single_modality_gradient_penalty(discriminator, real_data, fake_data, disc_all_inputs_template, modality_idx, tabular_cond):
    batch_size = tf.shape(real_data)[0]
    data_dtype = real_data.dtype
    alpha_shape = [batch_size] + [1] * (len(real_data.shape) - 1)
    alpha = tf.random.uniform(shape=alpha_shape, dtype=data_dtype)
    interpolated_modality = real_data + alpha * (fake_data - real_data)
    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated_modality)
        current_disc_inputs = list(disc_all_inputs_template)
        current_disc_inputs[modality_idx] = interpolated_modality
        current_disc_inputs[-1] = tabular_cond
        pred = discriminator(current_disc_inputs, training=True)
        pred_f32 = tf.cast(pred, tf.float32)
    grads = gp_tape.gradient(pred_f32, interpolated_modality)
    if grads is None: return tf.constant(0.0, dtype=tf.float32)
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=tf.range(1, tf.rank(grads))) + 1e-8)
    gp = tf.reduce_mean((norm - 1.0) ** 2)
    return gp

@tf.function
def calculate_physics_loss_gan(fake_gw, fake_xray, fake_vol, tabular_conds):
    physics_loss = tf.constant(0.0, dtype=tf.float32)
    return physics_loss

@tf.function
def calculate_cross_modal_consistency_loss_gan(fake_img, fake_gw, fake_xray, fake_vol, tabular_conds):
    consistency_loss = tf.constant(0.0, dtype=tf.float32)
    return consistency_loss

@tf.function
def train_step_multimodal(generator, discriminator,
                          real_img, real_gw, real_xray, real_vol, real_tabular_conds,
                          batch_size, latent_dim,
                          generator_optimizer, discriminator_optimizer,
                          gp_lambda_val, n_critic,
                          physics_loss_weight_val,
                          consistency_loss_weight_val):
    real_img = tf.cast(real_img, tf.float32)
    real_gw = tf.cast(real_gw, tf.float32)
    real_xray = tf.cast(real_xray, tf.float32)
    real_vol = tf.cast(real_vol, tf.float32)
    real_tabular_conds = tf.cast(real_tabular_conds, tf.float32)

    d_loss_final_iter, raw_gp_final_iter = tf.constant(0.0, dtype=tf.float32), tf.constant(0.0, dtype=tf.float32)
    mean_real_pred_final_iter, mean_fake_pred_final_iter = tf.constant(0.0, dtype=tf.float32), tf.constant(0.0, dtype=tf.float32)

    for _ in tf.range(n_critic):
        with tf.GradientTape() as disc_tape:
            noise = tf.random.normal([batch_size, latent_dim], dtype=tf.float32)
            fake_outputs_list = generator([noise, real_tabular_conds], training=True)
            fake_img, fake_gw, fake_xray, fake_vol = [tf.cast(o, tf.float32) for o in fake_outputs_list]

            real_pred_critic = discriminator([real_img, real_gw, real_xray, real_vol, real_tabular_conds], training=True)
            fake_pred_critic = discriminator([fake_img, fake_gw, fake_xray, fake_vol, real_tabular_conds], training=True)

            loss_real = -tf.reduce_mean(real_pred_critic)
            loss_fake = tf.reduce_mean(fake_pred_critic)

            gp_img = single_modality_gradient_penalty(discriminator, real_img, fake_img, [None, real_gw, real_xray, real_vol, real_tabular_conds], 0, real_tabular_conds)
            gp_gw = single_modality_gradient_penalty(discriminator, real_gw, fake_gw, [real_img, None, real_xray, real_vol, real_tabular_conds], 1, real_tabular_conds)
            gp_xray = single_modality_gradient_penalty(discriminator, real_xray, fake_xray, [real_img, real_gw, None, real_vol, real_tabular_conds], 2, real_tabular_conds)
            gp_vol = single_modality_gradient_penalty(discriminator, real_vol, fake_vol, [real_img, real_gw, real_xray, None, real_tabular_conds], 3, real_tabular_conds)
            gp_total = (gp_img + gp_gw + gp_xray + gp_vol) / 4.0
            d_loss_iter = loss_real + loss_fake + tf.cast(gp_lambda_val, tf.float32) * gp_total

        gradients_of_discriminator = disc_tape.gradient(d_loss_iter, discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
        d_loss_final_iter = d_loss_iter
        raw_gp_final_iter = gp_total
        mean_real_pred_final_iter = -loss_real
        mean_fake_pred_final_iter = loss_fake

    calculated_physics_loss = tf.constant(0.0, dtype=tf.float32)
    calculated_consistency_loss = tf.constant(0.0, dtype=tf.float32)

    with tf.GradientTape() as gen_tape:
        noise = tf.random.normal([batch_size, latent_dim], dtype=tf.float32)
        fake_outputs_for_g_list = generator([noise, real_tabular_conds], training=True)
        fake_img_g, fake_gw_g, fake_xray_g, fake_vol_g = [tf.cast(o, tf.float32) for o in fake_outputs_for_g_list]
        fake_pred_for_g = discriminator([fake_img_g, fake_gw_g, fake_xray_g, fake_vol_g, real_tabular_conds], training=True)
        g_adversarial_loss = -tf.reduce_mean(fake_pred_for_g)
        g_loss = g_adversarial_loss
        current_physics_weight = tf.cast(physics_loss_weight_val, tf.float32)
        if current_physics_weight > 0:
            calculated_physics_loss = calculate_physics_loss_gan(fake_gw_g, fake_xray_g, fake_vol_g, real_tabular_conds)
            g_loss += current_physics_weight * calculated_physics_loss
        current_consistency_weight = tf.cast(consistency_loss_weight_val, tf.float32)
        if current_consistency_weight > 0:
            calculated_consistency_loss = calculate_cross_modal_consistency_loss_gan(fake_img_g, fake_gw_g, fake_xray_g, fake_vol_g, real_tabular_conds)
            g_loss += current_consistency_weight * calculated_consistency_loss
    gradients_of_generator = gen_tape.gradient(g_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    return d_loss_final_iter, g_loss, mean_real_pred_final_iter, mean_fake_pred_final_iter, raw_gp_final_iter, calculated_physics_loss, calculated_consistency_loss

def train_gan_multimodal(generator, discriminator,
                         all_real_data_tuple,
                         tabular_conditions_real_all_scaled,
                         latent_dim, epochs, batch_size=GlobalConfig.GAN_BATCH_SIZE, wandb_run=None,
                         lr_g_sweep=None, lr_d_sweep=None, gp_lambda_sweep=None, n_critic_sweep=None,
                         physics_loss_weight_sweep=DEFAULT_PHYSICS_LOSS_WEIGHT,
                         consistency_loss_weight_sweep=DEFAULT_CONSISTENCY_LOSS_WEIGHT):

    current_lr_g = lr_g_sweep if lr_g_sweep is not None else DEFAULT_LR_G
    current_lr_d = lr_d_sweep if lr_d_sweep is not None else DEFAULT_LR_D
    current_gp_lambda = gp_lambda_sweep if gp_lambda_sweep is not None else DEFAULT_GP_LAMBDA
    current_n_critic = n_critic_sweep if n_critic_sweep is not None else DEFAULT_N_CRITIC
    current_physics_weight = physics_loss_weight_sweep
    current_consistency_weight = consistency_loss_weight_sweep

    generator_optimizer_gan = Adam(learning_rate=current_lr_g, beta_1=0.5, beta_2=0.9, clipvalue=1.0)
    discriminator_optimizer_gan = Adam(learning_rate=current_lr_d, beta_1=0.5, beta_2=0.9, clipvalue=1.0)
    
    print(f"\n--- Training Multimodal Conditional WGAN-GP (Epochs: {epochs}, Batch Size: {batch_size}, N_Critic: {current_n_critic}, "
          f"LR_G: {current_lr_g:.1e}, LR_D: {current_lr_d:.1e}, GP_Lambda: {current_gp_lambda}, "
          f"Physics_W: {current_physics_weight:.2e}, Consistency_W: {current_consistency_weight:.2e}) ---")
    
    images_real_all_norm = (all_real_data_tuple[0].astype('float32') - 127.5) / 127.5
    gw_data_orig = all_real_data_tuple[1].astype('float32')
    gw_min, gw_max = np.min(gw_data_orig), np.max(gw_data_orig)
    gw_real_all_norm = ((gw_data_orig - gw_min) / (gw_max - gw_min + 1e-7) * 2.0 - 1.0) if (gw_max - gw_min) > 1e-7 else np.zeros_like(gw_data_orig)
    xray_data_orig = all_real_data_tuple[2].astype('float32')
    xray_min_norm, xray_max_norm = np.min(xray_data_orig), np.max(xray_data_orig)
    if (xray_max_norm - xray_min_norm) > 1e-7:
        xray_real_all_norm = (xray_data_orig - xray_min_norm) / (xray_max_norm - xray_min_norm)
    else:
        xray_real_all_norm = np.zeros_like(xray_data_orig)
    vol_real_all_norm = (all_real_data_tuple[3].astype('float32') / 255.0)
    tabular_conditions_real_all_scaled = tabular_conditions_real_all_scaled.astype('float32')
    num_samples = images_real_all_norm.shape[0]
    if num_samples < batch_size or batch_size == 0 :
        batch_size = max(1, num_samples)
        if batch_size == 0: print("Error: No samples for GAN training."); return
    if gw_real_all_norm.ndim == 2: gw_real_all_norm = np.expand_dims(gw_real_all_norm, axis=-1)
    if xray_real_all_norm.ndim == 2: xray_real_all_norm = np.expand_dims(xray_real_all_norm, axis=-1)
    if vol_real_all_norm.ndim == 4: vol_real_all_norm = np.expand_dims(vol_real_all_norm, axis=-1)
    dataset = tf.data.Dataset.from_tensor_slices((
        images_real_all_norm, gw_real_all_norm, xray_real_all_norm, vol_real_all_norm, tabular_conditions_real_all_scaled
    )).shuffle(buffer_size=max(1,num_samples), seed=GlobalConfig.RANDOM_SEED).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    batches_per_epoch = len(dataset)
    if batches_per_epoch == 0: print(f"Error: Dataset has 0 batches (num_samples={num_samples}, batch_size={batch_size}). Skipping GAN."); return

    for epoch in range(epochs):
        epoch_d_loss_sum, epoch_g_loss_sum, epoch_real_score_sum = 0.0, 0.0, 0.0
        epoch_fake_score_sum, epoch_gp_sum, epoch_physics_loss_sum, epoch_consistency_loss_sum = 0.0, 0.0, 0.0, 0.0
        for batch_data in tqdm(dataset, desc=f"Epoch {epoch+1}/{epochs}", total=batches_per_epoch):
            batch_img, batch_gw, batch_xray, batch_vol, batch_tab_conds = batch_data
            current_batch_size_actual = tf.shape(batch_img)[0].numpy()
            if current_batch_size_actual == 0: continue
            d_loss, g_loss, real_score, fake_score, raw_gp, physics_loss_val, consistency_loss_val = train_step_multimodal(
                generator, discriminator,
                batch_img, batch_gw, batch_xray, batch_vol, batch_tab_conds,
                tf.cast(current_batch_size_actual, tf.int32),
                latent_dim,
                generator_optimizer_gan, discriminator_optimizer_gan,
                current_gp_lambda, tf.cast(current_n_critic, tf.int32),
                current_physics_weight, current_consistency_weight
            )
            epoch_d_loss_sum += d_loss.numpy()
            epoch_g_loss_sum += g_loss.numpy()
            epoch_real_score_sum += real_score.numpy()
            epoch_fake_score_sum += fake_score.numpy()
            epoch_gp_sum += raw_gp.numpy()
            epoch_physics_loss_sum += physics_loss_val.numpy()
            epoch_consistency_loss_sum += consistency_loss_val.numpy()
        avg_d_loss = epoch_d_loss_sum / batches_per_epoch
        avg_g_loss = epoch_g_loss_sum / batches_per_epoch
        avg_real_score = epoch_real_score_sum / batches_per_epoch
        avg_fake_score = epoch_fake_score_sum / batches_per_epoch
        avg_gp = epoch_gp_sum / batches_per_epoch
        avg_physics_loss = epoch_physics_loss_sum / batches_per_epoch
        avg_consistency_loss = epoch_consistency_loss_sum / batches_per_epoch
        g_adv_loss_approx = avg_g_loss - (current_physics_weight * avg_physics_loss) - (current_consistency_weight * avg_consistency_loss)
        print(f"Epoch {epoch+1}/{epochs} Avg [D:{avg_d_loss:.3f}] [G_total:{avg_g_loss:.3f}] [G_adv:{g_adv_loss_approx:.3f}] "
              f"[PhysL:{avg_physics_loss:.3f}] [ConsL:{avg_consistency_loss:.3f}] "
              f"[RealS:{avg_real_score:.3f}] [FakeS:{avg_fake_score:.3f}] [GP:{avg_gp:.3f}]")
        if wandb_run: wandb_run.log({ "gan_multimodal/d_loss": avg_d_loss, "gan_multimodal/g_loss_total": avg_g_loss, "gan_multimodal/g_adversarial_loss_approx": g_adv_loss_approx, "gan_multimodal/physics_loss": avg_physics_loss, "gan_multimodal/consistency_loss": avg_consistency_loss, "gan_multimodal/real_score_avg": avg_real_score, "gan_multimodal/fake_score_avg": avg_fake_score, "gan_multimodal/raw_gradient_penalty": avg_gp, }, step=epoch)
        if (epoch % max(1, (epochs // 10))) == 0 or epoch == epochs - 1:
            if tabular_conditions_real_all_scaled.shape[0] > 0:
                num_samples_to_plot = min(4, tabular_conditions_real_all_scaled.shape[0])
                if num_samples_to_plot > 0:
                    test_noise = np.random.normal(0, 1, (num_samples_to_plot, latent_dim))
                    sample_indices_for_cond = np.random.choice(tabular_conditions_real_all_scaled.shape[0], num_samples_to_plot, replace=False)
                    test_tab_cond = tabular_conditions_real_all_scaled[sample_indices_for_cond]
                    gen_outputs_list = generator.predict([test_noise, test_tab_cond], verbose=0)
                    gen_img_samples = gen_outputs_list[0]
                    img_filename_prefix = "gan_multimodal_img_epoch_"
                    plot_gan_generated_samples(gen_img_samples, epoch + 1, examples=num_samples_to_plot, filename_prefix=img_filename_prefix)
                    img_save_path = os.path.join(GlobalConfig.RESULTS_DIR, f"{img_filename_prefix}{epoch+1:04d}.png")
                    if wandb_run and os.path.exists(img_save_path): wandb_run.log({"gan_multimodal/generated_images": wandb.Image(img_save_path)}, step=epoch)
                    if len(gen_outputs_list) > 1 and gen_outputs_list[1].shape[0] > 0:
                        gen_gw_samples = gen_outputs_list[1]
                        plt.figure(figsize=(6,3)); plt.plot(gen_gw_samples[0, :, 0])
                        plt.title(f"Generated GW Epoch {epoch+1} Sample 0"); plt.xlabel("Timestep"); plt.ylabel("Amplitude (Normalized)")
                        gw_filename = f"gan_multimodal_gw_epoch_{epoch+1:04d}.png"
                        gw_save_path = os.path.join(GlobalConfig.RESULTS_DIR, gw_filename)
                        plt.savefig(gw_save_path); plt.close()
                        if wandb_run and os.path.exists(gw_save_path): wandb_run.log({"gan_multimodal/generated_gw": wandb.Image(gw_save_path)}, step=epoch)
                    if len(gen_outputs_list) > 2 and gen_outputs_list[2].shape[0] > 0:
                        gen_xray_samples = gen_outputs_list[2]
                        plt.figure(figsize=(6,3)); plt.plot(gen_xray_samples[0, :, 0], color='orange')
                        plt.title(f"Generated X-ray Epoch {epoch+1} Sample 0"); plt.xlabel("Timestep"); plt.ylabel("Brightness (Normalized)")
                        xray_filename = f"gan_multimodal_xray_epoch_{epoch+1:04d}.png"
                        xray_save_path = os.path.join(GlobalConfig.RESULTS_DIR, xray_filename)
                        plt.savefig(xray_save_path); plt.close()
                        if wandb_run and os.path.exists(xray_save_path): wandb_run.log({"gan_multimodal/generated_xray": wandb.Image(xray_save_path)}, step=epoch)
                    if len(gen_outputs_list) > 3 and gen_outputs_list[3].shape[0] > 0:
                        gen_vol_samples = gen_outputs_list[3]
                        first_vol_sample_for_plot = gen_vol_samples[0, :, :, :, 0]
                        vol_filename_prefix = f"gan_multimodal_vol_epoch_"
                        plot_3d_data_slices(first_vol_sample_for_plot, filename=os.path.join(GlobalConfig.RESULTS_DIR, f"{vol_filename_prefix}{epoch+1:04d}.png"))
                        vol_save_path = os.path.join(GlobalConfig.RESULTS_DIR, f"{vol_filename_prefix}{epoch+1:04d}.png")
                        if wandb_run and os.path.exists(vol_save_path): wandb_run.log({"gan_multimodal/generated_volumetric_slices": wandb.Image(vol_save_path)}, step=epoch)
    print("Multimodal GAN training finished. Saving models.")
    if not os.path.exists(GlobalConfig.MODELS_DIR): os.makedirs(GlobalConfig.MODELS_DIR)
    try:
        generator.save(os.path.join(GlobalConfig.MODELS_DIR, 'gan_multimodal_generator.keras'))
        discriminator.save(os.path.join(GlobalConfig.MODELS_DIR, 'gan_multimodal_discriminator.keras'))
        print("Multimodal GAN models saved in .keras format.")
    except Exception as e:
        print(f"Error saving Multimodal GAN models: {e}")
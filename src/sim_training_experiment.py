# src/sim_training_experiment.py (完整文件)
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os
from src.utils import generate_synthetic_physics_sim_data, plot_sim_prediction, plot_history
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint
from config import GlobalConfig
from tensorflow.keras.callbacks import ModelCheckpoint

def create_sequences(data, n_steps_in, n_steps_out=1):
    """ Splits the time series data into sequences for supervised learning. """
    X, y = [], []
    for i in range(len(data)):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        if out_end_ix > len(data):
            break
        seq_x, seq_y = data[i:end_ix], data[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def run_sim_training_experiment(wandb_run=None):
    print("\n--- Running Simulation-Based Training Experiment (Predicting Simplified Physics) ---")
    
    sim_data_filepath = os.path.join(GlobalConfig.DATA_DIR, 'sim_physics_data.npy')
    if not os.path.exists(GlobalConfig.DATA_DIR):
        os.makedirs(GlobalConfig.DATA_DIR)
    if not os.path.exists(sim_data_filepath):
        generate_synthetic_physics_sim_data(filename=sim_data_filepath)
    
    sim_data_all_series = np.load(sim_data_filepath)
    if sim_data_all_series.shape[0] == 0:
        print("Error: Loaded simulation data is empty. Skipping experiment."); return

    single_simulation_series = sim_data_all_series[0]
    scaler = MinMaxScaler(feature_range=(0, 1))
    single_simulation_series_scaled = scaler.fit_transform(single_simulation_series.reshape(-1, 1))

    n_steps_in, n_steps_out, epochs, batch_size = 10, 1, 50, 32
    X, y = create_sequences(single_simulation_series_scaled.flatten(), n_steps_in, n_steps_out)
    if X.shape[0] == 0:
        print("Not enough data to create sequences."); return

    X = X.reshape(X.shape[0], X.shape[1], 1)
    if n_steps_out == 1 and y.ndim == 1: y = y.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=GlobalConfig.RANDOM_SEED, shuffle=False)

    print("Building LSTM model...")
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(n_steps_in, 1), return_sequences=True),
        Dropout(0.2),
        LSTM(50, activation='relu'),
        Dropout(0.2),
        Dense(n_steps_out)
    ])

    learning_rate = 0.001
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    model.summary()

    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=GlobalConfig.HP_TUNING_PATIENCE, restore_best_weights=True)]
    if wandb_run:
        wandb_config = {"model_type": "LSTM_Physics_Sim", "n_steps_in": n_steps_in, "n_steps_out": n_steps_out, "lstm_units": 50, "dropout_rate": 0.2, "learning_rate": learning_rate, "epochs": epochs, "batch_size": batch_size}
        wandb_run.config.update({"sim_training_experiment": wandb_config})
        callbacks.append(WandbMetricsLogger(log_freq="epoch"))
        if GlobalConfig.WANDB_LOG_MODELS:
            checkpoint_filepath = f"wandb_models/{wandb_run.id or 'sim_exp'}/sim_lstm_best.keras"
            callbacks.append(ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_loss', mode='min', save_best_only=True, save_weights_only=False))

    print("Training LSTM model (using GPU if available)...")
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=1, callbacks=callbacks)
    
    plot_history(history, filename=os.path.join(GlobalConfig.RESULTS_DIR, 'sim_training_history.png'))

    test_predictions_scaled = model.predict(X_test)
    test_predictions = scaler.inverse_transform(test_predictions_scaled)
    y_test_original = scaler.inverse_transform(y_test)

    if len(y_test_original) > 0:
        plot_sim_prediction(y_test_original.flatten(), test_predictions.flatten(), filename=os.path.join(GlobalConfig.RESULTS_DIR, 'sim_prediction_plot.png'))
        mse_test = np.mean((y_test_original - test_predictions)**2)
        mae_test = np.mean(np.abs(y_test_original - test_predictions))
        print(f"Mean Squared Error on test set: {mse_test:.4f}")
        print(f"Mean Absolute Error on test set: {mae_test:.4f}")
        if wandb_run: wandb_run.log({"sim_training/test_mse": mse_test, "sim_training/test_mae": mae_test})
    else:
        print("Not enough test data to plot prediction.")

    # --- KEY FIX: Saving in .keras format ---
    model_filepath = os.path.join(GlobalConfig.MODELS_DIR, 'sim_trained_model.keras')
    if not os.path.exists(GlobalConfig.MODELS_DIR): os.makedirs(GlobalConfig.MODELS_DIR)
    model.save(model_filepath)
    print(f"Simulation-based trained model saved to {model_filepath}")
    
    print("Simulation-Based Training Experiment finished.")
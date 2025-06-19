# config.py
import numpy as np

# --- Global Configuration ---
class GlobalConfig:
    # Path Configuration
    DATA_DIR = 'data'
    MODELS_DIR = 'models'
    RESULTS_DIR = 'results'
    KERAS_TUNER_DIR = 'keras_tuner_dir'
    RESEARCH_PAPERS_DIR = f'{DATA_DIR}/research_papers'

    # Dataset Sizes
    NUM_INITIAL_SAMPLES = 1000
    NUM_STREAM_BATCHES = 5
    SAMPLES_PER_BATCH_STREAM = 15

    # Modality Data Dimensions
    IMG_SIZE = 64
    GW_TIMESTEPS = 128
    X_RAY_TIMESTEPS = 100
    VOLUMETRIC_SIZE = 16

    # Classification Labels
    LABELS_MAP = {
        0: "Non-BH",
        1: "Black Hole",
        2: "White Hole",
        3: "BH Formation"
    }
    NUM_CLASSES = len(LABELS_MAP) # This will be updated in main.py if data has different num_classes

    # --- ADDED: Placeholder Feature Dimensions (IMPORTANT: ADJUST THESE TO YOUR EXTRACTOR OUTPUTS) ---
    # These dimensions should match the actual output feature dimensions of your respective extractor models.
    # If an extractor is not loaded, these values are used to correctly shape the input to the fusion model.
    # For build_tabular_feature_extractor, the output is Dense(32, ...), so it's 32.
    DUMMY_IMAGE_FEATURE_DIM = 512   # Example: For a large image model (e.g., ResNet50 output)
    DUMMY_TABULAR_FEATURE_DIM = 32   # Matches `build_tabular_feature_extractor` output
    DUMMY_GW_FEATURE_DIM = 128        # Example: For a time-series model (e.g., LSTM/CNN output)
    DUMMY_XRAY_FEATURE_DIM = 128      # Example: For a time-series model (e.g., LSTM/CNN output)
    DUMMY_VOLUMETRIC_FEATURE_DIM = 256 # Example: For a 3D CNN output

    # Training Configuration
    INITIAL_TRAINING_EPOCHS = 40
    INITIAL_TRAINING_BATCH_SIZE = 32
    ONLINE_LEARNING_EPOCHS = 2
    ONLINE_LEARNING_BATCH_SIZE = 16

    # GAN Configuration
    GAN_LATENT_DIM = 100
    GAN_EPOCHS = 500
    GAN_BATCH_SIZE = 128

    # Default GAN Hyperparameters
    GAN_LEARNING_RATE_G_DEFAULT = 0.00002
    GAN_LEARNING_RATE_D_DEFAULT = 0.00002
    GAN_GP_LAMBDA_DEFAULT = 10.0
    GAN_N_CRITIC_DEFAULT = 7
    GAN_PHYSICS_LOSS_WEIGHT_DEFAULT = 0.1  # Added
    GAN_CONSISTENCY_LOSS_WEIGHT_DEFAULT = 0.05 # Added


    # Pre-trained Feature Extractor Configuration
    FEAT_EXTRACTOR_EPOCHS = 30
    FEAT_EXTRACTOR_BATCH_SIZE = 32

    # Hyperparameter Tuning Configuration
    HP_TUNING_MAX_TRIALS =10
    HP_TUNING_MAX_EPOCHS_PER_TRIAL = 5
    HP_TUNING_PATIENCE = 3

    # Active Learning Configuration
    ACTIVE_LEARNING_UNCERTAINTY_THRESHOLD = 0.7
    ACTIVE_LEARNING_SELECTION_RATIO = 0.5
    REPLAY_BUFFER_MAX_SIZE = 1000

    # XAI Configuration
    XAI_NUM_SAMPLES = 50

    # --- ADDED: Recommendation System Thresholds ---
    # Threshold for regression prediction to be considered high energy risk
    REGRESSION_RISK_THRESHOLD_HIGH_ENERGY = 25.0
    # List of classification labels considered as high risk (e.g., rare or critical types)
    CLASSIFICATION_RISK_LABELS = ["White Hole", "BH Formation"]
    # Classification uncertainty threshold for flagging (e.g., above 0.20 for overall epistemic uncertainty)
    CLASSIFICATION_UNCERTAINTY_THRESHOLD = 0.20
    # Regression uncertainty factors for flagging (total_reg_unc_approx > (relative_factor * abs(pred) + absolute_offset))
    REGRESSION_UNCERTAINTY_RELATIVE_FACTOR = 0.25
    REGRESSION_UNCERTAINTY_ABSOLUTE_OFFSET = 5.0


    # Weights & Biases Configuration
    WANDB_PROJECT = "BlackHole_Research_Framework_V3_1"
    WANDB_ENTITY = "bcs24020043-university-of-technology-sarawak" # Replace with your entity
    WANDB_LOG_MODELS = True

    # Other
    RANDOM_SEED = 42
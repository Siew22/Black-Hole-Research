# src/xai_analysis.py (最终完整版 - 采纳你的见解并修复逻辑)

import numpy as np
import tensorflow as tf
from tensorflow import keras
import shap
import os
import matplotlib.pyplot as plt
import wandb
from config import GlobalConfig
import cv2
import traceback

# 导入你的自定义层
from src.cnn_image_extractor import IdentityLayer


# --- Grad-CAM 辅助函数 (已验证成功，保持不变) ---
# ... (为了简洁，这里省略了 make_gradcam_heatmap 的代码，但你应该在你的文件中保留它) ...
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + keras.backend.epsilon())
    return heatmap.numpy()

# --- 新增：采纳你的见解，编写健壮的 SHAP 绘图辅助函数 ---
def create_shap_bar_plot(shap_values, feature_names, plot_title, num_to_display=20):
    """
    一个健壮的函数，用于创建并保存 SHAP 特征重要性条形图。
    这个版本能正确处理多分类和回归的 SHAP 值形状。
    """
    # 1. 计算每个特征的全局平均绝对 SHAP 值
    if isinstance(shap_values, np.ndarray):
        if shap_values.ndim == 3:  # 多分类情况, shape: (n_samples, n_features, n_classes)
            # 我们沿着样本轴(axis=0)和类别轴(axis=2)求平均
            global_shap_values = np.mean(np.abs(shap_values), axis=(0, 2))
        elif shap_values.ndim == 2:  # 回归或二分类情况, shape: (n_samples, n_features)
            global_shap_values = np.mean(np.abs(shap_values), axis=0)
        else:
            raise ValueError(f"Unsupported SHAP values ndarray format with ndim={shap_values.ndim}")
    else:
        # 新版 SHAP 倾向于返回 ndarray，但以防万一处理 list 的情况
        if isinstance(shap_values, list):
            shap_values_np = np.array(shap_values) # shape (n_classes, n_samples, n_features)
            global_shap_values = np.mean(np.abs(shap_values_np), axis=(0, 1))
        else:
            raise ValueError(f"Unsupported SHAP values format: {type(shap_values)}")

    # 确保特征数量匹配
    if len(global_shap_values) != len(feature_names):
        raise ValueError(f"Shape mismatch: {len(global_shap_values)} SHAP values vs {len(feature_names)} feature names.")

    # 2. 排序并获取最重要的特征
    num_to_display = min(num_to_display, len(global_shap_values))
    sorted_indices = np.argsort(global_shap_values)[-num_to_display:]
    
    sorted_values = global_shap_values[sorted_indices]
    sorted_names = np.array(feature_names)[sorted_indices]

    # 3. 使用 matplotlib 创建条形图
    plt.figure(figsize=(10, num_to_display * 0.4 + 1.5))
    y_pos = np.arange(len(sorted_names))
    plt.barh(y_pos, sorted_values)
    plt.yticks(y_pos, sorted_names)
    plt.gca().invert_yaxis() # 让最重要的在最上面
    plt.xlabel("mean(|SHAP value|) (average impact on model output magnitude)")
    plt.title(plot_title)
    plt.tight_layout()
    
    return plt.gcf()


def run_xai_analysis(X_images_raw, X_tabular_raw, X_gw_raw, X_xray_raw, X_volumetric_raw,
                     y_labels_raw, y_regression_targets_raw,
                     multimodal_fusion_system_instance,
                     wandb_run=None):
    """
    运行完整的 XAI 分析，包括 Grad-CAM 和 SHAP。
    """
    print("\n--- XAI Analysis Stage ---")
    print("\n--- Running XAI Analysis ---")

    if not os.path.exists(GlobalConfig.RESULTS_DIR):
        os.makedirs(GlobalConfig.RESULTS_DIR)

    mfs = multimodal_fusion_system_instance
    if not mfs.fusion_model:
        print("XAI Error: Fusion model is not available. Skipping XAI analysis.")
        return
    if not mfs.is_scaler_fitted:
        print("XAI Warning: Scalers are not fitted in the fusion system. SHAP results may be inaccurate.")

    # --- 1. Grad-CAM Analysis ---
    print("\nRunning Grad-CAM analysis on a sample image...")
    full_image_model_path = os.path.join(GlobalConfig.MODELS_DIR, 'cnn_full_classifier_model.keras')

    if not os.path.exists(full_image_model_path):
        print(f"Grad-CAM Error: Full image classifier not found at {full_image_model_path}.")
    elif X_images_raw is None or X_images_raw.shape[0] == 0:
        print("Grad-CAM Error: No image data provided for analysis.")
    else:
        try:
            full_image_model = keras.models.load_model(
                full_image_model_path,
                custom_objects={'IdentityLayer': IdentityLayer},
                compile=False
            )
            
            resnet_submodel = full_image_model.get_layer('resnet50')
            last_conv_layer = None
            for layer in reversed(resnet_submodel.layers):
                if isinstance(layer, keras.layers.Conv2D):
                    last_conv_layer = layer
                    break
            if last_conv_layer is None: raise ValueError("No Conv2D layer in ResNet.")
            
            print(f"  Using target layer for Grad-CAM: '{last_conv_layer.name}' in submodel 'resnet50'")

            last_conv_layer_model = keras.Model(resnet_submodel.inputs, last_conv_layer.output)
            classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
            x = full_image_model.get_layer('flatten')(classifier_input)
            x = full_image_model.get_layer('image_features_dense')(x)
            x = full_image_model.get_layer('batch_normalization')(x)
            x = full_image_model.get_layer('dropout')(x)
            x = full_image_model.get_layer('image_features_output')(x)
            classifier_output = full_image_model.get_layer('classification_output')(x)
            classifier_model = keras.Model(classifier_input, classifier_output)

            sample_idx = 0
            sample_image_raw = X_images_raw[sample_idx]
            img_array = np.expand_dims(sample_image_raw.astype('float32') / 255.0, axis=0)

            with tf.GradientTape() as tape:
                last_conv_layer_output = last_conv_layer_model(img_array)
                tape.watch(last_conv_layer_output)
                preds = classifier_model(last_conv_layer_output)
                top_pred_index = tf.argmax(preds[0])
                top_class_channel = preds[:, top_pred_index]
            
            grads = tape.gradient(top_class_channel, last_conv_layer_output)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            heatmap = (last_conv_layer_output[0] @ pooled_grads[..., tf.newaxis]).numpy()
            heatmap = np.maximum(heatmap, 0)
            if np.max(heatmap) > 0: heatmap /= np.max(heatmap)
            heatmap = np.squeeze(heatmap)

            heatmap_resized = cv2.resize(heatmap, (sample_image_raw.shape[1], sample_image_raw.shape[0]))
            heatmap_jet = plt.cm.jet(heatmap_resized)[:, :, :3]
            superimposed_img = np.clip(heatmap_jet * 0.4 + (sample_image_raw.astype('float32') / 255.0), 0, 1)
            
            filename_gradcam = os.path.join(GlobalConfig.RESULTS_DIR, 'grad_cam_example.png')
            plt.figure(figsize=(8, 8))
            plt.imshow(superimposed_img)
            plt.title(f"Grad-CAM - Pred: {GlobalConfig.LABELS_MAP.get(top_pred_index.numpy(), 'Unknown')}")
            plt.axis('off')
            plt.savefig(filename_gradcam); plt.close()
            
            print(f"  Grad-CAM example saved to {filename_gradcam}")
            if wandb_run: wandb_run.log({"xai/grad_cam": wandb.Image(filename_gradcam)})

        except Exception as e_gradcam:
            print(f"  An error occurred during Grad-CAM analysis: {e_gradcam}")
            traceback.print_exc()

    # --- 2. SHAP Analysis ---
    print("\nRunning SHAP analysis on fused features...")
    
    num_xai_samples = X_images_raw.shape[0] if X_images_raw is not None and X_images_raw.ndim > 1 else 0
    if num_xai_samples == 0:
        print("  XAI Error: No valid data provided for SHAP analysis. Skipping.")
    else:
        img_feats_all, tab_feats_all, gw_feats_all, xray_feats_all, vol_feats_all = \
            mfs._extract_features(X_images_raw, X_tabular_raw, X_gw_raw, X_xray_raw, X_volumetric_raw, batch_size_pred=num_xai_samples)
        
        if any(f is None or f.shape[0] != num_xai_samples for f in [img_feats_all, tab_feats_all, gw_feats_all, xray_feats_all, vol_feats_all]):
            print("  XAI Error: Feature extraction for SHAP failed or returned inconsistent sample numbers. Skipping SHAP.")
        else:
            ts_combined_feats_all = np.concatenate([gw_feats_all, xray_feats_all], axis=-1)
            all_features_fused_for_shap = np.concatenate([img_feats_all, tab_feats_all, ts_combined_feats_all, vol_feats_all], axis=1)

            try:
                num_background_samples_shap = min(50, all_features_fused_for_shap.shape[0] // 2)
                num_explain_samples_shap = min(5, all_features_fused_for_shap.shape[0])
                
                if num_background_samples_shap < 1 or num_explain_samples_shap < 1:
                     raise ValueError("Not enough data for SHAP analysis.")

                background_data_shap = shap.sample(all_features_fused_for_shap, num_background_samples_shap, random_state=GlobalConfig.RANDOM_SEED)
                explain_data_shap = all_features_fused_for_shap[:num_explain_samples_shap]

                feature_names_shap = [f"Img_F{i}" for i in range(img_feats_all.shape[1])] + \
                                     [f"Tab_F{i}" for i in range(tab_feats_all.shape[1])] + \
                                     [f"TS_F{i}" for i in range(ts_combined_feats_all.shape[1])] + \
                                     [f"Vol_F{i}" for i in range(vol_feats_all.shape[1])]

                # --- SHAP for Classification ---
                def predict_cls_for_shap(data):
                    feat_dims = [img_feats_all.shape[1], tab_feats_all.shape[1], gw_feats_all.shape[1], xray_feats_all.shape[1], vol_feats_all.shape[1]]
                    inputs_list = []; current_idx = 0
                    for dim in feat_dims:
                        inputs_list.append(data[:, current_idx:current_idx + dim])
                        current_idx += dim
                    return mfs.fusion_model.predict(inputs_list, verbose=0)[0]

                print("  Initializing SHAP KernelExplainer for classification...")
                explainer_cls = shap.KernelExplainer(predict_cls_for_shap, background_data_shap)
                print(f"  Calculating SHAP values for classification (on {explain_data_shap.shape[0]} samples)...")
                shap_values_cls = explainer_cls.shap_values(explain_data_shap, nsamples='auto')
                
                print("  Plotting SHAP summary for Classification...")
                fig_cls = create_shap_bar_plot(shap_values_cls, feature_names_shap, "SHAP Feature Importance (Classification)")
                filename_shap_cls_summary = os.path.join(GlobalConfig.RESULTS_DIR, 'shap_summary_classification_bar.png')
                fig_cls.savefig(filename_shap_cls_summary, bbox_inches='tight'); plt.close(fig_cls)
                if wandb_run: wandb_run.log({"xai/shap_classification_summary": wandb.Image(filename_shap_cls_summary)})
                print(f"  SHAP classification summary saved to {filename_shap_cls_summary}")

                # --- SHAP for Regression ---
                def predict_reg_for_shap(data):
                    feat_dims = [img_feats_all.shape[1], tab_feats_all.shape[1], gw_feats_all.shape[1], xray_feats_all.shape[1], vol_feats_all.shape[1]]
                    inputs_list = []; current_idx = 0
                    for dim in feat_dims:
                        inputs_list.append(data[:, current_idx:current_idx + dim])
                        current_idx += dim
                    reg_output = mfs.fusion_model.predict(inputs_list, verbose=0)[1]
                    return reg_output[:, 0]

                print("  Initializing SHAP KernelExplainer for regression...")
                explainer_reg = shap.KernelExplainer(predict_reg_for_shap, background_data_shap)
                print(f"  Calculating SHAP values for regression (on {explain_data_shap.shape[0]} samples)...")
                shap_values_reg = explainer_reg.shap_values(explain_data_shap, nsamples='auto')
                
                print("  Plotting SHAP summary for Regression...")
                fig_reg = create_shap_bar_plot(shap_values_reg, feature_names_shap, "SHAP Feature Importance (Regression)")
                filename_shap_reg_summary = os.path.join(GlobalConfig.RESULTS_DIR, 'shap_summary_regression_bar.png')
                fig_reg.savefig(filename_shap_reg_summary, bbox_inches='tight'); plt.close(fig_reg)
                if wandb_run: wandb_run.log({"xai/shap_regression_summary": wandb.Image(filename_shap_reg_summary)})
                print(f"  SHAP regression summary saved to {filename_shap_reg_summary}")

            except Exception as e_shap:
                print(f"  An error occurred during SHAP analysis: {e_shap}")
                traceback.print_exc()

    print("XAI Analysis finished.")
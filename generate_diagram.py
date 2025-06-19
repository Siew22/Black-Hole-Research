# generate_diagram.py
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, concatenate, Dropout, BatchNormalization, Attention, Layer
from tensorflow.keras import regularizers

# 确保 Keras 可以加载包含 Lambda 或自定义层的模型
# 如果您已经用 IdentityLayer 替换了 Lambda, 这行依然是好习惯
tf.keras.config.enable_unsafe_deserialization()


# --- 复制 multimodal_fusion_system.py 中的模型构建函数 ---
# 我们在这里重新定义它，使此脚本可以独立运行
def build_hierarchical_fusion_model_for_diagram(
    image_feature_dim=512, 
    tabular_feature_dim=32, 
    gw_feature_dim=128, 
    xray_feature_dim=128, 
    volumetric_feature_dim=256, 
    num_classes=4, 
    l2_reg=1e-4
):
    """
    这是一个专门为生成图表而创建的模型构建函数。
    它与您在 multimodal_fusion_system.py 中的版本完全相同。
    """
    image_input = Input(shape=(image_feature_dim,), name='Image_Features')
    tabular_input = Input(shape=(tabular_feature_dim,), name='Tabular_Features')
    gw_input = Input(shape=(gw_feature_dim,), name='GW_Features')
    xray_input = Input(shape=(xray_feature_dim,), name='X-Ray_Features')
    volumetric_input = Input(shape=(volumetric_feature_dim,), name='Volumetric_Features')

    # --- Step 1: Weak Modality Pre-fusion ---
    weak_modalities_features = concatenate([gw_input, volumetric_input], name='Weak_Modalities_Concat')
    weak_fused_representation = Dense(128, activation='relu', name='Weak_Pre-Fusion_Dense')(weak_modalities_features)
    weak_fused_representation = BatchNormalization(name='Weak_Pre-Fusion_BN')(weak_fused_representation)
    weak_fused_representation = Dropout(0.4, name='Weak_Pre-Fusion_Dropout')(weak_fused_representation)

    # --- Step 2: Main Fusion ---
    final_fusion_input = concatenate([
        image_input, 
        tabular_input, 
        xray_input,
        weak_fused_representation, 
        gw_input, 
        volumetric_input
    ], name='Hierarchical_Fusion_Concat')

    # --- Step 3: Attention Mechanism ---
    attention_dense_units = int(final_fusion_input.shape[-1] * 0.5)
    attention_scores = Dense(attention_dense_units, activation='relu', name='Attention_Relu', kernel_regularizer=regularizers.l2(l2_reg))(final_fusion_input)
    attention_scores = Dense(final_fusion_input.shape[-1], activation='sigmoid', name='Attention_Weights')(attention_scores)
    fused_features_attn = tf.keras.layers.multiply([final_fusion_input, attention_scores], name='Attentive_Features')

    # --- Step 4: Shared Dense Layers ---
    x_shared = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(l2_reg), name='Shared_Dense_1')(fused_features_attn)
    x_shared = BatchNormalization()(x_shared)
    x_shared = Dropout(0.5)(x_shared)
    x_shared = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(l2_reg), name='Shared_Dense_2')(x_shared)
    x_shared = BatchNormalization()(x_shared)
    x_shared_output = Dropout(0.4, name='Shared_Features_Output')(x_shared)

    # --- Step 5: Prediction Heads ---
    classification_output = Dense(num_classes, activation='softmax', name='Classification_Output', dtype='float32')(x_shared_output)
    
    # 为了图表简洁，我们只显示回归头的最终输出层，省略内部结构
    regression_output_params = Dense(2, name='Regression_Parameters', dtype='float32')(x_shared_output)
    
    model = Model(
        inputs=[image_input, tabular_input, gw_input, xray_input, volumetric_input], 
        outputs=[classification_output, regression_output_params], 
        name='Hierarchical_Fusion_Model'
    )
    return model

def main():
    # 创建模型实例
    model = build_hierarchical_fusion_model_for_diagram()
    
    # 打印模型摘要，以便在终端确认
    model.summary(line_length=120)
    
    # 定义输出目录
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    
    # 定义文件名
    output_path = os.path.join(output_dir, 'HFM_architecture.png')
    
    # 生成并保存模型图
    # show_shapes=True: 显示每个层的输入输出形状
    # show_layer_names=True: 显示我们定义的层名称
    # rankdir='TB': 从上到下布局 (Top to Bottom)，适合论文
    # expand_nested=True: 如果有嵌套模型，则展开显示
    # dpi=96: 设置图像分辨率
    tf.keras.utils.plot_model(
        model,
        to_file=output_path,
        show_shapes=True,
        show_layer_names=True,
        rankdir='TB',
        expand_nested=True,
        dpi=96
    )
    
    print(f"\nModel architecture diagram saved to: {output_path}")
    print("You can now use this image in your report.")

if __name__ == "__main__":
    main()
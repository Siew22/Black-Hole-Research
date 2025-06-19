# 文件: utils.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw
import os
import random
from tqdm import tqdm
from skimage.draw import disk
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d
from scipy.ndimage import zoom
from mpl_toolkits.mplot3d import Axes3D
import math
import tensorflow as tf

import pyvista as pv

# 导入配置
from config import GlobalConfig
from scipy.ndimage import rotate, zoom

# --- 数据增强辅助函数 (保持不变) ---
def augment_time_series_data(ts_data):
    """
    Augments time series data with noise, scaling, and time warping.
    """
    augmented_ts = ts_data.copy()

    # 1. Gaussian Noise
    if random.random() < 0.5:
        noise_level = random.uniform(0.01, 0.05) * np.std(augmented_ts)
        augmented_ts += np.random.normal(0, noise_level, augmented_ts.shape)

    # 2. Scaling (Y-axis)
    if random.random() < 0.5:
        scale_factor = random.uniform(0.8, 1.2)
        augmented_ts *= scale_factor

    # 3. Time Warping
    if random.random() < 0.3:
        num_points = augmented_ts.shape[0]
        t_original = np.linspace(0, 1, num_points)
        t_warped = np.sort(t_original + np.random.normal(0, 0.05, num_points))
        t_warped = np.clip(t_warped, 0, 1)
        interp_func = interp1d(t_original, augmented_ts, kind='linear', fill_value="extrapolate")
        augmented_ts = interp_func(t_warped)

    return augmented_ts

def augment_volumetric_data(vol_data):
    """
    Augments volumetric data with a wider range of techniques.
    V3: Adds random zoom.
    """
    augmented_vol = vol_data.copy()

    # 1. Random Flipping
    if random.random() < 0.5:
        augmented_vol = np.flip(augmented_vol, axis=0) # Flip along X
    if random.random() < 0.5:
        augmented_vol = np.flip(augmented_vol, axis=1) # Flip along Y
    if random.random() < 0.5:
        augmented_vol = np.flip(augmented_vol, axis=2) # Flip along Z

    # 2. Random Rotation
    if random.random() < 0.7:
        angle = random.uniform(-20, 20)
        axes_to_rotate = random.choice([(0,1), (0,2), (1,2)])
        augmented_vol = rotate(augmented_vol, angle, axes=axes_to_rotate, reshape=False, mode='nearest')

    # 3. Random Zoom (NEW)
    if random.random() < 0.5:
        zoom_factor = random.uniform(0.85, 1.15)
        h, w, d = augmented_vol.shape
        
        # Zoom the volume
        zoomed_vol = zoom(augmented_vol, zoom_factor, order=1)
        
        # Crop or pad to original size
        zh, zw, zd = zoomed_vol.shape
        if zoom_factor < 1.0:
            # Pad (if smaller)
            pad_h = (h - zh) // 2
            pad_w = (w - zw) // 2
            pad_d = (d - zd) // 2
            padded_vol = np.zeros_like(augmented_vol)
            padded_vol[pad_h:pad_h+zh, pad_w:pad_w+zw, pad_d:pad_d+zd] = zoomed_vol
            augmented_vol = padded_vol
        else:
            # Crop (if larger)
            crop_h = (zh - h) // 2
            crop_w = (zw - w) // 2
            crop_d = (zd - d) // 2
            augmented_vol = zoomed_vol[crop_h:crop_h+h, crop_w:crop_w+w, crop_d:crop_d+d]

    # 4. Gaussian Blur
    if random.random() < 0.3:
        sigma = random.uniform(0.5, 1.2)
        augmented_vol = gaussian_filter(augmented_vol, sigma=sigma)

    # 5. Random Noise
    if random.random() < 0.4:
        noise_level = random.uniform(3, 10)
        augmented_vol += np.random.normal(0, noise_level, augmented_vol.shape)

    return np.clip(augmented_vol, 0, 255)

# --- 数据生成函数 ---

def generate_synthetic_ml_data(n_samples=1000):
    """
    Generates synthetic tabular data for a simplified binary classification task
    (Black Hole Candidate vs. Not).
    """
    current_random_state = np.random.RandomState(GlobalConfig.RANDOM_SEED + int(np.random.rand() * 10000))

    data = {
        'stellar_mass': current_random_state.normal(5, 2, n_samples),
        'luminosity': current_random_state.lognormal(2, 0.8, n_samples),
        'distance_kpc': current_random_state.uniform(1, 10, n_samples),
        'x_ray_activity_index': current_random_state.uniform(0, 10, n_samples)
    }
    df = pd.DataFrame(data)

    df['is_black_hole_candidate'] = ((df['stellar_mass'] > 6.5) &
                                     (df['x_ray_activity_index'] > 7.5) &
                                     (df['luminosity'] > 6) &
                                     (df['luminosity'] < 25)).astype(int)

    flip_rate = 0.02
    df['is_black_hole_candidate'] = df.apply(
        lambda row: 1 if (row['is_black_hole_candidate'] == 0 and current_random_state.rand() < flip_rate) else row['is_black_hole_candidate'],
        axis=1
    )
    if df['is_black_hole_candidate'].sum() == 0:
        df.loc[df.sample(n=max(1, int(0.1*n_samples)), random_state=current_random_state).index, 'is_black_hole_candidate'] = 1
    print(f"Generated {n_samples} ML samples. Black Hole Candidates: {df['is_black_hole_candidate'].sum()} (Imbalanced for demo)")
    return df


def _create_single_bh_image_advanced(size, mass, spin, inclination, is_formation_stage=False):
    """
    Generates a single synthetic image of a black hole or its formation stage.
    Args:
        size (int): Image dimension (size x size).
        mass (float): Mass of the black hole or progenitor star (affects shadow/disk size).
        spin (float): Spin of the black hole (0 to <1, affects shadow shape/disk inner edge).
        inclination (float): Viewing angle in radians (0 to pi/2, affects disk appearance).
        is_formation_stage (bool): If True, simulates a pre-BH state (e.g., stellar collapse).
    Returns:
        PIL.Image: An RGB image.
    """
    img = Image.new('L', (size, size), color=0)
    draw = ImageDraw.Draw(img)

    # 背景生成优化：限制背景类型，使其更简单、更均匀
    # 目的：让黑洞特征更容易学习
    bg_type = random.choice(['gradient', 'smooth_noise']) # <--- 限制为更简单的背景
    if bg_type == 'gradient':
        for x_coord in range(size):
            for y_coord in range(size):
                val = int(255 * (x_coord + y_coord) / (2 * size - 2))
                img.putpixel((x_coord, y_coord), val)
    # 移除 dense_stars 和 turbulent_noise，或调整其参数使其更平滑
    elif bg_type == 'smooth_noise': # 增加 smooth_noise 选项
        noise_map = np.random.rand(size, size) * 255
        img = Image.fromarray(gaussian_filter(noise_map, sigma=random.uniform(size/4, size/2))) # 更大的 sigma，非常平滑
    else: # Fallback, should not happen with new choice
        img = Image.new('L', (size, size), color=random.randint(0, 50)) # 默认纯黑或深灰


    if not is_formation_stage: # Mature Black Hole
        accretion_outer_norm = 0.3 + 0.1 * (mass / 100.0)
        accretion_inner_norm = accretion_outer_norm * (0.6 + 0.3 * (1 - spin))
        accretion_radius_outer = size * accretion_outer_norm
        accretion_radius_inner = size * accretion_inner_norm
        ellipse_ratio = 0.8 + 0.2 * np.cos(inclination)

        disk_center_x = size // 2 + int(random.randint(-5, 5) * (1 - inclination / (np.pi/2)))
        disk_center_y = size // 2 + int(random.randint(-5, 5) * (inclination / (np.pi/2)))

        disk_material = np.zeros((size, size), dtype=np.float32)
        for x_coord in range(size):
            for y_coord in range(size):
                transformed_x = (x_coord - disk_center_x)
                transformed_y = (y_coord - disk_center_y) / ellipse_ratio
                dist = np.sqrt(transformed_x**2 + transformed_y**2)
                if accretion_radius_inner < dist < accretion_radius_outer:
                    # 调整吸积盘亮度，使其更明显
                    brightness = int(255 * (1 - abs(dist - (accretion_radius_inner + accretion_radius_outer)/2) / ((accretion_radius_outer - accretion_radius_inner)/2)))
                    disk_material[y_coord, x_coord] = np.clip(brightness + random.uniform(0, 50), 0, 255) # 增加随机亮度
        
        # 添加热点
        if random.random() < 0.5: # 增加热点概率
            num_hot_spots = random.randint(1, 4)
            for _ in range(num_hot_spots):
                spot_r = random.uniform(accretion_radius_inner, accretion_radius_outer)
                spot_theta = random.uniform(0, 2*np.pi)
                spot_x = disk_center_x + spot_r * np.cos(spot_theta)
                spot_y = disk_center_y + spot_r * np.sin(spot_theta) * ellipse_ratio
                rr_spot, cc_spot = disk((spot_y, spot_x), random.uniform(size * 0.03, size * 0.08), shape=(size, size)) # 增大热点尺寸
                disk_material[rr_spot, cc_spot] = np.clip(disk_material[rr_spot, cc_spot] + random.uniform(100, 200), 0, 255) # 提高热点亮度

        img_array_out = np.array(img, dtype=np.float32)
        img_array_out = np.maximum(img_array_out, disk_material)
    else: # Formation Stage
        img_array_out = np.array(img, dtype=np.float32)
        core_radius = size * (0.1 + mass/200)
        rr_core, cc_core = disk((size//2, size//2), core_radius, shape=(size,size))
        img_array_out[rr_core, cc_core] = np.maximum(img_array_out[rr_core,cc_core], random.uniform(200,255))

        pil_img_for_draw = Image.fromarray(img_array_out.astype(np.uint8)).convert('L')
        draw_on_pil = ImageDraw.Draw(pil_img_for_draw)
        for _ in range(random.randint(5,10)):
            start_point = (size//2 + random.randint(-int(core_radius*0.2),int(core_radius*0.2)),
                           size//2 + random.randint(-int(core_radius*0.2),int(core_radius*0.2)))
            end_point_r = random.uniform(core_radius, size*0.4)
            end_point_angle = random.uniform(0, 2*np.pi)
            end_point = (start_point[0] + int(end_point_r * np.cos(end_point_angle)),
                         start_point[1] + int(end_point_r * np.sin(end_point_angle)))
            draw_on_pil.line([start_point, end_point], fill=random.randint(150,250), width=random.randint(1,3))
        img_array_out = np.array(pil_img_for_draw, dtype=np.float32)
        img_array_out = gaussian_filter(img_array_out, sigma=random.uniform(1.0, 2.0))

    # 黑洞阴影（事件视界代理）
    # 确保阴影足够黑，即使背景有亮度
    if not is_formation_stage or random.random() < 0.3:
        shadow_radius_norm = 0.1 + 0.15 * (1 - spin) + 0.05 * (mass / 100.0)
        if is_formation_stage: shadow_radius_norm *= random.uniform(0.1, 0.5)
        shadow_radius = size * shadow_radius_norm
        shadow_center_x = size // 2 + int(random.randint(-size//20, size//20) * (1 - inclination / (np.pi/2)))
        shadow_center_y = size // 2 + int(random.randint(-size//20, size//20) * (inclination / (np.pi/2)))
        rr_shadow, cc_shadow = disk((shadow_center_y, shadow_center_x), shadow_radius, shape=(size, size))
        img_array_out[rr_shadow, cc_shadow] = 0 # 确保阴影区域为纯黑
    
    # 添加全局噪声 (保持不变)
    img_array_out += np.random.normal(0, random.uniform(5, 15), (size, size))
    img_array_out = np.clip(img_array_out, 0, 255).astype(np.uint8)
    return Image.fromarray(img_array_out).convert('RGB')

def _create_single_non_bh_image(size):
    """
    Generates a single synthetic image that does NOT represent a black hole.
    This is used to create negative samples for classification.
    NEW: 简化非黑洞图像，使其更具结构性而非纯随机。
    """
    img = Image.new('L', (size, size), color=random.randint(0, 100)) # 调整背景颜色范围，更深
    draw = ImageDraw.Draw(img)
    num_shapes = random.randint(3, 10) # <--- 减少形状数量，使图像更简洁
    for _ in range(num_shapes):
        shape_type = random.choice(['circle', 'rect']) # <--- 限制形状类型，移除 line/polygon，避免过于复杂
        color = random.randint(50, 200) # 调整形状颜色范围
        if shape_type == 'circle':
            radius = random.randint(size // 8, size // 3) # 调整尺寸范围
            center_x, center_y = random.randint(0, size-1), random.randint(0, size-1)
            rr_circle, cc_circle = disk((center_y, center_x), radius, shape=(size, size))
            img_array_temp = np.array(img)
            img_array_temp[rr_circle, cc_circle] = color
            img = Image.fromarray(img_array_temp)
        elif shape_type == 'rect':
            p1_x, p1_y = random.randint(0, size-1), random.randint(0, size-1)
            p2_x, p2_y = random.randint(0, size-1), random.randint(0, size-1)
            draw.rectangle([(min(p1_x,p2_x), min(p1_y,p2_y)), (max(p1_x,p2_x), max(p1_y,p2_y))], fill=color)
        # 移除 line 和 polygon 的生成逻辑

    img_array_out = np.array(img, dtype=np.float32)
    img_array_out = gaussian_filter(img_array_out, sigma=random.uniform(1.0, 3.0)) # 增加一些平滑
    img_array_out += np.random.normal(0, random.uniform(5, 15), (size, size)) # 减少噪声强度
    img_array_out = np.clip(img_array_out, 0, 255).astype(np.uint8)
    return Image.fromarray(img_array_out).convert('RGB')


# --- (其他数据生成函数 _generate_gw_waveform_simplified, _generate_x_ray_curve_simplified, _generate_3d_volume_simplified 保持不变) ---
def _generate_gw_waveform_simplified(timesteps, mass, spin, distance, is_white_hole=False, is_formation_stage=False):
    """ V2: Increased signal distinction """
    t_vals = np.linspace(-1, 0, timesteps)

    if is_formation_stage:
        # Strong, noisy burst followed by a clear ringdown
        waveform = np.zeros(timesteps)
        burst_time = random.uniform(-0.6, -0.4)
        burst_amp = (mass/40.0) * random.uniform(10, 20) / (distance/100 + 1e-6)
        # Main burst with high-frequency noise
        waveform += burst_amp * np.exp(-((t_vals - burst_time)**2) / (random.uniform(0.005, 0.01)**2)) * np.sin(2*np.pi*random.uniform(150,300)*t_vals)
        # Add a clear ringdown phase
        ringdown_freq = 120 + (mass/25) * random.uniform(0.9,1.1)
        ringdown_q = random.uniform(3,6)
        ringdown_amp = burst_amp * random.uniform(0.2, 0.4)
        ringdown_start = burst_time + 0.1
        mask = t_vals > ringdown_start
        waveform[mask] += ringdown_amp * np.exp(-(t_vals[mask] - ringdown_start)/(ringdown_q/(ringdown_freq + 1e-6))) * \
                           np.sin(2*np.pi*ringdown_freq* (t_vals[mask] - ringdown_start))
        noise_level = random.uniform(0.15, 0.3)
    elif is_white_hole:
        # Time-reversed chirp: frequency decreases, ends in a 'pop'
        f_start = 250 + mass * 4 + random.uniform(-20,20)
        f_end = 20 + spin * 10 + random.uniform(-5,5)
        frequency = f_end + (f_start - f_end) * np.exp(t_vals * 4) # Exponential decay of frequency
        amplitude = np.exp(t_vals * 2.5) / (distance / 100 + 1e-6)
        waveform = amplitude * np.sin(2 * np.pi * np.cumsum(frequency) * (t_vals[1]-t_vals[0]))
        # Add a final 'pop'
        waveform += np.exp(- (t_vals**2) / 0.001) * 1.5 / (distance/100 + 1e-6)
        noise_level = random.uniform(0.05, 0.1)
    else: # Mature Black Hole
        # Standard chirp: frequency and amplitude increase
        f_min = 20 + spin * 5
        f_max = 300 + mass * 8
        frequency = f_min + (f_max - f_min) * (t_vals + 1)**2.5 # Steeper increase
        amplitude = (t_vals + 1)**2.5 / (distance / 100 + 1e-6)
        waveform = amplitude * np.sin(2 * np.pi * np.cumsum(frequency) * (t_vals[1]-t_vals[0]))
        noise_level = random.uniform(0.05, 0.15)

    return waveform + np.random.normal(0, noise_level, timesteps)

def _generate_x_ray_curve_simplified(timesteps, mass, spin, is_white_hole=False, is_formation_stage=False):
    """ V3: Drastically simplified and distinguished logic to ensure learnability. """
    t_vals = np.linspace(0, 10, timesteps)
    curve = np.zeros_like(t_vals)
    
    base_level = random.uniform(0.8, 1.2)
    noise_level = random.uniform(0.1, 0.2)

    if is_formation_stage:
        # A single, massive, broad flare representing the initial collapse/supernova.
        peak_time = random.uniform(2, 5)
        peak_amp = random.uniform(40, 60) + mass / 10
        width = random.uniform(2, 4)
        curve = peak_amp * np.exp(-((t_vals - peak_time)**2) / width**2)
        noise_level *= 1.5

    elif is_white_hole:
        # A sharp, exponentially decaying flare. The "anti-accretion" event.
        peak_time = random.uniform(1, 8)
        peak_amp = random.uniform(15, 30) + mass / 5
        decay_rate = random.uniform(0.5, 1.5)
        mask = t_vals >= peak_time
        curve[mask] = peak_amp * np.exp(-(t_vals[mask] - peak_time) / decay_rate)
        
    else: # Mature Black Hole (Standard Accretion)
        # Quasi-periodic oscillations (QPOs) modeled as a sum of sine waves with noise.
        base_level += mass / 20
        num_qpos = random.randint(2, 5)
        for _ in range(num_qpos):
            qpo_amp = random.uniform(0.5, 2.0) * (1 + spin)
            qpo_freq = random.uniform(0.5, 5.0)
            qpo_phase = random.uniform(0, 2 * np.pi)
            curve += qpo_amp * np.sin(2 * np.pi * qpo_freq * t_vals + qpo_phase)**2

    final_curve = base_level + curve + np.random.normal(0, noise_level, timesteps)
    return np.maximum(0, final_curve)

def _generate_3d_volume_simplified(size, mass, is_white_hole=False, is_formation_stage=False):
    """
    Generates a simplified 3D volumetric data cube.
    """
    volume = np.zeros((size, size, size), dtype=np.float32)
    center = np.array([size // 2, size // 2, size // 2])

    if is_formation_stage:
        core_radius = size // 4 + mass / 30 + random.uniform(-size*0.05, size*0.05)
        for x_coord in range(size):
            for y_coord in range(size):
                for z_coord in range(size):
                    dist_from_center = np.linalg.norm(np.array([x_coord, y_coord, z_coord]) - center)
                    if dist_from_center < core_radius:
                        volume[x_coord, y_coord, z_coord] = 200 + random.uniform(-50, 55)

        num_shells = random.randint(1,3)
        for i in range(num_shells):
            shell_radius_base = core_radius * (1.2 + i*0.5)
            shell_radius = random.uniform(shell_radius_base, shell_radius_base + size*0.1)
            shell_thickness = random.uniform(size*0.02, size*0.05)
            shell_density = random.uniform(50, 150) * (0.8**i)
            for x_coord in range(size):
                for y_coord in range(size):
                    for z_coord in range(size):
                        dist = np.linalg.norm(np.array([x_coord, y_coord, z_coord]) - center)
                        if shell_radius - shell_thickness < dist < shell_radius + shell_thickness:
                            volume[x_coord,y_coord,z_coord] = max(volume[x_coord,y_coord,z_coord], shell_density * (1 - random.random()*0.3) )
        noise_level = random.uniform(5,15)
    elif is_white_hole:
        core_radius = size // 8 + mass / 20 + random.uniform(-2,2)
        ejection_strength = (mass / 100) * 50 + random.uniform(-10,10)

        if random.random() < 0.5:
            jet_width = random.uniform(size * 0.05, size * 0.1)
            jet_axis = random.choice([0,1,2])
            c_x, c_y, c_z = center.astype(int)
            j_w_half = int(jet_width/2)
            if jet_axis == 0:
                volume[c_x-j_w_half : c_x+j_w_half, :, :] += ejection_strength * random.uniform(0.5, 1.5) * 0.5
            elif jet_axis == 1:
                volume[:, c_y-j_w_half : c_y+j_w_half, :] += ejection_strength * random.uniform(0.5, 1.5) * 0.5
            else:
                volume[:, :, c_z-j_w_half : c_z+j_w_half] += ejection_strength * random.uniform(0.5, 1.5) * 0.5


        for x_coord in range(size):
            for y_coord in range(size):
                for z_coord in range(size):
                    dist = np.linalg.norm(np.array([x_coord, y_coord, z_coord]) - center)
                    if dist < core_radius:
                        volume[x_coord, y_coord, z_coord] = 255
                    else:
                        volume[x_coord, y_coord, z_coord] += np.exp(-(dist - core_radius)**2 / random.uniform(size*0.1,size*0.3)**2) * ejection_strength
        noise_level = random.uniform(3,8)
    else:
        hole_radius = size // 5 + mass / 50 + random.uniform(-3,3)
        for x_coord in range(size):
            for y_coord in range(size):
                for z_coord in range(size):
                    dist = np.linalg.norm(np.array([x_coord, y_coord, z_coord]) - center)
                    if dist < hole_radius:
                        volume[x_coord, y_coord, z_coord] = 0
                    else:
                        volume[x_coord, y_coord, z_coord] = ((dist - hole_radius) / (size*0.5 - hole_radius + 1e-6)) * 150
                        if random.random() < 0.05:
                             volume[x_coord, y_coord, z_coord] *= random.uniform(1.1, 1.5)
        noise_level = random.uniform(3,8)

    return np.clip(volume + np.random.normal(0, noise_level, volume.shape), 0, 255)

def generate_paired_multimodal_bh_data(n_samples=400, img_size=128, gw_timesteps=256, x_ray_timesteps=100, volumetric_size=32):
    """
    Generates paired multimodal synthetic data for black holes, white holes, formation events, and non-BH objects.
    Includes classification labels (0: Non-BH, 1: BH, 2: WH, 3: BH Formation) and a regression target (energy release index).
    NEW: Integrates data augmentation for time series and volumetric data.
    """
    print(f"Generating {n_samples} paired multimodal samples...")
    all_images, all_tabular_params, all_gw_waveforms, all_x_ray_curves, all_3d_volumes, all_labels, all_regression_targets = [],[],[],[],[],[],[]

    bh_prob = 0.25
    wh_prob = 0.05
    bh_formation_prob = 0.10

    for _ in tqdm(range(n_samples)):
        event_type_rand = random.random()

        mass = random.uniform(5, 100)
        spin = random.uniform(0, 0.99)
        inclination = random.uniform(0, np.pi/2)
        distance_gpc = random.uniform(0.1, 10)

        energy_release_index = (mass**1.5 * spin) / (distance_gpc * 10 + 1) + random.uniform(0, 2)

        gw_data_raw, xray_data_raw, vol_data_raw = None, None, None

        if event_type_rand < bh_prob:
            current_label = 1
            img_data = _create_single_bh_image_advanced(img_size, mass, spin, inclination, is_formation_stage=False)
            gw_data_raw = _generate_gw_waveform_simplified(gw_timesteps, mass, spin, distance_gpc, is_white_hole=False, is_formation_stage=False)
            xray_data_raw = _generate_x_ray_curve_simplified(x_ray_timesteps, mass, spin, is_white_hole=False, is_formation_stage=False)
            vol_data_raw = _generate_3d_volume_simplified(volumetric_size, mass, is_white_hole=False, is_formation_stage=False)
            tab_data = np.array([mass, spin, inclination, distance_gpc, random.uniform(7, 10)])
            energy_release_index += 5.0
        elif event_type_rand < bh_prob + wh_prob:
            current_label = 2
            img_data_bh_like = _create_single_bh_image_advanced(img_size, mass, spin, inclination, is_formation_stage=False)
            img_data_np = np.array(img_data_bh_like).astype(np.float32)
            img_data_np = np.clip(img_data_np * 1.2 + random.uniform(20,50), 0, 255)
            img_data_np[img_data_np < 10] = random.uniform(5, 30)
            img_data = Image.fromarray(img_data_np.astype(np.uint8))

            gw_data_raw = _generate_gw_waveform_simplified(gw_timesteps, mass, spin, distance_gpc, is_white_hole=True, is_formation_stage=False)
            xray_data_raw = _generate_x_ray_curve_simplified(x_ray_timesteps, mass, spin, is_white_hole=True, is_formation_stage=False)
            vol_data_raw = _generate_3d_volume_simplified(volumetric_size, mass, is_white_hole=True, is_formation_stage=False)
            tab_data = np.array([mass, spin, inclination, distance_gpc, random.uniform(8, 10)])
            energy_release_index += 10.0
        elif event_type_rand < bh_prob + wh_prob + bh_formation_prob:
            current_label = 3
            progenitor_mass = random.uniform(20, 150)
            formed_bh_mass = progenitor_mass * random.uniform(0.1, 0.3)
            formed_bh_spin = random.uniform(0, 0.7)

            img_data = _create_single_bh_image_advanced(img_size, progenitor_mass, formed_bh_spin, inclination, is_formation_stage=True)
            gw_data_raw = _generate_gw_waveform_simplified(gw_timesteps, progenitor_mass, formed_bh_spin, distance_gpc, is_white_hole=False, is_formation_stage=True)
            xray_data_raw = _generate_x_ray_curve_simplified(x_ray_timesteps, progenitor_mass, formed_bh_spin, is_white_hole=False, is_formation_stage=True)
            vol_data_raw = _generate_3d_volume_simplified(volumetric_size, progenitor_mass, is_white_hole=False, is_formation_stage=True)
            tab_data = np.array([progenitor_mass, formed_bh_spin, inclination, distance_gpc, random.uniform(5, 8)])
            energy_release_index += random.uniform(15, 50)
        else:
            current_label = 0
            img_data = _create_single_non_bh_image(img_size)
            gw_data_raw = np.random.normal(0, 0.01, gw_timesteps)
            xray_data_raw = np.random.normal(1, 0.2, x_ray_timesteps) + np.abs(np.sin(np.linspace(0,10,x_ray_timesteps)*random.uniform(0.5,2))*random.uniform(0.1,0.5))
            vol_data_raw = np.random.rand(volumetric_size, volumetric_size, volumetric_size) * random.uniform(50, 150)
            tab_data = np.array([random.uniform(0.5, 5), random.uniform(0, 0.1), random.uniform(0, np.pi/2),
                                 random.uniform(0.01, 0.5), random.uniform(0, 6)])
            energy_release_index = np.clip(energy_release_index, 0, 5)

        # --- Apply data augmentation to time series and volumetric data (NEW) ---
        if gw_data_raw is not None and random.random() < 0.7:
            gw_data = augment_time_series_data(gw_data_raw)
        else:
            gw_data = gw_data_raw

        if xray_data_raw is not None and random.random() < 0.7:
            xray_data = xray_data = xray_data_raw
        else:
            xray_data = xray_data_raw

        if vol_data_raw is not None and random.random() < 0.7:
            vol_data = augment_volumetric_data(vol_data_raw)
        else:
            vol_data = vol_data_raw
        # --- End Data Augmentation ---

        all_images.append(np.array(img_data))
        all_tabular_params.append(tab_data)
        all_gw_waveforms.append(gw_data)
        all_x_ray_curves.append(xray_data)
        all_3d_volumes.append(vol_data)
        all_labels.append(current_label)
        all_regression_targets.append(energy_release_index)

    images_np = np.array(all_images)
    tabular_np = np.array(all_tabular_params)
    gw_np = np.array(all_gw_waveforms)
    xray_np = np.array(all_x_ray_curves)
    vol_np = np.array(all_3d_volumes)
    labels_np = np.array(all_labels)
    regression_targets_np = np.array(all_regression_targets)

    np.random.seed(GlobalConfig.RANDOM_SEED)
    p = np.random.permutation(len(images_np))
    images_np, tabular_np, gw_np, xray_np, vol_np, labels_np, regression_targets_np = \
        images_np[p], tabular_np[p], gw_np[p], xray_np[p], vol_np[p], labels_np[p], regression_targets_np[p]

    print(f"Generated {len(images_np)} paired samples.")
    print(f"Labels counts: Non-BH({np.sum(labels_np == 0)}), BH({np.sum(labels_np == 1)}), WH({np.sum(labels_np == 2)}), BH_Formation({np.sum(labels_np == 3)})")
    print(f"Regression targets (Energy Release) range: {regression_targets_np.min():.2f} - {regression_targets_np.max():.2f}")

    if not os.path.exists(GlobalConfig.DATA_DIR): os.makedirs(GlobalConfig.DATA_DIR)
    np.savez(os.path.join(GlobalConfig.DATA_DIR, 'paired_multimodal_bh_data.npz'),
             images=images_np,
             tabular=tabular_np,
             gw=gw_np,
             xray=xray_np,
             volumetric=vol_np,
             labels=labels_np,
             regression_targets=regression_targets_np)
    print(f"Paired multimodal data saved to {GlobalConfig.DATA_DIR}/paired_multimodal_bh_data.npz")
    return images_np, tabular_np, gw_np, xray_np, vol_np, labels_np, regression_targets_np

def generate_synthetic_physics_sim_data(n_simulations=10, n_timesteps=200, filename=None):
    if filename is None:
        filename = os.path.join(GlobalConfig.DATA_DIR, 'sim_physics_data.npy')

    print(f"Generating {n_simulations} synthetic physics simulations with {n_timesteps} timesteps each...")
    all_sim_data = []
    current_random_state = np.random.RandomState(GlobalConfig.RANDOM_SEED + int(np.random.rand() * 10000))
    
    for i in range(n_simulations):
        initial_pos = current_random_state.uniform(0.5, 1.5)
        initial_vel = current_random_state.uniform(-0.1, 0.1)
        damping_factor = current_random_state.uniform(0.01, 0.05)
        spring_const = current_random_state.uniform(0.1, 0.5)
        dt = 0.1

        positions = [initial_pos]
        current_pos = initial_pos
        current_vel = initial_vel

        for _ in range(n_timesteps - 1):
            force = -spring_const * current_pos - damping_factor * current_vel
            acceleration = force
            current_vel += acceleration * dt
            current_pos += current_vel * dt
            current_pos += current_random_state.normal(0, 0.005)
            positions.append(current_pos)
        all_sim_data.append(positions)

    sim_data_np = np.array(all_sim_data)
    if not os.path.exists(GlobalConfig.DATA_DIR): os.makedirs(GlobalConfig.DATA_DIR)
    np.save(filename, sim_data_np)
    print(f"Synthetic physics simulation data saved to {filename}")
    return sim_data_np


def generate_simulated_data_stream(num_batches=10, samples_per_batch=20,
                                   img_size=128, gw_timesteps=256, x_ray_timesteps=100, volumetric_size=32):
    print(f"\n--- Simulating Real-time Data Stream ({num_batches} batches, {samples_per_batch} samples/batch) ---")
    for batch_idx in range(num_batches):
        print(f"Generating Stream Batch {batch_idx + 1}/{num_batches}")
        images, tabular, gw, xray, volumetric, labels, regression_targets = \
            generate_paired_multimodal_bh_data(
                n_samples=samples_per_batch,
                img_size=img_size,
                gw_timesteps=gw_timesteps,
                x_ray_timesteps=x_ray_timesteps,
                volumetric_size=volumetric_size
            )
        yield {
            'images': images,
            'tabular': tabular,
            'gw': gw,
            'xray': xray,
            'volumetric': volumetric,
            'labels': labels,
            'regression_targets': regression_targets
        }

# --- Plotting Functions ---
def plot_history(history, filename=None):
    if filename is None:
        filename = os.path.join(GlobalConfig.RESULTS_DIR, "training_history.png")
    if not os.path.exists(GlobalConfig.RESULTS_DIR): os.makedirs(GlobalConfig.RESULTS_DIR)
    plt.figure(figsize=(12, 5))

    acc_key = None
    if 'accuracy' in history.history: acc_key = 'accuracy'
    elif 'classification_output_accuracy' in history.history: acc_key = 'classification_output_accuracy'

    if acc_key:
        plt.subplot(1, 2, 1)
        plt.plot(history.history[acc_key], label='Training Accuracy')
        if f'val_{acc_key}' in history.history:
            plt.plot(history.history[f'val_{acc_key}'], label='Validation Accuracy')
        plt.title('Model Accuracy'); plt.ylabel('Accuracy'); plt.xlabel('Epoch'); plt.legend()

    plt.subplot(1, 2, 2 if acc_key else 1)
    loss_key_base = 'Overall Loss'
    if 'classification_output_loss' in history.history:
        plt.plot(history.history['classification_output_loss'], label='Training Classification Loss')
        if 'val_classification_output_loss' in history.history:
            plt.plot(history.history['val_classification_output_loss'], label='Validation Classification Loss')
        if 'regression_output_loss' in history.history:
            plt.plot(history.history['regression_output_loss'], label='Training Regression Loss')
            if 'val_regression_output_loss' in history.history:
                plt.plot(history.history[f'val_{acc_key}'], label='Validation Regression Loss') # Changed from val_accuracy to val_regression_loss
        loss_key_base = 'Overall Loss'
    else:
        plt.plot(history.history['loss'], label='Training Loss')
        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'], label='Validation Loss')

    plt.title(f'Model Loss ({loss_key_base})'); plt.ylabel('Loss'); plt.xlabel('Epoch'); plt.legend()
    plt.tight_layout(); plt.savefig(filename); plt.close()

def plot_confusion_matrix(cm, classes, filename=None):
    if filename is None:
        filename = os.path.join(GlobalConfig.RESULTS_DIR, "confusion_matrix.png")
    if not os.path.exists(GlobalConfig.RESULTS_DIR): os.makedirs(GlobalConfig.RESULTS_DIR)
    plt.figure(figsize=(max(8, len(classes)), max(6, len(classes)*0.8)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix'); plt.ylabel('Actual Label'); plt.xlabel('Predicted Label')
    plt.savefig(filename); plt.close()

def plot_sim_prediction(actual, predicted, filename=None):
    if filename is None:
        filename = os.path.join(GlobalConfig.RESULTS_DIR, "sim_prediction_plot.png")
    if not os.path.exists(GlobalConfig.RESULTS_DIR): os.makedirs(GlobalConfig.RESULTS_DIR)
    plt.figure(figsize=(10, 6))
    plt.plot(actual, label='Actual Trajectory', alpha=0.7)
    plt.plot(predicted, label='Predicted Trajectory', linestyle='--', alpha=0.7)
    plt.title('Simplified Physics Simulation: Actual vs. Predicted Trajectory'); plt.xlabel('Timestep'); plt.ylabel('Position'); plt.legend(); plt.grid(True)
    plt.savefig(filename); plt.close()

def plot_sample_images(images, labels, filename=None, num_samples=10):
    if filename is None:
        filename = os.path.join(GlobalConfig.RESULTS_DIR, "sample_bh_images.png")
    if not os.path.exists(GlobalConfig.RESULTS_DIR): os.makedirs(GlobalConfig.RESULTS_DIR)
    
    label_map = GlobalConfig.LABELS_MAP
    
    actual_num_samples = min(num_samples, len(images))
    if actual_num_samples == 0: return

    cols = min(actual_num_samples, num_samples // 2 if num_samples // 2 > 0 else 1)
    rows = (actual_num_samples + cols - 1) // cols

    plt.figure(figsize=(cols * 2.5, rows * 2.5))
    for i in range(actual_num_samples):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images[i])
        plt.title(f"{label_map.get(labels[i], f'Unknown ({labels[i]})')}")
        plt.axis('off')
    plt.suptitle(f"Sample Synthetic Images ({images.shape[1]}x{images.shape[2]})")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.savefig(filename); plt.close()

def plot_3d_data_slices(data_cube, filename=None):
    if filename is None:
        filename = os.path.join(GlobalConfig.RESULTS_DIR, "sim_3d_slices.png")
    if not os.path.exists(GlobalConfig.RESULTS_DIR): os.makedirs(GlobalConfig.RESULTS_DIR)
    size = data_cube.shape[0]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    slice_x = data_cube[size // 2, :, :]
    slice_y = data_cube[:, size // 2, :]
    slice_z = data_cube[:, :, size // 2]

    axes[0].imshow(slice_x, cmap='viridis', aspect='auto'); axes[0].set_title(f'Slice X={size // 2}'); axes[0].axis('off')
    axes[1].imshow(slice_y, cmap='viridis', aspect='auto'); axes[1].set_title(f'Slice Y={size // 2}'); axes[1].axis('off')
    axes[2].imshow(slice_z, cmap='viridis', aspect='auto'); axes[2].set_title(f'Slice Z={size // 2}'); axes[2].axis('off')
    
    fig.suptitle("3D Synthetic Data Cube Slices"); plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.savefig(filename); plt.close()

def plot_3d_data_interactive(data_cube, title="Interactive 3D Volume", opacity_threshold=0.1):
    if not os.path.exists(GlobalConfig.RESULTS_DIR): os.makedirs(GlobalConfig.RESULTS_DIR)

    if not isinstance(data_cube, np.ndarray) or data_cube.ndim != 3:
        print(f"plot_3d_data_interactive Error: data_cube is not a 3D numpy array. Shape: {data_cube.shape}")
        return

    grid = pv.ImageData()
    grid.dimensions = data_cube.shape
    grid.spacing = (1, 1, 1)
    grid.origin = (0, 0, 0)
    grid.point_data["values"] = data_cube.flatten(order="F")

    plotter = pv.Plotter(window_size=[800, 800])

    data_min, data_max = grid.active_scalars.min(), grid.active_scalars.max()
    if data_min == data_max:
        opacity_transfer_function = [data_min, 0.1, data_max, 0.1]
    else:
        opacity_transfer_function = [
            data_min, 0.0,
            data_min + (data_max - data_min) * 0.1, 0.01,
            data_min + (data_max - data_min) * 0.3, 0.1,
            data_min + (data_max - data_min) * 0.7, 0.4,
            data_max, 0.8
        ]
    try:
        plotter.add_volume(grid, cmap='viridis', opacity=opacity_transfer_function, shade=True)
    except TypeError:
        print("PyVista TypeError in add_volume with list of pairs opacity. Attempting fallback opacity (list of floats)...")
        simple_opacity = [0.0, 0.01, 0.1, 0.4, 0.8]
        try:
            plotter.add_volume(grid, cmap='viridis', opacity=simple_opacity, shade=True)
        except Exception as e2:
            print(f"Fallback opacity also failed: {e2}. Using default opacity.")
            plotter.add_volume(grid, cmap='viridis', shade=True)
    except Exception as e_generic:
        print(f"An unexpected error occurred during PyVista volume rendering: {e_generic}. Using default opacity.")
        plotter.add_volume(grid, cmap='viridis', shade=True)

    plotter.add_text(title, position='upper_left', color='white', font_size=16)
    print(f"Opening interactive 3D plot window for '{title}'. Close the window to continue program execution.")
    plotter.show()
    print(f"Interactive 3D plot for '{title}' displayed. Program continues.")


def plot_sample_multimodal_event(image, tabular, gw_waveform, xray_curve, volumetric, label, filename=None, regression_target=None):
    if filename is None:
        filename = os.path.join(GlobalConfig.RESULTS_DIR, "sample_multimodal_event.png")
    if not os.path.exists(GlobalConfig.RESULTS_DIR): os.makedirs(GlobalConfig.RESULTS_DIR)
    
    label_map = GlobalConfig.LABELS_MAP
    fig = plt.figure(figsize=(16, 9));
    fig.suptitle(f"Sample Multimodal Event - Type: {label_map.get(label, f'Unknown ({label})')}", fontsize=16)

    ax1 = fig.add_subplot(2, 3, 1); ax1.imshow(image); ax1.set_title("Image"); ax1.axis('off')

    ax2 = fig.add_subplot(2, 3, 2);
    param_names = ["Mass/ProgM", "Spin/FormS", "Inclination", "Distance", "X-ray Act."]
    for i, param_val in enumerate(tabular):
        if i < len(param_names):
            ax2.text(0.05, 0.9 - i*0.18, f"{param_names[i]}: {param_val:.2f}", fontsize=9, transform=ax2.transAxes)
        else:
            ax2.text(0.05, 0.9 - i*0.18, f"Param{i+1}: {param_val:.2f}", fontsize=9, transform=ax2.transAxes)

    if regression_target is not None:
        ax2.text(0.05, 0.9 - len(tabular)*0.18, f"Energy Idx: {regression_target:.2f}", fontsize=9, transform=ax2.transAxes, color='blue')
    ax2.set_title("Tabular Features"); ax2.axis('off')

    ax3 = fig.add_subplot(2, 3, 3); ax3.plot(gw_waveform); ax3.set_title("GW Waveform"); ax3.set_xlabel("Timestep"); ax3.set_ylabel("Amplitude"); ax3.grid(True)
    ax4 = fig.add_subplot(2, 3, 4); ax4.plot(xray_curve, color='orange'); ax4.set_title("X-ray Light Curve"); ax4.set_xlabel("Timestep"); ax4.set_ylabel("Brightness"); ax4.grid(True)

    ax5 = fig.add_subplot(2, 3, 5);
    if volumetric is not None and volumetric.ndim == 3 and volumetric.shape[0] > 0 :
        size_vol = volumetric.shape[0]
        ax5.imshow(volumetric[size_vol // 2, :, :], cmap='viridis', aspect='auto');
        ax5.set_title(f"3D Volume (Slice X={size_vol // 2})");
    else:
        ax5.text(0.5, 0.5, "No Volumetric Data", ha='center', va='center')
        ax5.set_title("3D Volume")
    ax5.axis('off')

    ax6 = fig.add_subplot(2, 3, 6); ax6.axis('off'); ax6.text(0.5,0.5, "Reserved Plot", ha='center', va='center')


    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.savefig(filename); plt.close()

def plot_gan_generated_samples(generated_images, epoch, examples=16, filename_prefix="gan_generated_samples_epoch_"):
    if not os.path.exists(GlobalConfig.RESULTS_DIR): os.makedirs(GlobalConfig.RESULTS_DIR)
    fig = plt.figure(figsize=(max(4, int(np.sqrt(examples)*1.5)), max(4, int(np.sqrt(examples)*1.5))))
    
    actual_examples = min(examples, generated_images.shape[0])
    if actual_examples == 0: return

    grid_size = int(np.ceil(np.sqrt(actual_examples)))

    for i in range(actual_examples):
        plt.subplot(grid_size, grid_size, i+1)
        img_to_show_norm = (generated_images[i, :, :, :] * 0.5 + 0.5)
        img_to_show_clipped_float32 = np.clip(img_to_show_norm, 0, 1).astype(np.float32)
        plt.imshow(img_to_show_clipped_float32) 
        plt.axis('off')
    plt.suptitle(f"GAN Generated Samples (Epoch {epoch})")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    full_filename = os.path.join(GlobalConfig.RESULTS_DIR, f"{filename_prefix}{epoch:04d}.png")
    
    plt.savefig(full_filename)
    plt.close()

def plot_uncertainty_selection(images_batch, predictions_batch, selected_indices, filename=None, num_display=10):
    if filename is None:
        filename = os.path.join(GlobalConfig.RESULTS_DIR, "uncertainty_selection_plot.png")
    if not os.path.exists(GlobalConfig.RESULTS_DIR): os.makedirs(GlobalConfig.RESULTS_DIR)

    num_images_to_display = min(num_display, len(selected_indices))
    if num_images_to_display == 0:
        if len(images_batch) > 0 and len(predictions_batch) > 0 :
            num_images_to_display = min(num_display, len(images_batch))
            indices_to_plot = np.arange(num_images_to_display)
            plot_title = "Active Learning: Batch Samples (None Selected)"
        else:
            return
    else:
        indices_to_plot = selected_indices[:num_images_to_display]
        plot_title = "Active Learning: Selected Uncertain Samples"


    cols = min(5, num_images_to_display)
    rows = (num_images_to_display + cols - 1) // cols

    fig = plt.figure(figsize=(cols * 2.5, rows * 3))
    fig.suptitle(plot_title, fontsize=16)

    label_map_short = {
        0: "Non-BH",
        1: "BH",
        2: "WH",
        3: "BH-Form"
    }

    for plot_idx, original_batch_idx in enumerate(indices_to_plot):
        if original_batch_idx >= len(images_batch): continue

        ax = fig.add_subplot(rows, cols, plot_idx + 1)
        ax.imshow(images_batch[original_batch_idx])

        probs_for_sample = predictions_batch[original_batch_idx]
        max_prob_val = np.max(probs_for_sample)
        predicted_class_idx = np.argmax(probs_for_sample)
        predicted_label_str = label_map_short.get(predicted_class_idx, f'Unknown ({predicted_class_idx})')

        title_color = 'red' if original_batch_idx in selected_indices else 'black'
        ax.set_title(f"Idx:{original_batch_idx} Pred: {predicted_label_str}\nConf: {max_prob_val:.2f}", color=title_color, fontsize=8)
        ax.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename); plt.close()
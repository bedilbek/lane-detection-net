NVIDIA_H, NVIDIA_W = 256, 256

CONFIG = {
    'batchsize': 20,
    'epochs': 20,
    'input_width': NVIDIA_W,
    'input_height': NVIDIA_H,
    'input_channels': 1,
    'delta_correction': 0.25,
    'augmentation_steer_sigma': 0.2,
    'augmentation_value_min': 0.2,
    'augmentation_value_max': 1.5,
    'bias': 0.1,
    'crop_start_height_ratio': 0.5,
    'crop_end_height_ratio': 0.084,
}

import sys
import yaml
from argparse import ArgumentParser
from tqdm.auto import tqdm

import imageio
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte
import torch
from sync_batchnorm import DataParallelWithCallback

from modules.generator import OcclusionAwareGenerator
from modules.keypoint_detector import KPDetector
from animate import normalize_kp

import ffmpeg
from os.path import splitext
from shutil import copyfileobj
from tempfile import NamedTemporaryFile

# Ensure compatibility with CPU/GPU without cuda() calls
if sys.version_info[0] < 3:
    raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")


def load_checkpoints(config_path, checkpoint_path, cpu=False):
    # Load model configuration
    with open(config_path) as f:
        config = yaml.full_load(f)

    # Determine device: CPU if cpu=True, else CUDA if available, else CPU
    device = torch.device('cpu' if cpu else ('cuda' if torch.cuda.is_available() else 'cpu'))

    # Initialize generator and keypoint detector, move to device
    generator = OcclusionAwareGenerator(
        **config['model_params']['generator_params'],
        **config['model_params']['common_params']
    ).to(device)
    kp_detector = KPDetector(
        **config['model_params']['kp_detector_params'],
        **config['model_params']['common_params']
    ).to(device)

    # Load checkpoint with appropriate map_location, ignoring missing keys for new layers
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # Use strict=False to allow missing weights for any added layers (e.g., post_conv, post_norm)
    generator.load_state_dict(checkpoint['generator'], strict=False)
    kp_detector.load_state_dict(checkpoint['kp_detector'], strict=False)

    # Wrap in DataParallel only if using GPU
    if device.type == 'cuda':
        generator = DataParallelWithCallback(generator)
        kp_detector = DataParallelWithCallback(kp_detector)

    generator.eval()
    kp_detector.eval()

    return generator, kp_detector


def make_animation(source_image, driving_video, generator, kp_detector,
                   relative=True, adapt_movement_scale=True, cpu=False):
    with torch.no_grad():
        predictions = []
        source = torch.tensor(source_image[np.newaxis].astype(np.float32))
        source = source.permute(0, 3, 1, 2)
        device = next(generator.parameters()).device
        source = source.to(device)

        # Ensure driving_video is list of frames
        if isinstance(driving_video, np.ndarray):
            driving_video = [driving_video]
        driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32))
        driving = driving.permute(0, 4, 1, 2, 3)
        driving = driving.to(device)

        kp_source = kp_detector(source)
        kp_driving_initial = kp_detector(driving[:, :, 0])

        for frame_idx in tqdm(range(driving.shape[2])):
            driving_frame = driving[:, :, frame_idx]
            kp_driving = kp_detector(driving_frame)
            kp_norm = normalize_kp(
                kp_source=kp_source,
                kp_driving=kp_driving,
                kp_driving_initial=kp_driving_initial,
                use_relative_movement=relative,
                use_relative_jacobian=relative,
                adapt_movement_scale=adapt_movement_scale
            )
            # Use positional args to avoid missing kwargs
            out = generator(source, kp_norm, kp_source)
            predictions.append(
                np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0]
            )
    return predictions


def find_best_frame(source, driving, cpu=False):
    import face_alignment  # type: ignore
    from scipy.spatial import ConvexHull

    def normalize_kp_array(kp_array):
        kp = kp_array - kp_array.mean(axis=0, keepdims=True)
        area = ConvexHull(kp[:, :2]).volume
        area = np.sqrt(area)
        kp[:, :2] = kp[:, :2] / area
        return kp

    device = 'cpu' if cpu else ('cuda' if torch.cuda.is_available() else 'cpu')
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType._2D,
        flip_input=True,
        device=device
    )

    kp_source = fa.get_landmarks(255 * source)[0]
    kp_source = normalize_kp_array(kp_source)
    norm = float('inf')
    frame_num = 0
    for i, image in enumerate(driving):
        kp_driving = fa.get_landmarks(255 * image)[0]
        kp_driving = normalize_kp_array(kp_driving)
        new_norm = ((kp_source - kp_driving) ** 2).sum()
        if new_norm < norm:
            norm = new_norm
            frame_num = i
    return frame_num


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--checkpoint", default='vox-cpk.pth.tar', help="path to checkpoint to restore")
    parser.add_argument("--source_image", default='sup-mat/source.png', help="path to source image")
    parser.add_argument("--driving_video", default='')

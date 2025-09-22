import os
import json
import copy
import sys
import importlib
import argparse
import pandas as pd
from easydict import EasyDict as edict
from functools import partial
import numpy as np
import torch
import cv2
from PIL import Image
from tqdm import tqdm


def camera_to_world(points_cam, pose):
    """
    Convert points from camera coordinates to world coordinates
    :params points_cam: [B, N, M, 3] (in use, N is ray number, M is points on a ray)
    :params pose: [B, 4, 4] camera pose matrix
    """
    B, N, M, _ = points_cam.shape
    ones = torch.ones((B, N, M), device=points_cam.device)  # homogeneous
    x_cam = points_cam[:, :, :, 0]
    y_cam = points_cam[:, :, :, 1]
    z_cam = points_cam[:, :, :, 2]
    points_cam_homo = torch.stack((x_cam, y_cam, z_cam, ones), dim=-1)  # [B, N, M, 4]
    
    points_cam_homo = points_cam_homo.reshape(B, -1, 4)  # [B, N*M, 4]
    points_cam_homo = points_cam_homo.permute(0, 2, 1)  # [B, 4, N*M]

    points_world = torch.bmm(pose, points_cam_homo).permute(0, 2, 1)[:, :, :3]  # [B, N*M, 3]

    return points_world.reshape(B, N, M, 3)


def depth_to_pointcloud(depth_map, camera_intrinsics, camera_pose, mask=None):
    """
    Convert depth map to 3D point cloud in world coordinates
    
    Args:
        depth_map: (H, W) depth values
        camera_intrinsics: (3, 3) camera intrinsic matrix
        camera_pose: (4, 4) camera pose matrix (c2w)
        mask: (H, W) optional mask to filter points
    
    Returns:
        points: (N, 3) 3D points in world coordinates
    """
    H, W = depth_map.shape
    device = depth_map.device
    
    # Create pixel coordinates
    y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    
    # Apply mask if provided
    if mask is not None:
        valid_mask = (depth_map > 0) & (mask > 0)
    else:
        valid_mask = depth_map > 0
    
    # Get valid pixels
    valid_y = y[valid_mask]
    valid_x = x[valid_mask]
    valid_depth = depth_map[valid_mask]
    
    if len(valid_depth) == 0:
        return torch.empty((0, 3), device=device)
    
    # Convert to homogeneous coordinates
    pixel_coords = torch.stack([valid_x, valid_y, torch.ones_like(valid_x)], dim=0).float()  # (3, N)
    
    # Back-project to camera coordinates
    camera_coords = torch.inverse(camera_intrinsics) @ pixel_coords  # (3, N)
    camera_coords = camera_coords * valid_depth.unsqueeze(0)  # Scale by depth
    
    # Add homogeneous coordinate
    camera_coords_homo = torch.cat([camera_coords, torch.ones(1, camera_coords.shape[1], device=device)], dim=0)  # (4, N)
    
    # Transform to world coordinates
    world_coords = camera_pose @ camera_coords_homo  # (4, N)
    world_points = world_coords[:3].T  # (N, 3)
    
    return world_points


def load_camera_params(transforms_path):
    """Load camera parameters from transforms.json"""
    with open(transforms_path, 'r') as f:
        data = json.load(f)
    
    frames = data['frames']
    camera_params = []
    
    for frame in frames:
        # Get camera pose (transform matrix)
        transform_matrix = np.array(frame['transform_matrix'])
        camera_pose = torch.from_numpy(transform_matrix).float()
        
        # Get camera intrinsics from fov
        fov = frame['camera_angle_x']
        # Assuming square images and focal length calculation
        # This might need adjustment based on your specific camera model
        H = W = 512  # Default resolution, should match rendered images
        focal_length = W / (2 * np.tan(fov / 2))
        
        # Camera intrinsic matrix
        camera_intrinsics = torch.tensor([
            [focal_length, 0, W/2],
            [0, focal_length, H/2],
            [0, 0, 1]
        ]).float()
        
        camera_params.append({
            'pose': camera_pose,
            'intrinsics': camera_intrinsics,
            'file_path': frame['file_path']
        })
    
    return camera_params


def process_single_object(file_path, sha256, output_dir, **kwargs):
    """
    Process a single object to create partial point cloud
    
    Args:
        file_path: Path to the geo GLB file (not used for point cloud generation)
        sha256: SHA256 hash of the object
        output_dir: Output directory
        
    Returns:
        dict: Result dictionary with sha256 and processed status
    """
    try:
        renders_dir = os.path.join(output_dir, 'renders_geo', sha256)
        pointcloud_dir = os.path.join(output_dir, 'pointcloud')
        os.makedirs(pointcloud_dir, exist_ok=True)
        
        output_file = os.path.join(pointcloud_dir, f'{sha256}.ply')
        
        # Skip if already processed
        if os.path.exists(output_file):
            return {'sha256': sha256, 'pointcloud_processed': True, 'pointcloud_path': f'pointcloud/{sha256}.ply'}
        
        # Check if renders_geo directory exists
        if not os.path.exists(renders_dir):
            print(f"Warning: renders_geo directory not found for {sha256}")
            return {'sha256': sha256, 'pointcloud_processed': False, 'pointcloud_path': None}
        
        # Load camera parameters
        transforms_path = os.path.join(renders_dir, 'transforms.json')
        if not os.path.exists(transforms_path):
            print(f"Warning: transforms.json not found for {sha256}")
            return {'sha256': sha256, 'pointcloud_processed': False, 'pointcloud_path': None}
        
        camera_params = load_camera_params(transforms_path)
        
        # Collect all points from all views
        all_points = []
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        for i, cam_param in enumerate(camera_params):
            # Load depth image
            depth_path = os.path.join(renders_dir, f'{i:03d}_depth.png')
            if not os.path.exists(depth_path):
                continue
                
            # Load mask image (optional)
            mask_path = os.path.join(renders_dir, f'{i:03d}_mask.png')
            mask = None
            if os.path.exists(mask_path):
                mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                mask = torch.from_numpy(mask_img).float().to(device) / 255.0
            
            # Load depth image
            # Note: Depth images from Blender are typically 16-bit PNG
            depth_img = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
            if depth_img is None:
                continue
                
            # Convert depth to meters (adjust scaling as needed)
            # The depth values need to be scaled properly based on your Blender setup
            depth_map = torch.from_numpy(depth_img.astype(np.float32)).to(device)
            
            # Normalize depth values (this might need adjustment based on your setup)
            # Assuming depth is in the range [0, 65535] and needs to be scaled to real world units
            depth_map = depth_map / 65535.0 * 10.0  # Scale to [0, 10] meters, adjust as needed
            
            # Get camera parameters
            camera_pose = cam_param['pose'].to(device)
            camera_intrinsics = cam_param['intrinsics'].to(device)
            
            # Generate point cloud for this view
            points = depth_to_pointcloud(depth_map, camera_intrinsics, camera_pose, mask)
            
            if len(points) > 0:
                all_points.append(points)
        
        if not all_points:
            print(f"Warning: No valid points generated for {sha256}")
            return {'sha256': sha256, 'pointcloud_processed': False, 'pointcloud_path': None}
        
        # Combine all points
        combined_points = torch.cat(all_points, dim=0)
        
        # Convert to numpy for saving
        points_np = combined_points.cpu().numpy()
        
        # Save as PLY file
        save_pointcloud_ply(points_np, output_file)
        
        return {'sha256': sha256, 'pointcloud_processed': True, 'pointcloud_path': f'pointcloud/{sha256}.ply'}
        
    except Exception as e:
        print(f"Error processing {sha256}: {e}")
        return {'sha256': sha256, 'pointcloud_processed': False, 'pointcloud_path': None}


def save_pointcloud_ply(points, filename):
    """Save point cloud as PLY file"""
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        
        for point in points:
            f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")


if __name__ == '__main__':
    dataset_utils = importlib.import_module(f'datasets.{sys.argv[1]}')

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory containing the rendered data')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None,
                        help='Filter objects with aesthetic score lower than this value')
    parser.add_argument('--instances', type=str, default=None,
                        help='Instances to process')
    dataset_utils.add_args(parser)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--max_workers', type=int, default=8)
    
    opt = parser.parse_args(sys.argv[2:])
    opt = edict(vars(opt))

    os.makedirs(os.path.join(opt.output_dir, 'pointcloud'), exist_ok=True)
    
    # get file list
    if not os.path.exists(os.path.join(opt.output_dir, 'metadata.csv')):
        raise ValueError('metadata.csv not found')
    metadata = pd.read_csv(os.path.join(opt.output_dir, 'metadata.csv'))
    
    if opt.instances is None:
        # Filter for objects that have been rendered with geo
        if 'rendered_geo' in metadata.columns:
            metadata = metadata[metadata['rendered_geo'] == True]
        else:
            print("Warning: 'rendered_geo' column not found in metadata. Make sure to run render_geo.py first.")
            # Fallback: check if renders_geo directories exist
            renders_geo_exists = []
            for _, row in metadata.iterrows():
                sha256 = row['sha256']
                renders_geo_dir = os.path.join(opt.output_dir, 'renders_geo', sha256)
                renders_geo_exists.append(os.path.exists(os.path.join(renders_geo_dir, 'transforms.json')))
            metadata = metadata[renders_geo_exists]
        
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score]
        
        # Filter out objects that are already processed
        if 'pointcloud_processed' in metadata.columns:
            metadata = metadata[metadata['pointcloud_processed'] == False]
    else:
        if os.path.exists(opt.instances):
            with open(opt.instances, 'r') as f:
                instances = f.read().splitlines()
        else:
            instances = opt.instances.split(',')
        metadata = metadata[metadata['sha256'].isin(instances)]

    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]
    records = []

    # Filter out objects that are already processed
    for sha256 in copy.copy(metadata['sha256'].values):
        pointcloud_file = os.path.join(opt.output_dir, 'pointcloud', f'{sha256}.ply')
        if os.path.exists(pointcloud_file):
            records.append({
                'sha256': sha256, 
                'pointcloud_processed': True, 
                'pointcloud_path': f'pointcloud/{sha256}.ply'
            })
            metadata = metadata[metadata['sha256'] != sha256]
                
    print(f'Processing {len(metadata)} objects for point cloud generation...')

    # Process objects
    func = partial(process_single_object, output_dir=opt.output_dir)
    
    processed = dataset_utils.foreach_instance(metadata, opt.output_dir, func, max_workers=opt.max_workers, desc='Creating point clouds')
    processed = pd.concat([processed, pd.DataFrame.from_records(records)])
    processed.to_csv(os.path.join(opt.output_dir, f'pointcloud_processed_{opt.rank}.csv'), index=False)

import argparse
import cv2
import numpy as np
import open3d as o3d

def load_images(photo_file, depth_file):
    """
    Load photo and depth images from file paths.
    """
    photo = cv2.imread(photo_file, cv2.IMREAD_COLOR)
    depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)

    if photo is None or depth is None:
        raise ValueError("Error loading images. Please check the file paths and formats.")

    return photo, depth

def compute_3d_points(photo_file, depth_file, depth_scale, max_resolution):
    """
    Loads and resizes images, normalizes the depth map, and generates a 3D point cloud.
    """
    photo, depth = load_images(photo_file, depth_file)

    orig_h, orig_w = photo.shape[:2]

    # Determine the resizing scale
    if max_resolution == -1:  # Keep original resolution
        new_w, new_h = orig_w, orig_h
    else:
        scale = min(max_resolution / orig_w, max_resolution / orig_h, 1.0)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)

    # Resize images if necessary
    if new_w != orig_w or new_h != orig_h:
        photo = cv2.resize(photo, (new_w, new_h))
        depth = cv2.resize(depth, (new_w, new_h))

    # Normalize the depth to the range [0, 1]
    depth_min, depth_max = depth.min(), depth.max()
    depth_norm = (depth.astype(np.float32) - depth_min) / (depth_max - depth_min + 1e-8)

    # Create a grid of pixel coordinates
    xv, yv = np.meshgrid(np.arange(new_w), np.arange(new_h))
    x_flat, y_flat = xv.flatten(), yv.flatten()

    # **Flip Y-axis to correct upside-down issue**
    y_flat = new_h - y_flat  # Invert Y-axis

    # Scale the depth values
    z_scaled = (depth_norm * depth_scale * 100).flatten()

    # Convert photo from BGR to RGB and normalize colors to [0,1] for Open3D
    photo_rgb = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB).reshape(-1, 3)
    colors = photo_rgb.astype(np.float64) / 255.0

    return x_flat, y_flat, z_scaled, colors

def create_point_cloud(x, y, z, colors):
    """
    Creates a point cloud from the computed 3D points and colors.
    """
    points = np.vstack((x, y, z)).T

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd

def main():
    parser = argparse.ArgumentParser(description="Point Cloud Visualization using Open3D")
    parser.add_argument('-i', '--image', default="assets/1.png", help="Path to the photo image (default: 1.png)")
    parser.add_argument('-d', '--depth', default="assets/1_depth.png", help="Path to the depth map image (default: 1_depth.png)")
    parser.add_argument('-s', '--scale', default=1.0, type=float, help="Depth scale value (default: 1.0)")
    parser.add_argument('-r', '--resolution', default=256, type=int,
                        help="Max resolution for processing (default: 256, use -1 for full resolution)")
    args = parser.parse_args()

    # Compute 3D points from the depth and image
    x_flat, y_flat, z_scaled, colors = compute_3d_points(args.image, args.depth, args.scale, args.resolution)

    # Create the point cloud
    pcd = create_point_cloud(x_flat, y_flat, z_scaled, colors)

    print("Displaying point cloud using Open3D...")
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    main()


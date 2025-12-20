"""
TSDF fusion utilities for multi-view depth integration.
"""

import logging
from typing import List, Optional

import numpy as np

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False

logger = logging.getLogger(__name__)


class TSDFFusion:
    """TSDF volume fusion for multi-view depth integration."""
    
    def __init__(
        self,
        voxel_length: float = 0.01,
        sdf_trunc: float = 0.04,
        volume_bounds: Optional[np.ndarray] = None
    ):
        """
        Initialize TSDF volume.
        
        Args:
            voxel_length: Voxel size in meters (default: 1cm)
            sdf_trunc: SDF truncation distance in meters (default: 4cm)
            volume_bounds: Optional volume bounds (min_bound, max_bound)
        """
        if not HAS_OPEN3D:
            raise ImportError("open3d required for TSDF fusion")
        
        self.voxel_length = voxel_length
        self.sdf_trunc = sdf_trunc
        self.volume_bounds = volume_bounds
        
        # Initialize TSDF volume
        self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_length,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )
        
        self.num_frames_integrated = 0
    
    def integrate(
        self,
        depth: np.ndarray,
        rgb: np.ndarray,
        intrinsics: np.ndarray,
        extrinsics: np.ndarray
    ) -> None:
        """
        Integrate depth and RGB frame into TSDF volume.
        
        Args:
            depth: Depth map (H, W) in meters
            rgb: RGB image (H, W, 3) in range [0, 255]
            intrinsics: Camera intrinsics (3, 3)
            extrinsics: Camera extrinsics (4, 4) - world to camera
        """
        # Convert depth to Open3D format
        depth_o3d = o3d.geometry.Image(depth.astype(np.float32))
        
        # Convert RGB to Open3D format
        if rgb.dtype != np.uint8:
            rgb = (rgb * 255).astype(np.uint8)
        rgb_o3d = o3d.geometry.Image(rgb.astype(np.uint8))
        
        # Create RGBD image
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d,
            depth_o3d,
            depth_scale=1.0,  # Depth is already in meters
            depth_trunc=10.0,  # Truncate depth at 10 meters
            convert_rgb_to_intensity=False
        )
        
        # Create camera intrinsics
        height, width = depth.shape
        intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(
            width=width,
            height=height,
            fx=intrinsics[0, 0],
            fy=intrinsics[1, 1],
            cx=intrinsics[0, 2],
            cy=intrinsics[1, 2]
        )
        
        # Integrate into volume
        self.volume.integrate(
            rgbd,
            intrinsic_o3d,
            extrinsics
        )
        
        self.num_frames_integrated += 1
        
        if self.num_frames_integrated % 10 == 0:
            logger.debug(f"Integrated {self.num_frames_integrated} frames")
    
    def extract_mesh(self) -> 'o3d.geometry.TriangleMesh':
        """
        Extract triangle mesh from TSDF volume.
        
        Returns:
            Open3D triangle mesh
        """
        mesh = self.volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        return mesh
    
    def extract_point_cloud(self) -> 'o3d.geometry.PointCloud':
        """
        Extract point cloud from TSDF volume.
        
        Returns:
            Open3D point cloud
        """
        return self.volume.extract_point_cloud()
    
    def reset(self) -> None:
        """Reset TSDF volume."""
        self.volume.reset()
        self.num_frames_integrated = 0


def fuse_depth_maps(
    depth_maps: List[np.ndarray],
    rgb_images: List[np.ndarray],
    intrinsics: List[np.ndarray],
    extrinsics: List[np.ndarray],
    voxel_length: float = 0.01,
    sdf_trunc: float = 0.04,
    **fusion_kwargs
) -> 'o3d.geometry.TriangleMesh':
    """
    Fuse multiple depth maps into a single mesh using TSDF fusion.
    
    Args:
        depth_maps: List of depth maps (H, W) in meters
        rgb_images: List of RGB images (H, W, 3)
        intrinsics: List of camera intrinsics (3, 3) or single intrinsics
        extrinsics: List of camera extrinsics (4, 4)
        voxel_length: Voxel size in meters
        sdf_trunc: SDF truncation distance in meters
        **fusion_kwargs: Additional TSDF fusion parameters
    
    Returns:
        Fused triangle mesh
    """
    if not HAS_OPEN3D:
        raise ImportError("open3d required for TSDF fusion")
    
    if len(depth_maps) != len(rgb_images):
        raise ValueError("Number of depth maps must match number of RGB images")
    
    if len(depth_maps) != len(extrinsics):
        raise ValueError("Number of depth maps must match number of extrinsics")
    
    # Handle single intrinsics or per-frame intrinsics
    if not isinstance(intrinsics, list):
        intrinsics = [intrinsics] * len(depth_maps)
    
    # Initialize TSDF volume
    fusion = TSDFFusion(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        **fusion_kwargs
    )
    
    # Integrate all frames
    logger.info(f"Fusing {len(depth_maps)} depth maps...")
    
    for i, (depth, rgb, K, extr) in enumerate(zip(
        depth_maps, rgb_images, intrinsics, extrinsics
    )):
        try:
            fusion.integrate(depth, rgb, K, extr)
        except Exception as e:
            logger.warning(f"Failed to integrate frame {i}: {e}")
            continue
    
    # Extract mesh
    logger.info("Extracting mesh from TSDF volume...")
    mesh = fusion.extract_mesh()
    
    logger.info(
        f"Fusion complete: {len(mesh.vertices)} vertices, "
        f"{len(mesh.triangles)} triangles"
    )
    
    return mesh


def fuse_depth_maps_multiprocess(
    depth_maps: List[np.ndarray],
    rgb_images: List[np.ndarray],
    intrinsics: List[np.ndarray],
    extrinsics: List[np.ndarray],
    num_workers: int = 4,
    **fusion_kwargs
) -> 'o3d.geometry.TriangleMesh':
    """
    Fuse depth maps using multiple processes.
    
    Note: Due to Open3D limitations, this splits the work into chunks
    and fuses them sequentially, but preprocessing can be parallelized.
    
    Args:
        depth_maps: List of depth maps
        rgb_images: List of RGB images
        intrinsics: List of camera intrinsics
        extrinsics: List of camera extrinsics
        num_workers: Number of worker processes
        **fusion_kwargs: Additional TSDF fusion parameters
    
    Returns:
        Fused triangle mesh
    """
    # For now, fall back to sequential fusion
    # TODO: Implement chunked parallel fusion if needed
    logger.info(
        f"Multi-process fusion requested with {num_workers} workers, "
        "but using sequential fusion (Open3D limitation)"
    )
    
    return fuse_depth_maps(
        depth_maps, rgb_images, intrinsics, extrinsics, **fusion_kwargs
    )


def clean_mesh(
    mesh: 'o3d.geometry.TriangleMesh',
    remove_non_manifold_edges: bool = True,
    remove_degenerate_triangles: bool = True,
    remove_duplicated_vertices: bool = True,
    remove_duplicated_triangles: bool = True,
    cluster_connected_triangles: bool = True,
    min_cluster_size: int = 100
) -> 'o3d.geometry.TriangleMesh':
    """
    Clean and post-process reconstructed mesh.
    
    Args:
        mesh: Input mesh
        remove_non_manifold_edges: Remove non-manifold edges
        remove_degenerate_triangles: Remove degenerate triangles
        remove_duplicated_vertices: Remove duplicate vertices
        remove_duplicated_triangles: Remove duplicate triangles
        cluster_connected_triangles: Remove small disconnected components
        min_cluster_size: Minimum cluster size (triangles)
    
    Returns:
        Cleaned mesh
    """
    if not HAS_OPEN3D:
        raise ImportError("open3d required for mesh cleaning")
    
    logger.info("Cleaning mesh...")
    
    # Remove duplicates
    if remove_duplicated_vertices:
        mesh.remove_duplicated_vertices()
    
    if remove_duplicated_triangles:
        mesh.remove_duplicated_triangles()
    
    # Remove degenerate triangles
    if remove_degenerate_triangles:
        mesh.remove_degenerate_triangles()
    
    # Remove non-manifold edges
    if remove_non_manifold_edges:
        mesh.remove_non_manifold_edges()
    
    # Remove small disconnected components
    if cluster_connected_triangles:
        triangle_clusters, cluster_n_triangles, _ = (
            mesh.cluster_connected_triangles()
        )
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)
        
        # Keep only clusters larger than threshold
        triangles_to_remove = cluster_n_triangles[triangle_clusters] < min_cluster_size
        mesh.remove_triangles_by_mask(triangles_to_remove)
    
    # Recompute normals
    mesh.compute_vertex_normals()
    
    logger.info(
        f"Cleaned mesh: {len(mesh.vertices)} vertices, "
        f"{len(mesh.triangles)} triangles"
    )
    
    return mesh

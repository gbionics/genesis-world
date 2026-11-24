"""Differentiable pinhole camera rendering Genesis rigid visual meshes with PyTorch3D.

This module requires the optional ``pytorch3d`` and a CUDA device. It is not imported by ``genesis`` itself, use
``from genesis.vis.differentiable_camera import DifferentiableCamera``.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F
from pytorch3d.renderer.mesh.rasterize_meshes import rasterize_meshes
from pytorch3d.structures import Meshes

import genesis as gs
from genesis.engine.entities.rigid_entity import RigidEntity

from .camera import Camera


def rgb_to_uint8(rgb):
    """Make a detached uint8 NumPy copy of a float RGB tensor in [0, 1] for display or export."""
    return rgb.detach().clamp(0, 1).mul(255).round().to(torch.uint8).cpu().numpy()


class PinholeIntrinsics(torch.nn.Module):
    """Square-pixel, centered pinhole intrinsics whose only trainable parameter is the log focal length.

    Parameters
    ----------
    res : tuple of int, shape (2,)
        The resolution (width, height) in pixels.
    fov : float
        The initial vertical field of view in degrees.
    """

    def __init__(self, res=(256, 256), fov=70.0):
        super().__init__()
        if len(res) != 2 or any(not isinstance(n, int) or n <= 0 for n in res):
            raise ValueError("res must contain positive integer width and height")
        if not math.isfinite(fov) or not 0 < fov < 180:
            raise ValueError("vertical fov must be finite and between 0 and 180 degrees")
        self.res = tuple(res)
        f = res[1] / (2 * math.tan(math.radians(fov) / 2))
        self.log_f = torch.nn.Parameter(torch.tensor(math.log(f), dtype=torch.float64))

    @property
    def f_px(self):
        return self.log_f.exp()

    @property
    def fov_deg(self):
        return torch.rad2deg(2 * torch.atan(self.res[1] / (2 * self.f_px)))


class RigidMeshRasterizer:
    """PyTorch3D soft rasterization of the opaque, solid-color visual meshes of a scene.

    RGB uses unlit surface colors with soft coverage and depth-weighted foreground blending, composited over the
    background. Genesis supplies detached world-space vertices, explicit tensor vertices and camera transforms keep
    their autograd graphs.
    """

    def __init__(self, scene, supersampling=1):
        if not torch.cuda.is_available():
            raise RuntimeError("Differentiable RGB/depth rendering requires an NVIDIA CUDA GPU.")
        if scene.n_envs != 0:
            raise ValueError("Tensor rendering currently supports a single unbatched scene.")
        if any(not isinstance(entity, RigidEntity) for entity in scene.entities):
            raise ValueError("Tensor rendering currently supports rigid visual meshes only.")
        self.supersampling = supersampling
        self.device = gs.device if gs.device.type == "cuda" else torch.device("cuda:0")
        self.vgeoms = tuple(vgeom for entity in scene.entities for vgeom in entity.vgeoms)
        if not self.vgeoms:
            raise ValueError("Tensor rendering requires at least one visual mesh.")
        self.vertex_indices = []
        faces_list = []
        colors = []
        n_verts = 0
        for vgeom in self.vgeoms:
            rgba = vgeom.surface.get_rgba()
            if not isinstance(rgba, gs.textures.ColorTexture):
                raise ValueError("Tensor rendering supports solid-color surfaces, not image textures.")
            color = tuple(rgba.color)
            if len(color) == 4 and color[3] < 1.0:
                raise ValueError("Tensor rendering currently supports opaque surfaces only.")
            # Weld visual seam duplicates so that tensor vertex overrides share positions across connected faces.
            _, verts_idx, inverse = np.unique(vgeom.init_vverts, axis=0, return_index=True, return_inverse=True)
            self.vertex_indices.append(torch.as_tensor(verts_idx, device=self.device))
            faces_list.append(torch.as_tensor(inverse[vgeom.init_vfaces] + n_verts, device=self.device))
            colors.append(torch.tensor(color[:3], dtype=torch.float32, device=self.device).expand(len(verts_idx), 3))
            n_verts += len(verts_idx)
        self.n_verts = n_verts
        self.faces = torch.cat(faces_list).long()
        self.verts_color = torch.cat(colors)
        self.faces_color = self.verts_color[self.faces].mean(dim=1)
        self.background = torch.tensor(scene.options.vis.background_color, dtype=torch.float32, device=self.device)
        # Fine sphere silhouettes overlap many soft triangle footprints, and a small top-K budget truncates coverage.
        self.faces_per_pixel = 32
        self.edge_width_px = 0.5
        # Smooth competition between surfaces, in scene units.
        self.depth_temperature = 0.01

    def get_verts(self):
        """Fresh detached world-space vertices of the welded visual meshes."""
        return torch.cat(
            [
                vgeom.get_vverts().detach().to(device=self.device, dtype=torch.float32)[verts_idx]
                for vgeom, verts_idx in zip(self.vgeoms, self.vertex_indices)
            ]
        )

    def render(self, focal_length, camera_transform, res, near, far, verts=None, *, rgb=True, depth=False):
        """Render differentiable float32 CUDA buffers.

        Parameters
        ----------
        focal_length : torch.Tensor
            Focal length in pixels.
        camera_transform : torch.Tensor
            Camera-to-world matrix of shape (4, 4) in the OpenGL convention (+X right, +Y up, -Z forward).
        res : tuple of int, shape (2,)
            The resolution (width, height) in pixels.
        near, far : float
            Clipping distances in scene units.
        verts : torch.Tensor | None
            World-space vertices of shape (n_verts, 3) in the welded ordering of `get_verts`. Defaults to a detached
            snapshot of the simulation.
        rgb, depth : bool
            Which buffers to compute.

        Returns
        -------
        image, depth_image : torch.Tensor | None
            RGB of shape (height, width, 3) and the forward-axis distance of shape (height, width), softly blended
            with `far` at silhouettes. Uncovered pixels equal `far`.
        """
        if not (rgb or depth):
            return None, None
        verts = self.get_verts() if verts is None else verts.to(device=self.device, dtype=torch.float32)
        if verts.shape != (self.n_verts, 3):
            raise ValueError(f"verts must have shape ({self.n_verts}, 3)")
        transform = torch.as_tensor(camera_transform, device=self.device, dtype=torch.float32)
        if transform.shape != (4, 4):
            raise ValueError("camera_transform must be a camera-to-world matrix with shape (4, 4)")
        verts_view = (verts - transform[:3, 3]) @ transform[:3, :3]

        focal = focal_length.to(device=self.device, dtype=torch.float32)
        width, height = res
        verts_depth = -verts_view[:, 2]
        # Triangles behind the camera are removed by near-plane clipping, the clamp only avoids a division at zero.
        verts_safe_depth = torch.where(verts_depth >= 0, verts_depth.clamp_min(1e-6), verts_depth.clamp_max(-1e-6))
        # PyTorch3D uses +X left, +Y up, +Z forward, and scales NDC by the shorter image dimension.
        scale = 2 * focal / min(width, height)
        verts_ndc = torch.stack(
            (-verts_view[:, 0] * scale / verts_safe_depth, verts_view[:, 1] * scale / verts_safe_depth, verts_depth),
            dim=-1,
        )
        meshes = Meshes(verts=[verts_ndc], faces=[self.faces])
        image_size = (height * self.supersampling, width * self.supersampling)
        sigma = (2 * self.edge_width_px / min(image_size)) ** 2
        pix_to_face, zbuf, _, dists = rasterize_meshes(
            meshes,
            image_size=image_size,
            faces_per_pixel=self.faces_per_pixel,
            blur_radius=math.log(1 / 1e-4 - 1) * sigma,
            perspective_correct=True,
            clip_barycentric_coords=True,
            z_clip_value=near,
            cull_to_frustum=True,
            cull_backfaces=False,
        )
        is_valid = (pix_to_face >= 0) & (zbuf <= far)
        coverage = torch.sigmoid(-dists / sigma) * is_valid
        alpha = 1 - (1 - coverage).prod(dim=-1, keepdim=True)

        # Normalizing the weights over foreground fragments alone, then compositing with the coverage alpha, keeps the
        # silhouette gradients of uniformly colored objects (normalizing RGB alone would cancel them).
        nearest = torch.where(is_valid, zbuf, torch.inf).min(dim=-1, keepdim=True).values
        nearest = torch.where(is_valid.any(dim=-1, keepdim=True), nearest, 0.0)
        weights = coverage * torch.exp(-(zbuf - nearest).clamp_min(0) / self.depth_temperature)
        weights_sum = weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        buffers = []
        if rgb:
            colors = self.faces_color[pix_to_face.clamp_min(0)]
            # Centering on the nearest color zeroes the residuals of solid-color surfaces, avoiding cancellation.
            color_nearest = colors[..., 0, :]
            residual = (weights[..., None] * (colors - color_nearest[..., None, :])).sum(dim=-2)
            foreground = color_nearest + residual / weights_sum
            buffers.append(alpha * foreground + (1 - alpha) * self.background)
        else:
            buffers.append(None)
        if depth:
            residual = (weights * (zbuf - nearest)).sum(dim=-1, keepdim=True)
            foreground_depth = nearest + residual / weights_sum
            # Adjacent triangles share coverage at their edges. Summing it before the smoothstep brings internal seams
            # to full opacity, whereas the probabilistic union used for RGB leaks far into mesh interiors.
            depth_alpha = coverage.sum(dim=-1, keepdim=True).clamp(0, 1)
            depth_alpha = depth_alpha.square() * (3 - 2 * depth_alpha)
            buffers.append(depth_alpha * foreground_depth + (1 - depth_alpha) * far)
        else:
            buffers.append(None)

        outputs = []
        for buffer in buffers:
            if buffer is not None and self.supersampling > 1:
                buffer = F.avg_pool2d(buffer.permute(0, 3, 1, 2), self.supersampling).permute(0, 2, 3, 1)
            outputs.append(None if buffer is None else buffer[0])
        image, depth_image = outputs
        return image, None if depth_image is None else depth_image[..., 0]


class DifferentiableCamera(Camera):
    """Genesis camera rendering RGB and depth differentiably, with a trainable focal length.

    `render` returns float32 CUDA RGB and depth tensors that retain image gradients with respect to the focal length
    (``optics.log_f``), to the optional `camera_transform` and to the optional world-space `verts` overrides. Default
    poses and vertices are detached snapshots of the simulation. Segmentation and normals come from Genesis and are not
    differentiable. Only pinhole cameras with the Rasterizer, opaque solid-color rigid meshes and a single unbatched
    scene are supported. `render_native` returns the standard Genesis buffers.

    Parameters
    ----------
    supersampling : int
        Integer supersampling factor of the soft rasterization. Defaults to 1.
    """

    def __init__(self, visualizer, *, res=(256, 256), fov=70.0, supersampling=1, **kwargs):
        optics = PinholeIntrinsics(res, fov)
        if not isinstance(supersampling, int) or supersampling < 1:
            raise ValueError("supersampling must be a positive integer")
        if kwargs.get("model", "pinhole") != "pinhole":
            raise ValueError("only pinhole cameras are supported")
        super().__init__(visualizer, res=res, fov=fov, **kwargs)
        self.optics = optics
        self.supersampling = supersampling
        self._mesh_rasterizer = None
        self._is_recording_frame = False

    @classmethod
    def add_to_scene(cls, scene, **kwargs):
        """Create a differentiable camera and register it in the scene. Must be called before `scene.build()`."""
        if scene.is_built:
            raise ValueError("add cameras before scene.build()")
        if scene.visualizer.raytracer is not None or scene.visualizer.batch_renderer is not None:
            raise ValueError("only the Genesis Rasterizer is supported")
        cameras = scene.visualizer.cameras
        kwargs["idx"] = sum(camera.debug == kwargs.get("debug", False) for camera in cameras)
        camera = cls(scene.visualizer, **kwargs)
        cameras.append(camera)
        return camera

    def sync_to_genesis(self):
        """Propagate the current trainable field of view to the native Genesis camera."""
        self._fov = self.optics.fov_deg.detach().cpu().item()
        for name in ("f", "intrinsics", "projection_matrix"):
            self.__dict__.pop(name, None)

    def build(self):
        self.sync_to_genesis()
        super().build()

    def destroy(self):
        self._mesh_rasterizer = None
        super().destroy()

    def render_native(self, **kwargs):
        """Render the standard detached Genesis buffers, including uint8 NumPy RGB."""
        self.sync_to_genesis()
        return super().render(**kwargs)

    def update_recording(self):
        # Genesis video encoders require uint8 CPU frames.
        if not self._is_recording:
            return
        self._is_recording_frame = True
        try:
            with torch.no_grad():
                super().update_recording()
        finally:
            self._is_recording_frame = False

    def render(
        self,
        rgb=True,
        depth=False,
        segmentation=False,
        colorize_seg=False,
        normal=False,
        antialiasing=False,
        force_render=False,
        *,
        camera_transform=None,
        verts=None,
    ):
        """Render the camera view and return (rgb, depth, segmentation, normal).

        RGB and depth are differentiable float32 CUDA tensors of shape (height, width, 3) and (height, width), None
        when not requested. Segmentation and normals are native Genesis arrays, which cannot be requested together
        with the `camera_transform` or `verts` tensor overrides.
        """
        if (camera_transform is not None or verts is not None) and (segmentation or normal):
            raise ValueError("Tensor overrides cannot be combined with native auxiliary buffers.")
        self.sync_to_genesis()
        seg_image = normal_image = None
        if segmentation or normal:
            _, _, seg_image, normal_image = self.render_native(
                rgb=False,
                depth=False,
                segmentation=segmentation,
                colorize_seg=colorize_seg,
                normal=normal,
                antialiasing=antialiasing,
                force_render=force_render,
            )
        image = depth_image = None
        if rgb or depth:
            if self._mesh_rasterizer is None:
                if not self.is_built:
                    raise RuntimeError("Build the scene before rendering.")
                self._mesh_rasterizer = RigidMeshRasterizer(self._visualizer.scene, self.supersampling)
            image, depth_image = self._mesh_rasterizer.render(
                self.optics.f_px,
                self.transform if camera_transform is None else camera_transform,
                self.res,
                self.near,
                self.far,
                verts,
                rgb=rgb,
                depth=depth,
            )
        if self._is_recording_frame and image is not None:
            image = rgb_to_uint8(image)
        return image, depth_image, seg_image, normal_image

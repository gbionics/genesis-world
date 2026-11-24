import math

import pytest
import torch

import genesis as gs

from ..utils.assertions import assert_allclose

pytest.importorskip("pytorch3d")

from genesis.vis.differentiable_camera import DifferentiableCamera  # noqa: E402


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a CUDA device.")
def test_differentiable_camera_focal_length_and_pose(show_viewer):
    RADIUS = 0.5
    DISTANCE = 3.0
    RES = 128

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.0, -4.0, 1.0),
            camera_lookat=(0.0, 0.0, 1.0),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        morph=gs.morphs.Sphere(
            radius=RADIUS,
            pos=(0.0, 0.0, 1.0),
            fixed=True,
        ),
        surface=gs.surfaces.Default(
            color=(1.0, 0.0, 0.0),
        ),
    )
    camera = DifferentiableCamera.add_to_scene(
        scene,
        res=(RES, RES),
        pos=(0.0, -DISTANCE, 1.0),
        lookat=(0.0, 0.0, 1.0),
    )
    scene.build()

    rgb, depth, *_ = camera.render(depth=True)
    assert rgb.requires_grad and depth.requires_grad

    # The center pixel sees the sphere surface, at the distance of the camera minus the radius
    assert_allclose(depth[RES // 2, RES // 2], DISTANCE - RADIUS, atol=1e-2)
    # The silhouette radius follows the exact pinhole projection of a sphere
    f_px = camera.optics.f_px.item()
    radius_px = f_px * RADIUS / math.sqrt(DISTANCE**2 - RADIUS**2)
    area_px = (depth < camera.far - 1.0).sum().item()
    assert_allclose(math.sqrt(area_px / math.pi), radius_px, atol=1.0)

    # The silhouette area is differentiable with respect to the log focal length
    (rgb,) = camera.render()[:1]
    area = (rgb[..., 0] - rgb[..., 1]).sum()
    (grad,) = torch.autograd.grad(area, camera.optics.log_f)
    # The projected area scales with the squared focal length
    assert_allclose(grad, 2.0 * math.pi * radius_px**2, tol=3e-1)

    # The depth is differentiable with respect to the camera pose
    transform = torch.as_tensor(camera.transform, dtype=torch.float32, device=rgb.device).clone()
    transform.requires_grad_(True)
    _, depth_moved, *_ = camera.render(rgb=False, depth=True, camera_transform=transform)
    (grad_pose,) = torch.autograd.grad(depth_moved[RES // 2, RES // 2], transform)
    # Moving the camera along its forward axis (-Z) shortens the depth one to one
    forward = -transform.detach()[:3, 2]
    assert_allclose((grad_pose[:3, 3] * forward).sum(), -1.0, atol=5e-2)

    # Gradient descent on the focal length recovers the focal length of a target view
    target_f_px = 1.2 * f_px
    with torch.no_grad():
        camera.optics.log_f.fill_(math.log(target_f_px))
        rgb_target = camera.render()[0]
        camera.optics.log_f.fill_(math.log(f_px))
    optimizer = torch.optim.Adam([camera.optics.log_f], lr=0.05)
    for _ in range(40):
        optimizer.zero_grad()
        loss = (camera.render()[0] - rgb_target).square().mean()
        loss.backward()
        optimizer.step()
    assert_allclose(camera.optics.f_px, target_f_px, tol=5e-2)

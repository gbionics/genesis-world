from collections.abc import Sequence
from typing import NamedTuple

import torch


class PhysicalParameter(NamedTuple):
    """Select entity-local indices and supply their values as a Torch tensor."""

    indices: Sequence[int] | None
    values: torch.Tensor


class RigidPhysicalParameters(NamedTuple):
    """Hold numeric model parameters in the entity's resolved index order."""

    mass: torch.Tensor
    center_of_mass: torch.Tensor
    inertia: torch.Tensor
    inertial_quat: torch.Tensor
    link_offset: torch.Tensor
    collision_offset: torch.Tensor
    box_size: PhysicalParameter
    sphere_radius: PhysicalParameter
    capsule_size: PhysicalParameter
    cylinder_size: PhysicalParameter
    mesh_scale: PhysicalParameter
    visual_offset: torch.Tensor
    visual_scale: torch.Tensor

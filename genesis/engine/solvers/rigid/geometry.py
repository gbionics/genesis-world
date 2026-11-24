"""Read a rigid geom's current local shape over its immutable imported geometry."""

import quadrants as qd

import genesis as gs

from .data import DynInfo, RigidInfo


@qd.func
def func_geom_pos(i_g: int, i_b: int, dyn_info: DynInfo):
    pos = dyn_info.geoms.pos[i_g]
    if qd.static(len(dyn_info.geoms.design_idx.shape) == 1):
        i_design = dyn_info.geoms.design_idx[i_g]
        if i_design >= 0:
            pos = dyn_info.geoms_design.pos[i_design, i_b]
    return pos


@qd.func
def func_geom_size(i_g: int, i_b: int, dyn_info: DynInfo):
    data = dyn_info.geoms.data[i_g]
    size = gs.qd_vec3(data[0], data[1], data[2])
    if qd.static(len(dyn_info.geoms.design_idx.shape) == 1):
        i_design = dyn_info.geoms.design_idx[i_g]
        if i_design >= 0:
            size = dyn_info.geoms_design.size[i_design, i_b]
    return size


@qd.func
def func_mesh_scale(i_g: int, i_b: int, dyn_info: DynInfo):
    scale = gs.qd_vec3(1.0, 1.0, 1.0)
    if qd.static(len(dyn_info.geoms.design_idx.shape) == 1):
        i_design = dyn_info.geoms.design_idx[i_g]
        if i_design >= 0:
            scale = dyn_info.geoms_design.size[i_design, i_b]
    return scale


@qd.func
def func_geom_local_deformation(i_g: int, i_b: int, dyn_info: DynInfo, to_rest: qd.template()):
    scale = gs.qd_vec3(1.0, 1.0, 1.0)
    capsule_half_length = gs.qd_float(0.0)
    capsule_shift = gs.qd_float(0.0)
    if qd.static(len(dyn_info.geoms.design_idx.shape) == 1):
        i_design = dyn_info.geoms.design_idx[i_g]
        if i_design >= 0:
            geom_type = dyn_info.geoms.type[i_g]
            data = dyn_info.geoms.data[i_g]
            size = dyn_info.geoms_design.size[i_design, i_b]
            src = gs.qd_vec3(data[0], data[1], data[2])
            if geom_type == gs.GEOM_TYPE.MESH:
                src = gs.qd_vec3(1.0, 1.0, 1.0)
            dst = size
            if qd.static(to_rest):
                src, dst = dst, src

            if geom_type == gs.GEOM_TYPE.BOX or geom_type == gs.GEOM_TYPE.MESH:
                scale = dst / src
            elif geom_type == gs.GEOM_TYPE.SPHERE:
                radius_scale = dst[0] / src[0]
                scale = gs.qd_vec3(radius_scale, radius_scale, radius_scale)
            elif geom_type == gs.GEOM_TYPE.CYLINDER or geom_type == gs.GEOM_TYPE.CAPSULE:
                radius_scale = dst[0] / src[0]
                axial_scale = radius_scale
                if geom_type == gs.GEOM_TYPE.CAPSULE:
                    capsule_half_length = 0.5 * src[1]
                    capsule_shift = 0.5 * dst[1] - capsule_half_length * radius_scale
                else:
                    axial_scale = dst[1] / src[1]
                scale = gs.qd_vec3(radius_scale, radius_scale, axial_scale)
    return scale, capsule_half_length, capsule_shift


@qd.func
def func_deform_geom_point(
    point: qd.types.vector(3), scale: qd.types.vector(3), capsule_half_length: float, capsule_shift: float
):
    """Scale a point and translate it along a capsule's axis to resize the segment independently of its radius."""
    result = point * scale
    if capsule_half_length > 0.0:
        axial_fraction = qd.math.clamp(point[2] / capsule_half_length, -1.0, 1.0)
        result[2] = result[2] + axial_fraction * capsule_shift
    return result


@qd.func
def func_geom_local_point(i_g: int, i_b: int, point: qd.types.vector(3), dyn_info: DynInfo, to_rest: qd.template()):
    """Map a local point between the imported shape and the design in one environment."""
    result = point
    if qd.static(len(dyn_info.geoms.design_idx.shape) == 1):
        scale, capsule_half_length, capsule_shift = func_geom_local_deformation(i_g, i_b, dyn_info, to_rest)
        result = func_deform_geom_point(point, scale, capsule_half_length, capsule_shift)
    return result


@qd.func
def func_geom_local_aabb(i_g: int, i_b: int, dyn_info: DynInfo, rigid_info: RigidInfo):
    """Return the lower and upper local bounds of a geom's current shape."""
    lower = rigid_info.geoms_init_AABB[i_g, 0]
    upper = rigid_info.geoms_init_AABB[i_g, 7]
    if qd.static(len(dyn_info.geoms.design_idx.shape) == 1):
        scale, capsule_half_length, capsule_shift = func_geom_local_deformation(i_g, i_b, dyn_info, to_rest=False)
        for i_corner in range(2):
            point = lower if i_corner == 0 else upper
            point = func_deform_geom_point(point, scale, capsule_half_length, capsule_shift)
            if i_corner == 0:
                lower = point
            else:
                upper = point
    return lower, upper

import argparse

import torch

import genesis as gs
from genesis.engine.entities import PhysicalParameter


def main():
    parser = argparse.ArgumentParser(description="Optimize an imported arm length from its motion against a plane.")
    parser.add_argument("-v", "--vis", action="store_true", help="Show the interactive viewer.")
    parser.add_argument("-g", "--gpu", action="store_true", help="Run the simulation on the GPU.")
    parser.add_argument("-s", "--steps", type=int, default=5, help="Number of design updates.")
    args = parser.parse_args()

    gs.init(backend=gs.gpu if args.gpu else gs.cpu)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            requires_grad=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.7, -2.7, 1.4),
            camera_lookat=(0.9, 0.0, 0.2),
        ),
        show_viewer=args.vis,
    )
    scene.add_entity(
        morph=gs.morphs.Plane(),
    )
    robot = scene.add_entity(
        morph=gs.morphs.URDF(
            file="urdf/simple/two_link_arm.urdf",
            pos=(0.0, 0.0, 0.42),
            fixed=True,
            differentiable_parameters=True,
        ),
    )
    scene.build()

    arm, child = robot.get_link("arm1"), robot.get_link("arm2")
    i_arm, i_child = arm.idx_local, child.idx_local
    i_geom = arm.geoms[0].idx - robot.geom_start
    i_vgeom = arm.vgeoms[0].idx - robot.vgeoms[0].idx
    params = robot.get_physical_parameters()
    box_size = params.box_size.values[params.box_size.indices.index(i_geom)]
    length_ref = box_size[0]
    log_length = torch.nn.Parameter(length_ref.log())
    axis = box_size.new_tensor((1.0, 0.0, 0.0))
    optimizer = torch.optim.Adam((log_length,), lr=0.01)

    for i_rollout in range(args.steps):
        scene.reset()
        optimizer.zero_grad()
        length = log_length.exp()
        delta = (length - length_ref) * axis
        robot.bind_physical_parameters(
            center_of_mass=PhysicalParameter((i_arm,), (params.center_of_mass[i_arm] + 0.5 * delta)[None]),
            link_offset=PhysicalParameter((i_child,), (params.link_offset[i_child] + delta)[None]),
            collision_offset=PhysicalParameter((i_geom,), (params.collision_offset[i_geom] + 0.5 * delta)[None]),
            box_size=PhysicalParameter((i_geom,), (box_size + delta)[None]),
            visual_offset=PhysicalParameter((i_vgeom,), (params.visual_offset[i_vgeom] + 0.5 * delta)[None]),
            visual_scale=PhysicalParameter((i_vgeom,), (params.visual_scale[i_vgeom] + delta / length_ref)[None]),
        )
        robot.set_dofs_position((0.24, 0.12))
        for _ in range(2):
            scene.step()

        # Reduce the rebound of the first joint while the second arm contacts the ground.
        loss = robot.get_dofs_velocity()[0].square()
        loss.backward()
        optimizer.step()
        print(f"Rollout {i_rollout + 1}: length={length.item():.4f} m, loss={loss.item():.5g}")


if __name__ == "__main__":
    main()

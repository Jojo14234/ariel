import mujoco
import numpy as np
from mujoco import viewer

from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld


def random_move(model, data) -> None:
    num_joints = model.nu
    hinge_range = np.pi / 2
    rand_moves = np.random.uniform(low=-hinge_range,  # -pi/2
                                   high=hinge_range,  # pi/2
                                   size=num_joints)

    delta = 0.05
    data.ctrl += rand_moves * delta
    data.ctrl = np.clip(data.ctrl, -np.pi / 2, np.pi / 2)



def main():
    mujoco.set_mjcb_control(None)  # DO NOT REMOVE
    world = SimpleFlatWorld()
    gecko_core = gecko()  # DO NOT CHANGE
    world.spawn(gecko_core.spec, spawn_position=[0, 0, 0])

    model = world.spec.compile()
    data = mujoco.MjData(model)
    mujoco.set_mjcb_control(random_move)

    viewer.launch(
        model=model,
        data=data,
    )


if __name__ == "__main__":
    main()



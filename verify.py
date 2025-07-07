import mujoco
import mujoco.viewer

# open xml to edit
model = mujoco.MjModel.
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
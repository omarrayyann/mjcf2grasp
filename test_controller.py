import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)


with mujoco.viewer.launch_passive(
    model, data, show_left_ui=False, show_right_ui=False
) as viewer:
    while viewer.is_running():
        # model.body_pos[
        #     mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper_base")
        # ] = [0, 0, 1]
        mujoco.mj_step(model, data)
        viewer.sync()

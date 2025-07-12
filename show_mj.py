import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("/home/lambda1/Documents/thor-grasp/assets/Thor-Assets/Miscellaneous Objects/TennisRacquet/Prefabs/Tennis_Racquet_5/Tennis_Racquet_5.xml")
data = mujoco.MjData(model)

viewer = mujoco.viewer.launch_passive(model, data)
while viewer.is_running():
    mujoco.mj_step(model, data)
    viewer.sync()
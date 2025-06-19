import time
import numpy as np
import mujoco
import mujoco.viewer
import os
import xml.etree.ElementTree as ET
import argparse
import json
from scipy.spatial.transform import Rotation as R

parser = argparse.ArgumentParser()
parser.add_argument('object', type=str)
mesh_file = os.path.abspath(f"assets/objects/{parser.parse_args().object}.obj")
mesh_name = os.path.splitext(os.path.basename(mesh_file))[0]
object_name = os.path.splitext(os.path.basename(mesh_file))[0]
xml_path = os.path.join(os.path.dirname(__file__), 'assets/scene.xml')
tree = ET.parse(xml_path)
root = tree.getroot()
asset = root.find('asset')
if asset is not None:
    mesh = ET.Element('mesh', {
        'file': mesh_file,
        'scale': '1 1 1'
    })
    asset.append(mesh)
worldbody = root.find('worldbody')
if worldbody is not None:
    body = ET.Element('body', {'name': 'object_body', 'gravcomp': '1'})
    joint = ET.Element('joint', {
        'type': 'free',
        'damping': '10.'
    })
    geom = ET.Element('geom', {
        'name': 'object',
        'type': 'mesh',
        'mesh': object_name,
    })
    body.append(joint)
    body.append(geom)
    worldbody.append(body)

modified_xml_content = ET.tostring(root, encoding='unicode')

json_file_path = f"output/{object_name}_grasps.json"
with open(json_file_path, 'r') as f:
    data = json.load(f)
transforms = np.array(data['transforms'])
quality = np.array(data.get('quality_antipodal', data.get('quality_number_of_contacts', [1.0]*len(transforms))))
top_k = 10
top_indices = np.argsort(quality)[-top_k:][::-1]
transforms = [transforms[i] for i in top_indices]
qualities = [quality[i] for i in top_indices]

i = 0

model = mujoco.MjModel.from_xml_string(modified_xml_content)
data = mujoco.MjData(model)
start_time = None
step = 0.005
with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
    while viewer.is_running():

        if start_time is None or time.time() - start_time > 10:
            mujoco.mj_resetData(model, data)
            i += 1
            i %= len(transforms)
            start_time = time.time()
            transform = transforms[i]
            quality = qualities[i]
            gripper_pos = transform[:3, 3]
            gripper_rot = transform[:3, :3]
            gripper_quat = R.from_matrix(gripper_rot).as_quat(scalar_first=True)
            gripper_quat = gripper_quat
            model.body_pos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'gripper_base')] = transform[:3,3]
            model.body_quat[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'gripper_base')] = gripper_quat
            mujoco.mj_step(model, data)
            data.ctrl[0] = -1.0
            for _ in range(10000):
                mujoco.mj_step(model, data)


        data.ctrl[1] += step

        if data.ctrl[1] > 0.6:
            step = -0.005
        elif data.ctrl[1] < -0.6:
            step = 0.005

        mujoco.mj_step(model, data)
        viewer.sync()
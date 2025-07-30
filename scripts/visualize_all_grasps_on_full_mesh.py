import argparse
import json
import numpy as np
import trimesh
import open3d as o3d
import xml.etree.ElementTree as ET
import os
import trimesh.transformations as tra

def parse_joint_world_poses(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    mujoco = root if root.tag == 'mujoco' else root.find('mujoco')
    world_poses = {}
    body_stack = [(mujoco.find('worldbody'), np.zeros(3), np.array([0,0,0,1]))]
    while body_stack:
        body, parent_pos, parent_quat = body_stack.pop()
        for child in body:
            if child.tag == 'body':
                pos = np.array([float(x) for x in child.attrib.get('pos', '0 0 0').split()]) if 'pos' in child.attrib else np.zeros(3)
                quat = np.array([float(x) for x in child.attrib.get('quat', '0 0 0 1').split()]) if 'quat' in child.attrib else np.array([0,0,0,1])
                abs_pos = parent_pos + tra.quaternion_matrix(parent_quat)[:3,:3].dot(pos)
                abs_quat = tra.quaternion_multiply(parent_quat, quat)
                for j in child.findall('joint'):
                    joint_name = j.attrib['name']
                    joint_pos = abs_pos.copy()
                    joint_quat = abs_quat.copy()
                    world_poses[joint_name] = (joint_pos, joint_quat)
                body_stack.append((child, abs_pos, abs_quat))
    return world_poses

def plot_mesh(mesh, color=None):
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
    o3d_mesh.compute_vertex_normals()
    if color is not None:
        o3d_mesh.paint_uniform_color(color)
    else:
        o3d_mesh.paint_uniform_color([0.7, 0.7, 1.0])
    return o3d_mesh

def get_control_point_tensor():
    # Loads the panda gripper control points as in visualize_render.py
    control_points = np.load('assets/gripper_control_points/panda.npy')[:, :3]
    control_points = [[0, 0, 0], [0, 0, 0], control_points[0, :],
                      control_points[1, :], control_points[-2, :], control_points[-1, :]]
    control_points = np.asarray(control_points, dtype=np.float32)
    return control_points

def get_grasp_pc():
    grasp_pc = np.squeeze(get_control_point_tensor(), 0) if len(get_control_point_tensor().shape) == 3 else get_control_point_tensor()
    grasp_pc = grasp_pc.copy()
    grasp_pc[2, 2] = 0.059
    grasp_pc[3, 2] = 0.059
    mid_point = 0.5*(grasp_pc[2, :] + grasp_pc[3, :])
    modified_grasp_pc = []
    modified_grasp_pc.append(np.zeros((3,), np.float32))
    modified_grasp_pc.append(mid_point)
    modified_grasp_pc.append(grasp_pc[2])
    modified_grasp_pc.append(grasp_pc[4])
    modified_grasp_pc.append(grasp_pc[2])
    modified_grasp_pc.append(grasp_pc[3])
    modified_grasp_pc.append(grasp_pc[5])
    return np.asarray(modified_grasp_pc)

def plot_grasp_lines(grasp_tf, color=(0,1,0)):
    grasp_pc = get_grasp_pc()
    pts = np.matmul(grasp_pc, grasp_tf[:3, :3].T)
    pts += np.expand_dims(grasp_tf[:3, 3], 0)
    lines = []
    for j in range(len(pts) - 1):
        lines.append([j, j + 1])
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(pts)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color]*len(lines))
    return line_set

def main():
    parser = argparse.ArgumentParser(description='Visualize all per-joint grasps on the full mesh.')
    parser.add_argument('--grasps_json', type=str, required=True, help='Path to per-joint grasps summary JSON.')
    parser.add_argument('--xml', type=str, required=True, help='Path to MuJoCo XML file.')
    parser.add_argument('--full_mesh', type=str, required=True, help='Path to full .obj mesh.')
    parser.add_argument('--handle_meshes', action='store_true', help='Show handle meshes at each grasp (for debugging)')
    parser.add_argument('--max_grasps_per_joint', type=int, default=200, help='Maximum number of grasps to show per joint')
    parser.add_argument('--filtered_grasps', action='store_true', help='Show filtered grasps')
    args = parser.parse_args()

    # Load full mesh
    full_mesh = trimesh.load(args.full_mesh)
    # Parse joint world poses
    joint_poses = parse_joint_world_poses(args.xml)
    # Load per-joint grasps summary
    with open(args.grasps_json, 'r') as f:
        joint_grasps = json.load(f)
    geometries = [plot_mesh(full_mesh, color=[0.7,0.7,1.0])]
    for entry in joint_grasps:
        joint = entry['joint']
        if args.filtered_grasps:
            grasps_file = os.path.join(os.path.dirname(args.grasps_json), entry['filtered_grasps_file'])
        else:
            grasps_file = os.path.join(os.path.dirname(args.grasps_json), entry['grasps_file'])
        handle_mesh_file = os.path.join(os.path.dirname(args.grasps_json), entry['handle_mesh'])
        with open(grasps_file, 'r') as gf:
            grasps = json.load(gf)
        transforms = np.array(grasps['transforms'])
        # Limit number of grasps per joint
        if len(transforms) > args.max_grasps_per_joint:
            idx = np.random.choice(len(transforms), args.max_grasps_per_joint, replace=False)
            transforms = transforms[idx]
        # For each grasp, transform the grasp shape to world (grasp_only mode)
        for t in transforms:
            t = np.array(t)
            tf_world = t
            geometries.append(plot_grasp_lines(tf_world, color=(0,1,0)))
            if args.handle_meshes:
                handle_mesh = trimesh.load(handle_mesh_file)
                handle_mesh_tf = handle_mesh.copy()
                handle_mesh_tf.apply_transform(tf_world)
                geometries.append(plot_mesh(handle_mesh_tf, color=[1,0,0]))
    o3d.visualization.draw_geometries(geometries)

if __name__ == '__main__':
    main()

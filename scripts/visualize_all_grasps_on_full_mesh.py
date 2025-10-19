import argparse
import json
import numpy as np
import trimesh
import open3d as o3d
import xml.etree.ElementTree as ET
import os
import trimesh.transformations as tra
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from multiprocessing import Pool, cpu_count
from PIL import Image
import io
import time


def parse_joint_world_poses(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    mujoco = root if root.tag == "mujoco" else root.find("mujoco")
    world_poses = {}
    body_stack = [(mujoco.find("worldbody"), np.zeros(3), np.array([0, 0, 0, 1]))]
    while body_stack:
        body, parent_pos, parent_quat = body_stack.pop()
        for child in body:
            if child.tag == "body":
                pos = (
                    np.array(
                        [float(x) for x in child.attrib.get("pos", "0 0 0").split()]
                    )
                    if "pos" in child.attrib
                    else np.zeros(3)
                )
                quat = (
                    np.array(
                        [float(x) for x in child.attrib.get("quat", "0 0 0 1").split()]
                    )
                    if "quat" in child.attrib
                    else np.array([0, 0, 0, 1])
                )
                abs_pos = parent_pos + tra.quaternion_matrix(parent_quat)[:3, :3].dot(
                    pos
                )
                abs_quat = tra.quaternion_multiply(parent_quat, quat)
                for j in child.findall("joint"):
                    joint_name = j.attrib["name"]
                    joint_pos = abs_pos.copy()
                    joint_quat = abs_quat.copy()
                    world_poses[joint_name] = (joint_pos, joint_quat)
                body_stack.append((child, abs_pos, abs_quat))
    return world_poses


def plot_mesh_matplotlib(ax, mesh, color=None, alpha=0.3):
    if color is None:
        color = [0.7, 0.7, 1.0]

    vertices = mesh.vertices
    faces = mesh.faces

    triangles = []
    for face in faces:
        triangle = vertices[face]
        triangles.append(triangle)

    collection = Poly3DCollection(
        triangles, alpha=alpha, facecolor=color, edgecolor="none", linewidth=0.0
    )
    ax.add_collection3d(collection)

    return collection


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
    control_points = np.load("assets/grippers/robotiq/panda.npy")[:, :3]
    control_points = [
        [0, 0, 0],
        [0, 0, 0],
        control_points[0, :],
        control_points[1, :],
        control_points[-2, :],
        control_points[-1, :],
    ]
    control_points = np.asarray(control_points, dtype=np.float32)
    return control_points


def get_grasp_pc():
    grasp_pc = (
        np.squeeze(get_control_point_tensor(), 0)
        if len(get_control_point_tensor().shape) == 3
        else get_control_point_tensor()
    )
    grasp_pc = grasp_pc.copy()
    grasp_pc[2, 2] = 0.059
    grasp_pc[3, 2] = 0.059
    mid_point = 0.5 * (grasp_pc[2, :] + grasp_pc[3, :])
    modified_grasp_pc = []
    modified_grasp_pc.append(np.zeros((3,), np.float32))
    modified_grasp_pc.append(mid_point)
    modified_grasp_pc.append(grasp_pc[2])
    modified_grasp_pc.append(grasp_pc[4])
    modified_grasp_pc.append(grasp_pc[2])
    modified_grasp_pc.append(grasp_pc[3])
    modified_grasp_pc.append(grasp_pc[5])
    return np.asarray(modified_grasp_pc)


def plot_grasp_lines_matplotlib(ax, grasp_tf, color=(0, 1, 0)):
    grasp_pc = get_grasp_pc()
    pts = np.matmul(grasp_pc, grasp_tf[:3, :3].T)
    pts += np.expand_dims(grasp_tf[:3, 3], 0)

    for j in range(len(pts) - 1):
        ax.plot(
            [pts[j, 0], pts[j + 1, 0]],
            [pts[j, 1], pts[j + 1, 1]],
            [pts[j, 2], pts[j + 1, 2]],
            color=color,
            linewidth=2,
        )


def plot_grasp_lines(grasp_tf, color=(0, 1, 0)):
    grasp_pc = get_grasp_pc()
    pts = np.matmul(grasp_pc, grasp_tf[:3, :3].T)
    pts += np.expand_dims(grasp_tf[:3, 3], 0)
    lines = []
    for j in range(len(pts) - 1):
        lines.append([j, j + 1])
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(pts)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color] * len(lines))
    return line_set


def _plot_single_view(ax, full_mesh, joint_grasps, args, azim=45, elev=30):
    # Set up the 3D plot styling (similar to visualize.py)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("black")
    ax.yaxis.pane.set_edgecolor("black")
    ax.zaxis.pane.set_edgecolor("black")
    ax.xaxis.pane.set_alpha(0.0)
    ax.yaxis.pane.set_alpha(0.0)
    ax.zaxis.pane.set_alpha(0.0)
    ax.grid(False)
    ax.set_facecolor("black")

    # Plot the full mesh
    plot_mesh_matplotlib(ax, full_mesh, color=[0.7, 0.7, 1.0], alpha=0.3)

    # Colors for different joints
    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1)]

    for joint_idx, entry in enumerate(joint_grasps):
        joint = entry["joint"]
        if args.filtered_grasps:
            grasps_file = os.path.join(
                os.path.dirname(args.grasps_json), entry["filtered_grasps_file"]
            )
        else:
            grasps_file = os.path.join(
                os.path.dirname(args.grasps_json), entry["grasps_file"]
            )
        handle_mesh_file = os.path.join(
            os.path.dirname(args.grasps_json), entry["handle_mesh"]
        )

        if not os.path.exists(grasps_file):
            continue

        with open(grasps_file, "r") as gf:
            grasps = json.load(gf)
        transforms = np.array(grasps["transforms"])

        # Limit number of grasps per joint
        if len(transforms) > args.max_grasps_per_joint:
            idx = np.random.choice(
                len(transforms), args.max_grasps_per_joint, replace=False
            )
            transforms = transforms[idx]

        # Use different color for each joint
        joint_color = colors[joint_idx % len(colors)]

        # Plot grasps for this joint
        for t in transforms:
            t = np.array(t)
            plot_grasp_lines_matplotlib(ax, t, color=joint_color)

            if args.handle_meshes:
                try:
                    handle_mesh = trimesh.load(handle_mesh_file)
                    handle_mesh_tf = handle_mesh.copy()
                    handle_mesh_tf.apply_transform(t)
                    plot_mesh_matplotlib(
                        ax, handle_mesh_tf, color=joint_color, alpha=0.5
                    )
                except:
                    pass

    # Set proper bounds based on the mesh to ensure whole object is visible
    bounds = full_mesh.bounds
    center = full_mesh.center_mass
    extents = full_mesh.extents

    # Use a better balanced zoom factor
    zoom_factor = 3.0  # Better balance between visibility and closeness
    margin = np.max(extents) / zoom_factor

    # Set bounds based on actual mesh bounds with margin
    ax.set_xlim(bounds[0, 0] - margin, bounds[1, 0] + margin)
    ax.set_ylim(bounds[0, 1] - margin, bounds[1, 1] + margin)
    ax.set_zlim(bounds[0, 2] - margin, bounds[1, 2] + margin)

    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")


def _render_single_view_to_image(args_tuple):
    (view_idx, azim, elev, full_mesh, joint_grasps, args) = args_tuple

    fig = plt.figure(figsize=(10, 10), facecolor="black")
    ax = fig.add_subplot(111, projection="3d")

    _plot_single_view(ax, full_mesh, joint_grasps, args, azim, elev)

    buf = io.BytesIO()
    plt.savefig(
        buf, format="png", dpi=100, bbox_inches="tight", facecolor="black", pad_inches=0
    )
    plt.close(fig)

    buf.seek(0)
    return view_idx, buf.getvalue()


def _create_parallel_collage(full_mesh, joint_grasps, args, save_png):
    camera_setups = [
        # Top row - elevated views with better back coverage
        (0, 60),
        (90, 60),
        (180, 60),
        (270, 60),
        # Second row - eye level views
        (0, 0),
        (90, 0),
        (180, 0),
        (270, 0),
        # Third row - intermediate angles for better coverage
        (45, -15),
        (135, -15),
        (225, -15),
        (315, -15),
        # Bottom row - low angle views
        (0, -45),
        (90, -45),
        (180, -45),
        (270, -45),
    ]
    args_list = []
    for view_idx, (azim, elev) in enumerate(camera_setups):
        args_list.append((view_idx, azim, elev, full_mesh, joint_grasps, args))

    print(
        f"Rendering 16 views in parallel using {min(len(args_list), cpu_count())} processes..."
    )
    start_time = time.time()

    with Pool(processes=min(len(args_list), cpu_count())) as pool:
        results = pool.map(_render_single_view_to_image, args_list)

    parallel_time = time.time() - start_time
    print(f"Parallel rendering completed in {parallel_time:.2f} seconds")

    # Sort results by view index
    results.sort(key=lambda x: x[0])

    # Create collage
    images = []
    for view_idx, image_data in results:
        img = Image.open(io.BytesIO(image_data))
        images.append(img)

    img_width, img_height = images[0].size
    collage_width = img_width * 4  # 4x4 grid
    collage_height = img_height * 4

    collage = Image.new("RGB", (collage_width, collage_height), color="black")

    for i, img in enumerate(images):
        row = i // 4  # 4 columns per row
        col = i % 4
        x = col * img_width
        y = row * img_height
        collage.paste(img, (x, y))

    collage.save(save_png)
    total_time = time.time() - start_time
    print(
        f"High-quality 4x4 collage saved to {save_png} (total time: {total_time:.2f}s)"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Visualize all per-joint grasps on the full mesh."
    )
    parser.add_argument(
        "--grasps_json",
        type=str,
        required=True,
        help="Path to per-joint grasps summary JSON.",
    )
    parser.add_argument(
        "--xml", type=str, required=True, help="Path to MuJoCo XML file."
    )
    parser.add_argument(
        "--full_mesh", type=str, required=True, help="Path to full .obj mesh."
    )
    parser.add_argument(
        "--handle_meshes",
        action="store_true",
        help="Show handle meshes at each grasp (for debugging)",
    )
    parser.add_argument(
        "--max_grasps_per_joint",
        type=int,
        default=200,
        help="Maximum number of grasps to show per joint",
    )
    parser.add_argument(
        "--filtered_grasps", action="store_true", help="Show filtered grasps"
    )
    parser.add_argument(
        "--save-png",
        type=str,
        default=None,
        help="Save visualization as PNG file to specified path",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        default=True,
        help="Show interactive visualization window (default: True)",
    )
    parser.add_argument(
        "--no-render",
        dest="render",
        action="store_false",
        help="Do not show interactive visualization window",
    )
    args = parser.parse_args()

    # Load full mesh
    full_mesh = trimesh.load(args.full_mesh)
    # Parse joint world poses
    joint_poses = parse_joint_world_poses(args.xml)
    # Load per-joint grasps summary
    with open(args.grasps_json, "r") as f:
        joint_grasps = json.load(f)

    if args.save_png:
        # Create PNG collage
        _create_parallel_collage(full_mesh, joint_grasps, args, args.save_png)

    if args.render:
        # Create interactive Open3D visualization
        geometries = [plot_mesh(full_mesh, color=[0.7, 0.7, 1.0])]

        # Colors for different joints
        colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1)]

        for joint_idx, entry in enumerate(joint_grasps):
            joint = entry["joint"]
            if args.filtered_grasps:
                grasps_file = os.path.join(
                    os.path.dirname(args.grasps_json), entry["filtered_grasps_file"]
                )
            else:
                grasps_file = os.path.join(
                    os.path.dirname(args.grasps_json), entry["grasps_file"]
                )
            handle_mesh_file = os.path.join(
                os.path.dirname(args.grasps_json), entry["handle_mesh"]
            )

            if not os.path.exists(grasps_file):
                continue

            with open(grasps_file, "r") as gf:
                grasps = json.load(gf)
            transforms = np.array(grasps["transforms"])
            for i in range(len(transforms)):
                transforms[i][:3, 3] -= transforms[i][:3, :3] @ np.array(
                    [0, 0, 0.13675]
                )

            # Limit number of grasps per joint
            if len(transforms) > args.max_grasps_per_joint:
                idx = np.random.choice(
                    len(transforms), args.max_grasps_per_joint, replace=False
                )
                transforms = transforms[idx]

            # Use different color for each joint
            joint_color = colors[joint_idx % len(colors)]

            # For each grasp, transform the grasp shape to world (grasp_only mode)
            for t in transforms:
                t = np.array(t)
                tf_world = t
                geometries.append(plot_grasp_lines(tf_world, color=joint_color))
                if args.handle_meshes:
                    try:
                        handle_mesh = trimesh.load(handle_mesh_file)
                        handle_mesh_tf = handle_mesh.copy()
                        handle_mesh_tf.apply_transform(tf_world)
                        geometries.append(plot_mesh(handle_mesh_tf, color=joint_color))
                    except:
                        pass

        o3d.visualization.draw_geometries(geometries)

    if not args.render and not args.save_png:
        print("No visualization requested (--no-render and no --save-png)")


if __name__ == "__main__":
    main()

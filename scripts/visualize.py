from __future__ import print_function

import json
import trimesh
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import math
import numpy as np
import tensorflow as tf
import time
import trimesh.transformations as tra
import os
import argparse
import json
from multiprocessing import Pool, cpu_count
from PIL import Image
import io

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from assets.grippers.robotiq.robotiq_gripper import RobotiqGripper



parser = argparse.ArgumentParser(description="Visualize grasps from a JSON file.")
parser.add_argument("--grasps_path", type=str)
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
parser.add_argument(
    "--grasp-shape-only",
    action="store_true",
    help="Show only grasp shape lines without gripper mesh",
)
parser.add_argument(
    "--position",
    type=float,
    nargs=3,
    default=[0, 0, 0],
    help="Set position of the object in the scene (default: [0, 0, 0])",
)
parser.add_argument(
    "--rotation",
    type=float,
    nargs=4,
    default=[0, 0, 0, 1],
    help="Set rotation of the object in the scene as quaternion (default: [0, 0, 0, 1])",
)
parser.add_argument(
    "--min-contact-depth",
    type=float,
    default=0.0,
    help="Minimum contact depth (0.0=base, 1.0=tip). Only show grasps with contact depth >= this value (default: 0.0)",
)
parser.add_argument(
    "--max-contact-depth",
    type=float,
    default=0.3,
    help="Maximum contact depth (0.0=base, 1.0=tip). Only show grasps with contact depth <= this value (default: 1.0)",
)

GRIPPER_PC = np.load("assets/grippers/robotiq/panda_pc.npy", allow_pickle=True).item()[
    "points"
]
GRIPPER_PC[:, 3] = 1.0


class Object(object):
    def __init__(self, filename):
        self.mesh = trimesh.load(filename)
        self.scale = 1.0

        self.filename = filename
        if isinstance(self.mesh, list):
            print("Warning: Will do a concatenation")
            self.mesh = trimesh.util.concatenate(self.mesh)

        self.collision_manager = trimesh.collision.CollisionManager()
        self.collision_manager.add_object("object", self.mesh)

    def rescale(self, scale=1.0):
        self.scale = scale
        self.mesh.apply_scale(self.scale)

    def set_transform(self, position, rotation):
        if len(rotation) != 4:
            raise ValueError("Rotation must be a quaternion in xyzw format.")
        if not np.isclose(np.linalg.norm(rotation), 1.0):
            raise ValueError("Rotation must be a unit quaternion.")
        if len(position) != 3:
            raise ValueError("Position must be a 3D vector.")
        matrix = tra.quaternion_matrix(rotation)
        matrix[3, 3] = 1.0
        matrix[:3, 3] = position
        self.mesh.apply_transform(matrix)

    def resize(self, size=1.0):
        self.scale = size / np.max(self.mesh.extents)
        self.mesh.apply_scale(self.scale)

    def in_collision_with(self, mesh, transform):
        return self.collision_manager.in_collision_single(mesh, transform=transform)


def get_shape(x):
    return x.get_shape().as_list()


def count_nan(x):
    isnan = tf.cast(tf.is_nan(x), tf.int32)
    isnan = tf.reshape(isnan, [-1])
    return tf.reduce_sum(isnan)


def get_gripper_pc(batch_size, npoints, use_tf=True):
    output = np.copy(GRIPPER_PC)
    if npoints != -1:
        assert npoints > 0 and npoints <= output.shape[0], (
            "gripper_pc_npoint is too large {} > {}".format(npoints, output.shape[0])
        )
        output = output[:npoints]
        output = np.expand_dims(output, 0)
    else:
        raise ValueError("npoints should not be -1.")

    if use_tf:
        output = tf.convert_to_tensor(output, tf.float32)
        output = tf.tile(output, [batch_size, 1, 1])
        return output
    else:
        output = np.tile(output, [batch_size, 1, 1])

    return output


def get_control_point_tensor(batch_size, use_tf=True):
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
    control_points = np.tile(np.expand_dims(control_points, 0), [batch_size, 1, 1])

    if use_tf:
        return tf.convert_to_tensor(control_points)

    return control_points


def transform_control_points(
    gt_grasps, batch_size, mode="qt", scope="transform_gt_control_points"
):
    assert mode == "qt" or mode == "rt", mode
    grasp_shape = get_shape(gt_grasps)
    if mode == "qt":
        assert len(grasp_shape) == 2, grasp_shape
        assert grasp_shape[-1] == 7, grasp_shape
        with tf.variable_scope(scope):
            control_points = get_control_point_tensor(batch_size)
            num_control_points = get_shape(control_points)[1]
            input_gt_grasps = gt_grasps
            gt_grasps = tf.tile(
                tf.expand_dims(input_gt_grasps, 1), [1, num_control_points, 1]
            )
            gt_q = tf.slice(
                gt_grasps,
                [0, 0, 0],
                [get_shape(gt_grasps)[0], get_shape(gt_grasps)[1], 4],
            )
            gt_t = tf.slice(
                gt_grasps,
                [0, 0, 4],
                [get_shape(gt_grasps)[0], get_shape(gt_grasps)[1], 3],
            )

            gt_control_points = rotate_point_by_quaternion(control_points, gt_q)
            gt_control_points += gt_t

            return gt_control_points
    else:
        assert len(grasp_shape) == 3, grasp_shape
        assert grasp_shape[1] == 4 and grasp_shape[2] == 4, grasp_shape
        with tf.variable_scope(scope):
            control_points = get_control_point_tensor(batch_size)
            shape = get_shape(control_points)
            ones = tf.ones((shape[0], shape[1], 1), dtype=tf.float32)
            control_points = tf.concat((control_points, ones), -1)
            return tf.matmul(
                control_points, gt_grasps, transpose_a=False, transpose_b=True
            )


def quaternion_mult(Q, R):
    Q_shape = Q.get_shape().as_list()
    R_shape = R.get_shape().as_list()
    assert Q_shape[-1] == 4
    assert R_shape[-1] == 4
    q = tf.split(Q, 4, axis=-1)
    r = tf.split(R, 4, axis=-1)
    outputs_list = [
        r[0] * q[0] - r[1] * q[1] - r[2] * q[2] - r[3] * q[3],
        r[0] * q[1] + r[1] * q[0] - r[2] * q[3] + r[3] * q[2],
        r[0] * q[2] + r[1] * q[3] + r[2] * q[0] - r[3] * q[1],
        r[0] * q[3] - r[1] * q[2] + r[2] * q[1] + r[3] * q[0],
    ]
    outputs = tf.concat(outputs_list, axis=-1)
    return outputs


def conj_quaternion(q):
    q_conj = tf.split(q, 4, axis=-1)
    q_conj = tf.concat([q_conj[0], -q_conj[1], -q_conj[2], -q_conj[3]], axis=-1)
    return q_conj


def rotate_point_by_quaternion(point, q):
    shape = point.get_shape().as_list()
    q_shape = q.get_shape().as_list()

    assert len(shape) == 3, "point shape = {} q shape = {}".format(shape, q_shape)
    assert shape[-1] == 3, "point shape = {} q shape = {}".format(shape, q_shape)
    assert len(q_shape) == 3, "point shape = {} q shape = {}".format(shape, q_shape)
    assert q_shape[-1] == 4, "point shape = {} q shape = {}".format(shape, q_shape)
    assert q_shape[1] == shape[1], "point shape = {} q shape = {}".format(
        shape, q_shape
    )

    q_conj = conj_quaternion(q)
    r = tf.concat(
        [tf.zeros((shape[0], shape[1], 1), dtype=point.dtype), point], axis=-1
    )
    final_point = quaternion_mult(quaternion_mult(q, r), q_conj)
    final_output = tf.slice(final_point, [0, 0, 1], shape, name="sliceeeeeee")
    return final_output


class QuaternionTest(tf.test.TestCase):
    def test_mult(self):
        np.random.seed(int(time.time()))
        batch_size = 30
        control_points = 50
        a = np.random.rand(batch_size, control_points, 4)
        b = np.random.rand(batch_size, control_points, 4)
        norm_a = np.sqrt(np.sum(a * a, axis=-1))
        norm_b = np.sqrt(np.sum(b * b, axis=-1))
        a /= np.tile(np.expand_dims(norm_a, -1), [1, 1, 4])
        b /= np.tile(np.expand_dims(norm_b, -1), [1, 1, 4])
        output = np.zeros((batch_size, control_points, 4), dtype=np.float32)
        for bindex in range(batch_size):
            for c in range(control_points):
                output[bindex, c, :] = tra.quaternion_multiply(
                    a[bindex, c, :], b[bindex, c, :]
                )

        ta = tf.convert_to_tensor(a)
        tb = tf.convert_to_tensor(b)
        tf_output = quaternion_mult(ta, tb)

        ok = True
        with self.test_session():
            if np.all(np.abs(tf_output.eval() - output) < 1e-4):
                print("----------> Mult passed")
            else:
                raise ValueError(
                    "did not match {} != {}".format(tf_output.eval(), output)
                )

    def test_rotation(self):
        np.random.seed(int(time.time()))
        batch_size = 30
        control_points = 16

        rot_matrix = np.zeros((batch_size, control_points, 3, 3), dtype=np.float32)
        quat_matrix = np.zeros((batch_size, control_points, 4), dtype=np.float32)
        points = np.random.rand(batch_size, control_points, 3)

        rotated_points = np.random.rand(batch_size, control_points, 3)
        for b in range(batch_size):
            for c in range(control_points):
                angles = np.random.uniform(
                    low=0,
                    high=math.pi * 2.0,
                    size=[
                        3,
                    ],
                )
                rot_matrix[b, c, :, :] = tra.euler_matrix(
                    angles[0], angles[1], angles[2]
                )[:3, :3]
                quat_matrix[b, c, :] = tra.quaternion_from_euler(
                    angles[0], angles[1], angles[2]
                )

                rotated_points[b, c, :] = np.matmul(
                    rot_matrix[b, c, :, :], points[b, c, :]
                )

        tf_rotated_points = rotate_point_by_quaternion(
            tf.convert_to_tensor(points, dtype=tf.float32),
            tf.convert_to_tensor(quat_matrix, dtype=tf.float32),
        )

        with self.test_session():
            if np.all(np.abs(tf_rotated_points.eval() - rotated_points) < 1e-4):
                print("----------> Rotation passed")
            else:
                raise ValueError(
                    "test rotatation did not match {} != {}".format(
                        tf_rotated_points.eval(), rotated_points
                    )
                )


def tf_rotation_matrix(az, el, th, batched=False):
    if batched:
        cx = tf.cos(tf.reshape(az, [-1, 1]))
        cy = tf.cos(tf.reshape(el, [-1, 1]))
        cz = tf.cos(tf.reshape(th, [-1, 1]))
        sx = tf.sin(tf.reshape(az, [-1, 1]))
        sy = tf.sin(tf.reshape(el, [-1, 1]))
        sz = tf.sin(tf.reshape(th, [-1, 1]))

        ones = tf.ones_like(cx)
        zeros = tf.zeros_like(cx)

        rx = tf.concat([ones, zeros, zeros, zeros, cx, -sx, zeros, sx, cx], axis=-1)
        ry = tf.concat([cy, zeros, sy, zeros, ones, zeros, -sy, zeros, cy], axis=-1)
        rz = tf.concat([cz, -sz, zeros, sz, cz, zeros, zeros, zeros, ones], axis=-1)

        rx = tf.reshape(rx, [-1, 3, 3])
        ry = tf.reshape(ry, [-1, 3, 3])
        rz = tf.reshape(rz, [-1, 3, 3])

        return tf.matmul(rz, tf.matmul(ry, rx))
    else:
        cx = tf.cos(az)
        cy = tf.cos(el)
        cz = tf.cos(th)
        sx = tf.sin(az)
        sy = tf.sin(el)
        sz = tf.sin(th)

        rx = tf.to_float(tf.stack([[1.0, 0.0, 0.0], [0, cx, -sx], [0, sx, cx]], axis=0))
        ry = tf.to_float(tf.stack([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], axis=0))
        rz = tf.to_float(tf.stack([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], axis=0))

        return tf.matmul(rz, tf.matmul(ry, rx))


def get_color_plasma_org(x):
    import matplotlib.pyplot as plt

    return tuple([x for i, x in enumerate(plt.cm.plasma(x)) if i < 3])


def get_color_plasma(x):
    return tuple([float(1 - x), float(x), float(0)])


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


def draw_scene(
    pc,
    grasps=[],
    grasp_scores=None,
    grasp_color=None,
    gripper_color=(0, 1, 0),
    mesh=None,
    show_gripper_mesh=False,
    grasps_selection=None,
    visualize_diverse_grasps=False,
    min_seperation_distance=0.03,
    pc_color=None,
    plasma_coloring=False,
    save_png=None,
    render=True,
    grasp_widths=None,
):
    max_grasps = 200
    grasps = np.array(grasps)

    if len(grasps) == 0:
        print("No grasps to visualize")
        return

    if grasp_scores is not None:
        grasp_scores = np.array(grasp_scores)

    if grasp_widths is not None:
        grasp_widths = np.array(grasp_widths)

    if len(grasps) > max_grasps:
        print("Downsampling grasps, there are too many")
        chosen_ones = np.random.randint(low=0, high=len(grasps), size=max_grasps)
        grasps = grasps[chosen_ones]
        if grasp_scores is not None:
            grasp_scores = grasp_scores[chosen_ones]
        if grasp_widths is not None:
            grasp_widths = grasp_widths[chosen_ones]

    if save_png:
        _create_parallel_collage(
            pc,
            grasps,
            grasp_scores,
            grasp_color,
            gripper_color,
            mesh,
            show_gripper_mesh,
            grasps_selection,
            visualize_diverse_grasps,
            min_seperation_distance,
            pc_color,
            plasma_coloring,
            grasp_widths,
            save_png,
        )

    elif render:
        fig = plt.figure(figsize=(12, 10), facecolor="black")
        ax = fig.add_subplot(111, projection="3d")

        _plot_single_view(
            ax,
            pc,
            grasps,
            grasp_scores,
            grasp_color,
            gripper_color,
            mesh,
            show_gripper_mesh,
            grasps_selection,
            visualize_diverse_grasps,
            min_seperation_distance,
            pc_color,
            plasma_coloring,
            grasp_widths,
        )

        plt.show()

    if not render and not save_png:
        print("No visualization requested (--no-render and no --save-png)")


def _plot_single_view(
    ax,
    pc,
    grasps,
    grasp_scores,
    grasp_color,
    gripper_color,
    mesh,
    show_gripper_mesh,
    grasps_selection,
    visualize_diverse_grasps,
    min_seperation_distance,
    pc_color,
    plasma_coloring,
    grasp_widths,
    azim=45,
    elev=30,
):
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

    if mesh is not None:
        if type(mesh) == list:
            for elem in mesh:
                plot_mesh_matplotlib(ax, elem, color=[0.7, 0.7, 1.0], alpha=0.3)
        else:
            plot_mesh_matplotlib(ax, mesh, color=[0.7, 0.7, 1.0], alpha=0.3)

    if pc is not None:
        if pc_color is None:
            if plasma_coloring:
                z_values = pc[:, 2]
                z_normalized = (z_values - np.min(z_values)) / (
                    np.max(z_values) - np.min(z_values)
                )
                colors = np.zeros((len(pc), 3))
                colors[:, 0] = z_normalized
                colors[:, 1] = 1 - z_normalized
                colors[:, 2] = 0.5
            else:
                colors = "blue"
        else:
            colors = pc_color

        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=colors, s=1, alpha=0.6)

    if grasp_scores is not None:
        indexes = np.argsort(-np.asarray(grasp_scores))
    else:
        indexes = range(len(grasps))

    selected_grasps_so_far = []
    removed = 0

    if grasp_scores is not None:
        min_score = np.min(grasp_scores)
        max_score = np.max(grasp_scores)

    for ii in range(len(grasps)):
        i = indexes[ii]
        if grasps_selection is not None:
            if grasps_selection[i] == False:
                continue

        g = grasps[i]
        is_diverse = True

        grasp_pc = np.squeeze(get_control_point_tensor(1, False), 0)
        grasp_pc[2, 2] = 0.059
        grasp_pc[3, 2] = 0.059

        if grasp_widths is not None:
            grasp_width = grasp_widths[i] + 0.03
        else:
            grasp_width = 0.08

        grasp_pc[2, 0] = grasp_width * 0.5
        grasp_pc[3, 0] = -grasp_width * 0.5
        grasp_pc[4, 0] = grasp_width * 0.5
        grasp_pc[5, 0] = -grasp_width * 0.5

        mid_point = 0.5 * (grasp_pc[2, :] + grasp_pc[3, :])

        modified_grasp_pc = []
        modified_grasp_pc.append(np.zeros((3,), np.float32))
        modified_grasp_pc.append(mid_point)
        modified_grasp_pc.append(grasp_pc[2])
        modified_grasp_pc.append(grasp_pc[4])
        modified_grasp_pc.append(grasp_pc[2])
        modified_grasp_pc.append(grasp_pc[3])
        modified_grasp_pc.append(grasp_pc[5])

        grasp_pc = np.asarray(modified_grasp_pc)

        for prevg in selected_grasps_so_far:
            distance = np.linalg.norm(prevg[:3, 3] - g[:3, 3])

            if distance < min_seperation_distance:
                is_diverse = False
                break

        if visualize_diverse_grasps:
            if not is_diverse:
                removed += 1
                continue
            else:
                if grasp_scores is not None:
                    print("selected", i, grasp_scores[i], min_score, max_score)
                else:
                    print("selected", i)
                selected_grasps_so_far.append(g)

        current_gripper_color = gripper_color
        if isinstance(gripper_color, list) and len(gripper_color) > i:
            current_gripper_color = gripper_color[i]
        elif grasp_scores is not None:
            normalized_score = (grasp_scores[i] - min_score) / (
                max_score - min_score + 0.0001
            )
            if grasp_color is not None and len(grasp_color) > ii:
                current_gripper_color = grasp_color[ii]
            else:
                current_gripper_color = get_color_plasma(normalized_score)

            if min_score == 1.0:
                current_gripper_color = (0.0, 1.0, 0.0)

        pts = np.matmul(grasp_pc, g[:3, :3].T)
        pts += np.expand_dims(g[:3, 3], 0)

        for j in range(len(pts) - 1):
            ax.plot(
                [pts[j, 0], pts[j + 1, 0]],
                [pts[j, 1], pts[j + 1, 1]],
                [pts[j, 2], pts[j + 1, 2]],
                color=current_gripper_color,
                linewidth=2,
            )

    if mesh is not None:
        bounds = mesh.bounds
        center = mesh.center_mass
        max_extent = np.max(mesh.extents)
    elif pc is not None:
        bounds = np.array([np.min(pc, axis=0), np.max(pc, axis=0)])
        center = np.mean(pc, axis=0)
        max_extent = np.max(bounds[1] - bounds[0])
    else:
        center = np.array([0, 0, 0])
        max_extent = 1.0

    zoom_factor = 2.5
    ax.set_xlim(
        center[0] - max_extent / zoom_factor, center[0] + max_extent / zoom_factor
    )
    ax.set_ylim(
        center[1] - max_extent / zoom_factor, center[1] + max_extent / zoom_factor
    )
    ax.set_zlim(
        center[2] - max_extent / zoom_factor, center[2] + max_extent / zoom_factor
    )

    ax.view_init(elev=elev, azim=azim)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")


def _render_single_view_to_image(args):
    (
        view_idx,
        azim,
        elev,
        pc,
        grasps,
        grasp_scores,
        grasp_color,
        gripper_color,
        mesh,
        show_gripper_mesh,
        grasps_selection,
        visualize_diverse_grasps,
        min_seperation_distance,
        pc_color,
        plasma_coloring,
        grasp_widths,
    ) = args

    fig = plt.figure(figsize=(10, 10), facecolor="black")
    ax = fig.add_subplot(111, projection="3d")

    _plot_single_view(
        ax,
        pc,
        grasps,
        grasp_scores,
        grasp_color,
        gripper_color,
        mesh,
        show_gripper_mesh,
        grasps_selection,
        visualize_diverse_grasps,
        min_seperation_distance,
        pc_color,
        plasma_coloring,
        grasp_widths,
        azim,
        elev,
    )

    buf = io.BytesIO()
    plt.savefig(
        buf, format="png", dpi=100, bbox_inches="tight", facecolor="black", pad_inches=0
    )
    plt.close(fig)

    buf.seek(0)
    return view_idx, buf.getvalue()


def _create_parallel_collage(
    pc,
    grasps,
    grasp_scores,
    grasp_color,
    gripper_color,
    mesh,
    show_gripper_mesh,
    grasps_selection,
    visualize_diverse_grasps,
    min_seperation_distance,
    pc_color,
    plasma_coloring,
    grasp_widths,
    save_png,
):
    camera_setups = [
        (0, 60),
        (60, 60),
        (120, 60),
        (0, 0),
        (60, 0),
        (120, 0),
        (0, -60),
        (60, -60),
        (120, -60),
    ]

    args_list = []
    for view_idx, (azim, elev) in enumerate(camera_setups):
        args_list.append(
            (
                view_idx,
                azim,
                elev,
                pc,
                grasps,
                grasp_scores,
                grasp_color,
                gripper_color,
                mesh,
                show_gripper_mesh,
                grasps_selection,
                visualize_diverse_grasps,
                min_seperation_distance,
                pc_color,
                plasma_coloring,
                grasp_widths,
            )
        )

    print(
        f"Rendering 9 views in parallel using {min(len(args_list), cpu_count())} processes..."
    )
    start_time = time.time()

    with Pool(processes=min(len(args_list), cpu_count())) as pool:
        results = pool.map(_render_single_view_to_image, args_list)

    parallel_time = time.time() - start_time
    print(f"Parallel rendering completed in {parallel_time:.2f} seconds")

    results.sort(key=lambda x: x[0])

    images = []
    for view_idx, image_data in results:
        img = Image.open(io.BytesIO(image_data))
        images.append(img)

    img_width, img_height = images[0].size
    collage_width = img_width * 3
    collage_height = img_height * 3

    collage = Image.new("RGB", (collage_width, collage_height), color="black")

    for i, img in enumerate(images):
        row = i // 3
        col = i % 3
        x = col * img_width
        y = row * img_height
        collage.paste(img, (x, y))

    collage.save(save_png)
    total_time = time.time() - start_time
    print(
        f"High-quality 3x3 collage saved to {save_png} (total time: {total_time:.2f}s)"
    )


def get_axis():
    pass


args = parser.parse_args()


json_file = os.path.abspath(args.grasps_path)
if not os.path.exists(json_file):
    print(f"Grasps JSON file does not exist: {json_file}")
    sys.exit(1)

with open(json_file, "r") as f:
    data = json.load(f)

mesh = trimesh.load(data["object"])

mesh.apply_scale(data["object_scale"])

pose = tra.quaternion_matrix(data["object_rotation"])
pose[:3, 3] = data["object_position"]
pose[3, 3] = 1.0
mesh.apply_transform(pose)

transforms = np.array(data["transforms"])
gripper = RobotiqGripper()

for i in range(len(transforms)):
    transforms[i][:3, 3] -= transforms[i][:3, :3] @ (gripper.tcp_offset-0.03)

quality = np.array(
    data.get(
        "quality_antipodal",
        data.get("quality_number_of_contacts", [1.0] * len(transforms)),
    )
)
grasp_widths = np.array(data.get("grasp_widths", [0.0] * len(transforms)))
contact_depths = np.array(data.get("contact_depths", [0.5] * len(transforms)))

# Filter by contact depth range
if args.min_contact_depth > 0.0 or args.max_contact_depth < 1.0:
    depth_mask = (contact_depths >= args.min_contact_depth) & (contact_depths <= args.max_contact_depth)
    transforms = transforms[depth_mask]
    quality = quality[depth_mask]
    grasp_widths = grasp_widths[depth_mask]
    contact_depths = contact_depths[depth_mask]
    print(f"Filtered grasps by contact depth [{args.min_contact_depth}, {args.max_contact_depth}]: {len(transforms)} grasps remaining")

top_k = 2000

transforms = transforms[:top_k]
quality = quality[:top_k]
grasp_widths = grasp_widths[:top_k]
contact_depths = contact_depths[:top_k]

draw_scene(
    pc=None,
    grasps=transforms,
    grasp_scores=quality,
    mesh=mesh,
    show_gripper_mesh=not args.grasp_shape_only,
    plasma_coloring=True,
    save_png=args.save_png,
    render=args.render,
    grasp_widths=grasp_widths,
)

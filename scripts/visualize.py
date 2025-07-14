from __future__ import print_function

import json
import trimesh
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless operation
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
parser = argparse.ArgumentParser(description='Visualize grasps from a JSON file.')
parser.add_argument('object_name', type=str)
parser.add_argument('--filtered', action='store_true')
parser.add_argument('--compare', action='store_true', 
                   help='Show both filtered (green) and unfiltered (red) grasps')
parser.add_argument('--save-png', type=str, default=None, 
                   help='Save visualization as PNG file to specified path')
parser.add_argument('--render', action='store_true', default=True,
                   help='Show interactive visualization window (default: True)')
parser.add_argument('--no-render', dest='render', action='store_false',
                   help='Do not show interactive visualization window')
parser.add_argument('--grasp-shape-only', action='store_true',
                   help='Show only grasp shape lines without gripper mesh')
parser.add_argument('--position', type=float, nargs=3, default=[0, 0, 0],
                   help='Set position of the object in the scene (default: [0, 0, 0])')
parser.add_argument('--rotation', type=float, nargs=4, default=[0, 0, 0, 1],
                   help='Set rotation of the object in the scene as quaternion (default: [0, 0, 0, 1])')

GRIPPER_PC = np.load(
    'assets/gripper_models/panda_pc.npy', allow_pickle=True).item()['points']
GRIPPER_PC[:, 3] = 1.



class Object(object):
    """Represents a graspable object."""

    def __init__(self, filename):
        """Constructor.

        :param filename: Mesh to load
        :param scale: Scaling factor
        """
        self.mesh = trimesh.load(filename)
        self.scale = 1.0

        # print(filename)
        self.filename = filename
        if isinstance(self.mesh, list):
            # this is fixed in a newer trimesh version:
            # https://github.com/mikedh/trimesh/issues/69
            print("Warning: Will do a concatenation")
            self.mesh = trimesh.util.concatenate(self.mesh)

        self.collision_manager = trimesh.collision.CollisionManager()
        self.collision_manager.add_object('object', self.mesh)

    def rescale(self, scale=1.0):
        """Set scale of object mesh.

        :param scale
        """
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
        """Set longest of all three lengths in Cartesian space.

        :param size
        """
        self.scale = size / np.max(self.mesh.extents)
        self.mesh.apply_scale(self.scale)

    def in_collision_with(self, mesh, transform):
        """Check whether the object is in collision with the provided mesh.

        :param mesh:
        :param transform:
        :return: boolean value
        """
        return self.collision_manager.in_collision_single(mesh, transform=transform)

def get_shape(x):
    """
      Gets the shape of the tensor x.
    """
    return x.get_shape().as_list()


def count_nan(x):
    """
      Debug function: counts the nan values in tensor x.
    """
    isnan = tf.cast(tf.is_nan(x), tf.int32)
    isnan = tf.reshape(isnan, [-1])
    return tf.reduce_sum(isnan)


def get_gripper_pc(batch_size, npoints, use_tf=True):
    """
      Returns a numpy array or a tensor of shape (batch_size x npoints x 4).
      Represents gripper with the sepcified number of points.
      use_tf: switches between output tensor or numpy array.
    """
    output = np.copy(GRIPPER_PC)
    if npoints != -1:
        assert(
            npoints > 0 and npoints <= output.shape[0]), 'gripper_pc_npoint is too large {} > {}'.format(
            npoints, output.shape[0])
        output = output[:npoints]
        output = np.expand_dims(output, 0)
    else:
        raise ValueError('npoints should not be -1.')
        
    if use_tf:
        output = tf.convert_to_tensor(output, tf.float32)
        output = tf.tile(output, [batch_size, 1, 1])
        return output
    else:
        output = np.tile(output, [batch_size, 1, 1])

    return output


def get_control_point_tensor(batch_size, use_tf=True):
    """
      Outputs a tensor of shape (batch_size x 6 x 3).
      use_tf: switches between outputing a tensor and outputing a numpy array.
    """
    control_points = np.load('assets/gripper_control_points/panda.npy')[:, :3]
    control_points = [[0, 0, 0], [0, 0, 0], control_points[0, :],
                      control_points[1, :], control_points[-2, :], control_points[-1, :]]
    control_points = np.asarray(control_points, dtype=np.float32)
    control_points = np.tile(
        np.expand_dims(
            control_points, 0), [
            batch_size, 1, 1])

    if use_tf:
        return tf.convert_to_tensor(control_points)

    return control_points


def transform_control_points(
        gt_grasps,
        batch_size,
        mode='qt',
        scope='transform_gt_control_points'):
    """
      Transforms canonical points using gt_grasps.
      mode = 'qt' expects gt_grasps to have (batch_size x 7) where each 
        element is concatenation of quaternion and translation for each
        grasps.
      mode = 'rt': expects to have shape (batch_size x 4 x 4) where
        each element is 4x4 transformation matrix of each grasp.
    """
    assert(mode == 'qt' or mode == 'rt'), mode
    grasp_shape = get_shape(gt_grasps)
    if mode == 'qt':
        assert(len(grasp_shape) == 2), grasp_shape
        assert(grasp_shape[-1] == 7), grasp_shape
        with tf.variable_scope(scope):
            control_points = get_control_point_tensor(batch_size)
            num_control_points = get_shape(control_points)[1]
            input_gt_grasps = gt_grasps
            gt_grasps = tf.tile(
                tf.expand_dims(
                    input_gt_grasps, 1), [
                    1, num_control_points, 1])
            gt_q = tf.slice(
                gt_grasps, [
                    0, 0, 0], [
                    get_shape(gt_grasps)[0], get_shape(gt_grasps)[1], 4])
            gt_t = tf.slice(
                gt_grasps, [
                    0, 0, 4], [
                    get_shape(gt_grasps)[0], get_shape(gt_grasps)[1], 3])

            gt_control_points = rotate_point_by_quaternion(
                control_points, gt_q)
            gt_control_points += gt_t

            return gt_control_points
    else:
        assert(len(grasp_shape) == 3), grasp_shape
        assert(grasp_shape[1] == 4 and grasp_shape[2] == 4), grasp_shape
        with tf.variable_scope(scope):
            control_points = get_control_point_tensor(batch_size)
            shape = get_shape(control_points)
            ones = tf.ones((shape[0], shape[1], 1), dtype=tf.float32)
            control_points = tf.concat((control_points, ones), -1)
            return tf.matmul(
                control_points,
                gt_grasps,
                transpose_a=False,
                transpose_b=True)


def quaternion_mult(Q, R):
    """
      Computes the multiplication of quaternions Q and R.
    """
    Q_shape = Q.get_shape().as_list()
    R_shape = R.get_shape().as_list()
    assert (Q_shape[-1] == 4)
    assert (R_shape[-1] == 4)
    q = tf.split(Q, 4, axis=-1)
    r = tf.split(R, 4, axis=-1)
    outputs_list = [
        r[0] * q[0] - r[1] * q[1] - r[2] * q[2] - r[3] * q[3],
        r[0] * q[1] + r[1] * q[0] - r[2] * q[3] + r[3] * q[2],
        r[0] * q[2] + r[1] * q[3] + r[2] * q[0] - r[3] * q[1],
        r[0] * q[3] - r[1] * q[2] + r[2] * q[1] + r[3] * q[0]
    ]
    outputs = tf.concat(outputs_list, axis=-1)
    return outputs


def conj_quaternion(q):
    """
      Conjugate of quaternion q.
    """
    q_conj = tf.split(q, 4, axis=-1)
    q_conj = tf.concat(
        [q_conj[0], -q_conj[1], -q_conj[2], -q_conj[3]], axis=-1)
    return q_conj


def rotate_point_by_quaternion(point, q):
    """
      Takes in points with shape of (batch_size x n x 3) and quaternions with
      shape of (batch_size x n x 4) and returns a tensor with shape of 
      (batch_size x n x 3) which is the rotation of the point with quaternion
      q. 
    """
    shape = point.get_shape().as_list()
    q_shape = q.get_shape().as_list()

    assert(
        len(shape) == 3), 'point shape = {} q shape = {}'.format(
        shape, q_shape)
    assert(shape[-1] == 3), 'point shape = {} q shape = {}'.format(shape, q_shape)
    assert(
        len(q_shape) == 3), 'point shape = {} q shape = {}'.format(
        shape, q_shape)
    assert(q_shape[-1] ==
           4), 'point shape = {} q shape = {}'.format(shape, q_shape)
    assert(
        q_shape[1] == shape[1]), 'point shape = {} q shape = {}'.format(
        shape, q_shape)

    q_conj = conj_quaternion(q)
    r = tf.concat(
        [tf.zeros((shape[0], shape[1], 1), dtype=point.dtype), point], axis=-1)
    final_point = quaternion_mult(quaternion_mult(q, r), q_conj)
    final_output = tf.slice(final_point, [0, 0, 1], shape, name='sliceeeeeee')
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
                    a[bindex, c, :], b[bindex, c, :])

        ta = tf.convert_to_tensor(a)
        tb = tf.convert_to_tensor(b)
        tf_output = quaternion_mult(ta, tb)

        ok = True
        with self.test_session():
            if np.all(np.abs(tf_output.eval() - output) < 1e-4):
                print('----------> Mult passed')
            else:
                raise ValueError(
                    'did not match {} != {}'.format(
                        tf_output.eval(), output))

    def test_rotation(self):
        np.random.seed(int(time.time()))
        batch_size = 30
        control_points = 16

        rot_matrix = np.zeros(
            (batch_size, control_points, 3, 3), dtype=np.float32)
        quat_matrix = np.zeros(
            (batch_size, control_points, 4), dtype=np.float32)
        points = np.random.rand(batch_size, control_points, 3)

        rotated_points = np.random.rand(batch_size, control_points, 3)
        for b in range(batch_size):
            for c in range(control_points):
                angles = np.random.uniform(
                    low=0, high=math.pi * 2., size=[3, ])
                rot_matrix[b, c, :, :] = tra.euler_matrix(
                    angles[0], angles[1], angles[2])[:3, :3]
                quat_matrix[b, c, :] = tra.quaternion_from_euler(
                    angles[0], angles[1], angles[2])

                rotated_points[b, c, :] = np.matmul(
                    rot_matrix[b, c, :, :], points[b, c, :])

        tf_rotated_points = rotate_point_by_quaternion(
            tf.convert_to_tensor(
                points, dtype=tf.float32), tf.convert_to_tensor(
                quat_matrix, dtype=tf.float32))

        with self.test_session():
            if np.all(
                np.abs(
                    tf_rotated_points.eval() -
                    rotated_points) < 1e-4):
                print('----------> Rotation passed')
            else:
                raise ValueError(
                    'test rotatation did not match {} != {}'.format(
                        tf_rotated_points.eval(), rotated_points))


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

        rx = tf.concat([ones, zeros, zeros, zeros,
                        cx, -sx, zeros, sx, cx], axis=-1)
        ry = tf.concat([cy, zeros, sy, zeros, ones,
                        zeros, -sy, zeros, cy], axis=-1)
        rz = tf.concat([cz, -sz, zeros, sz, cz, zeros,
                        zeros, zeros, ones], axis=-1)

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

        rx = tf.to_float(
            tf.stack([[1., 0., 0.], [0, cx, -sx], [0, sx, cx]], axis=0))
        ry = tf.to_float(
            tf.stack([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], axis=0))
        rz = tf.to_float(
            tf.stack([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], axis=0))

        return tf.matmul(rz, tf.matmul(ry, rx))



def get_color_plasma_org(x):
    import matplotlib.pyplot as plt
    return tuple([x for i, x in enumerate(plt.cm.plasma(x)) if i < 3])

def get_color_plasma(x):
    return tuple([float(1 - x), float(x) , float(0)])

def plot_mesh_matplotlib(ax, mesh, color=None, alpha=0.3):
    """Plot trimesh using matplotlib 3D."""
    if color is None:
        color = [0.7, 0.7, 1.0]
    
    # Create a 3D collection from the mesh faces
    vertices = mesh.vertices
    faces = mesh.faces
    
    # Create a collection of triangles
    triangles = []
    for face in faces:
        triangle = vertices[face]
        triangles.append(triangle)
    
    # Add the collection to the plot
    collection = Poly3DCollection(triangles, alpha=alpha, facecolor=color, edgecolor='none', linewidth=0.0)
    ax.add_collection3d(collection)
    
    return collection

def draw_scene(
    pc, 
    grasps=[], 
    grasp_scores=None, 
    grasp_color=None, 
    gripper_color=(0,1,0), 
    mesh=None, 
    show_gripper_mesh=False, 
    grasps_selection=None, 
    visualize_diverse_grasps=False, 
    min_seperation_distance=0.03, 
    pc_color=None, 
    plasma_coloring=False,
    save_png=None,
    render=True,
    grasp_widths=None
    ):
    """
    Draws the 3D scene for the object and the scene using matplotlib.
    Args:
      pc: point cloud of the object
      grasps: list of 4x4 numpy array indicating the transformation of the grasps.
        grasp_scores: grasps will be colored based on the scores. If left 
        empty, grasps are visualized in green.
      grasp_color: if it is a tuple, sets the color for all the grasps. If list
        is provided it is the list of tuple(r,g,b) for each grasp.
      mesh: If not None, shows the mesh of the object. Type should be trimesh 
         mesh.
      show_gripper_mesh: If True, shows the gripper mesh for each grasp. 
      grasp_selection: if provided, filters the grasps based on the value of 
        each selection. 1 means select ith grasp. 0 means exclude the grasp.
      visualize_diverse_grasps: sorts the grasps based on score. Selects the 
        top score grasp to visualize and then choose grasps that are not within
        min_seperation_distance distance of any of the previously selected
        grasps. Only set it to True to declutter the grasps for better
        visualization.
      pc_color: if provided, should be a n x 3 numpy array for color of each 
        point in the point cloud pc. Each number should be between 0 and 1.
      plasma_coloring: If True, sets the plasma colormap for visualizting the 
        pc.
    """
    
    max_grasps = 200
    grasps = np.array(grasps)

    if len(grasps) == 0:
        print('No grasps to visualize')
        return

    if grasp_scores is not None:
        grasp_scores = np.array(grasp_scores)

    if grasp_widths is not None:
        grasp_widths = np.array(grasp_widths)

    if len(grasps) > max_grasps:
        print('Downsampling grasps, there are too many')
        chosen_ones = np.random.randint(low=0, high=len(grasps), size=max_grasps)
        grasps = grasps[chosen_ones]
        if grasp_scores is not None:
            grasp_scores = grasp_scores[chosen_ones]
        if grasp_widths is not None:
            grasp_widths = grasp_widths[chosen_ones]

    # Create matplotlib figure and axes
    if save_png:
        # Create 3x3 collage for PNG output using parallel rendering
        _create_parallel_collage(pc, grasps, grasp_scores, grasp_color, gripper_color, mesh, 
                                show_gripper_mesh, grasps_selection, visualize_diverse_grasps,
                                min_seperation_distance, pc_color, plasma_coloring, grasp_widths, save_png)
        
    elif render:
        # Single interactive view
        fig = plt.figure(figsize=(12, 10), facecolor='black')
        ax = fig.add_subplot(111, projection='3d')
        
        _plot_single_view(ax, pc, grasps, grasp_scores, grasp_color, gripper_color, mesh, 
                        show_gripper_mesh, grasps_selection, visualize_diverse_grasps,
                        min_seperation_distance, pc_color, plasma_coloring, grasp_widths)
        
        plt.show()
    
    # If neither render nor save_png, just return without creating any windows
    if not render and not save_png:
        print("No visualization requested (--no-render and no --save-png)")


def _plot_single_view(ax, pc, grasps, grasp_scores, grasp_color, gripper_color, mesh, 
                     show_gripper_mesh, grasps_selection, visualize_diverse_grasps,
                     min_seperation_distance, pc_color, plasma_coloring, grasp_widths,
                     azim=45, elev=30):
    """Plot a single 3D view using matplotlib."""
    
    # Set background color to black and remove grid
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('black')
    ax.yaxis.pane.set_edgecolor('black')
    ax.zaxis.pane.set_edgecolor('black')
    ax.xaxis.pane.set_alpha(0.0)
    ax.yaxis.pane.set_alpha(0.0)
    ax.zaxis.pane.set_alpha(0.0)
    ax.grid(False)
    ax.set_facecolor('black')
    
    # Add mesh to scene
    if mesh is not None:
        if type(mesh) == list:
            for elem in mesh:
                plot_mesh_matplotlib(ax, elem, color=[0.7, 0.7, 1.0], alpha=0.3)
        else:
            plot_mesh_matplotlib(ax, mesh, color=[0.7, 0.7, 1.0], alpha=0.3)

    # Add point cloud to scene
    if pc is not None:
        if pc_color is None:
            if plasma_coloring:
                # Create plasma-like coloring based on z-coordinate
                z_values = pc[:, 2]
                z_normalized = (z_values - np.min(z_values)) / (np.max(z_values) - np.min(z_values))
                colors = np.zeros((len(pc), 3))
                colors[:, 0] = z_normalized  # Red channel
                colors[:, 1] = 1 - z_normalized  # Green channel
                colors[:, 2] = 0.5  # Blue channel
            else:
                colors = 'blue'
        else:
            colors = pc_color
        
        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=colors, s=1, alpha=0.6)

    # Process grasps
    if grasp_scores is not None:
        indexes = np.argsort(-np.asarray(grasp_scores))
    else:
        indexes = range(len(grasps))

    print('draw scene ', len(grasps))
    
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

        grasp_pc[2,0] = grasp_width * 0.5  # Left finger base
        grasp_pc[3,0] = -grasp_width * 0.5  # Left finger tip
        grasp_pc[4,0] = grasp_width * 0.5  # Left finger base (duplicate)
        grasp_pc[5,0] = -grasp_width * 0.5  # Right finger base

        mid_point = 0.5*(grasp_pc[2, :] + grasp_pc[3, :])

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
                    print('selected', i, grasp_scores[i], min_score, max_score)
                else:
                    print('selected', i)
                selected_grasps_so_far.append(g)

        # Determine grasp color
        current_gripper_color = gripper_color
        if isinstance(gripper_color, list) and len(gripper_color) > i:
            current_gripper_color = gripper_color[i]
        elif grasp_scores is not None:
            normalized_score = (grasp_scores[i] - min_score) / (max_score - min_score + 0.0001)
            if grasp_color is not None and len(grasp_color) > ii:
                current_gripper_color = grasp_color[ii]
            else:
                current_gripper_color = get_color_plasma(normalized_score)

            if min_score == 1.0:
                current_gripper_color = (0.0, 1.0, 0.0)

        # Create gripper visualization
        if show_gripper_mesh:
            # Load and transform gripper mesh
            try:
                object = Object('assets/gripper_models/rum_gripper/model.obj')
                gripper_mesh = object.mesh.copy()
                gripper_mesh.apply_transform(g)
                plot_mesh_matplotlib(ax, gripper_mesh, color=current_gripper_color, alpha=0.5)
            except:
                print("Warning: Could not load gripper mesh, showing lines only")
        
        # Create line visualization for grasp
        pts = np.matmul(grasp_pc, g[:3, :3].T)
        pts += np.expand_dims(g[:3, 3], 0)
        
        # Draw lines connecting the grasp points
        for j in range(len(pts) - 1):
            ax.plot([pts[j, 0], pts[j+1, 0]], 
                   [pts[j, 1], pts[j+1, 1]], 
                   [pts[j, 2], pts[j+1, 2]], 
                   color=current_gripper_color, linewidth=2)

    # Set equal aspect ratio and view
    if mesh is not None:
        # Get mesh bounds for setting axis limits
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
    
    # Set axis limits with tighter zoom (reduced from /2 to /2.5 for more zoom)
    zoom_factor = 2.5
    ax.set_xlim(center[0] - max_extent/zoom_factor, center[0] + max_extent/zoom_factor)
    ax.set_ylim(center[1] - max_extent/zoom_factor, center[1] + max_extent/zoom_factor)
    ax.set_zlim(center[2] - max_extent/zoom_factor, center[2] + max_extent/zoom_factor)
    
    # Set view angle
    ax.view_init(elev=elev, azim=azim)
    
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    print('removed {} similar grasps'.format(removed))  


def _render_single_view_to_image(args):
    """Render a single view and return the image data as bytes.
    
    This function is designed to be used with multiprocessing.
    """
    (view_idx, azim, elev, pc, grasps, grasp_scores, grasp_color, gripper_color, 
     mesh, show_gripper_mesh, grasps_selection, visualize_diverse_grasps,
     min_seperation_distance, pc_color, plasma_coloring, grasp_widths) = args
    
    # Create a single subplot figure for this view
    fig = plt.figure(figsize=(10, 10), facecolor='black')
    ax = fig.add_subplot(111, projection='3d')
    
    # Render this specific view
    _plot_single_view(ax, pc, grasps, grasp_scores, grasp_color, gripper_color, mesh, 
                     show_gripper_mesh, grasps_selection, visualize_diverse_grasps,
                     min_seperation_distance, pc_color, plasma_coloring, grasp_widths, azim, elev)
    
    # Save to bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='black', pad_inches=0)
    plt.close(fig)
    
    # Return view index and image data
    buf.seek(0)
    return view_idx, buf.getvalue()


def _create_parallel_collage(pc, grasps, grasp_scores, grasp_color, gripper_color, mesh, 
                           show_gripper_mesh, grasps_selection, visualize_diverse_grasps,
                           min_seperation_distance, pc_color, plasma_coloring, grasp_widths, save_png):
    """Create a 3x3 collage using parallel rendering of individual views."""
    
    # Define 9 diverse camera positions
    camera_setups = [
        # Row 1: Top views
        (0, 60),      # azim=0, elev=60 (top-front)
        (60, 60),     # azim=60, elev=60 (top-right)
        (120, 60),    # azim=120, elev=60 (top-back-right)
        
        # Row 2: Eye level views
        (0, 0),       # azim=0, elev=0 (front)
        (60, 0),      # azim=60, elev=0 (right)
        (120, 0),     # azim=120, elev=0 (back-right)
        
        # Row 3: Bottom views
        (0, -60),     # azim=0, elev=-60 (bottom-front)
        (60, -60),    # azim=60, elev=-60 (bottom-right)
        (120, -60),   # azim=120, elev=-60 (bottom-back-right)
    ]
    
    # Prepare arguments for parallel processing
    args_list = []
    for view_idx, (azim, elev) in enumerate(camera_setups):
        args_list.append((view_idx, azim, elev, pc, grasps, grasp_scores, grasp_color, gripper_color, 
                         mesh, show_gripper_mesh, grasps_selection, visualize_diverse_grasps,
                         min_seperation_distance, pc_color, plasma_coloring, grasp_widths))
    
    print(f"Rendering 9 views in parallel using {min(len(args_list), cpu_count())} processes...")
    start_time = time.time()
    
    # Render views in parallel
    with Pool(processes=min(len(args_list), cpu_count())) as pool:
        results = pool.map(_render_single_view_to_image, args_list)
    
    parallel_time = time.time() - start_time
    print(f"Parallel rendering completed in {parallel_time:.2f} seconds")
    
    # Sort results by view index to maintain correct order
    results.sort(key=lambda x: x[0])
    
    # Load images and combine into 3x3 grid
    images = []
    for view_idx, image_data in results:
        img = Image.open(io.BytesIO(image_data))
        images.append(img)
    
    # Create 3x3 collage
    img_width, img_height = images[0].size
    collage_width = img_width * 3
    collage_height = img_height * 3
    
    collage = Image.new('RGB', (collage_width, collage_height), color='black')
    
    for i, img in enumerate(images):
        row = i // 3
        col = i % 3
        x = col * img_width
        y = row * img_height
        collage.paste(img, (x, y))
    
    # Save the final collage
    collage.save(save_png)
    total_time = time.time() - start_time
    print(f"High-quality 3x3 collage saved to {save_png} (total time: {total_time:.2f}s)")


def get_axis():
    """Create coordinate frame using matplotlib - deprecated, not needed anymore"""
    pass


args = parser.parse_args()

# Define file paths - check both new structure and old structure
base_json_file = os.path.abspath(f"output/{args.object_name}/{args.object_name}_grasps.json")
filtered_json_file = os.path.abspath(f"output/{args.object_name}/{args.object_name}_grasps_filtered.json")

# Fallback to old structure if new structure doesn't exist
if not os.path.exists(base_json_file):
    base_json_file = os.path.abspath(f"output/{args.object_name}_grasps.json")
if not os.path.exists(filtered_json_file):
    filtered_json_file = os.path.abspath(f"output/{args.object_name}_grasps_filtered.json")

# Load saved grasp data
if args.compare:
    # Load both files for comparison
    try:
        with open(base_json_file, 'r') as f:
            base_data = json.load(f)
        
        with open(filtered_json_file, 'r') as f:
            filtered_data = json.load(f)
            
        # Load object mesh from either file (they should be the same)
        mesh = trimesh.load(base_data['object'])
        mesh.apply_scale(base_data['object_scale'])

        pose = np.identity(4)
        pose[:3, 3] = base_data['object_position']
        pose[:3, :3] = tra.quaternion_matrix(base_data['object_rotation'])
        mesh.apply_transform(pose)
        
        # Get all transforms
        all_transforms = np.array(base_data['transforms'])
        all_quality = np.array(base_data.get('quality_antipodal', 
                              base_data.get('quality_number_of_contacts', [1.0]*len(all_transforms))))
        
        # Get filtered transforms
        filtered_transforms = np.array(filtered_data['transforms'])
        
        # Convert filtered transforms to list of lists for easier comparison
        filtered_transform_lists = [t.tolist() if not isinstance(t, list) else t for t in filtered_transforms]
        
        # Create color list and separate transforms based on filtered status
        colors = []
        filtered_indices = []
        unfiltered_indices = []
        
        for i, transform in enumerate(all_transforms):
            transform_list = transform.tolist() if not isinstance(transform, list) else transform
            if any(np.allclose(np.array(transform_list), np.array(ft)) for ft in filtered_transform_lists):
                colors.append((0, 1, 0))  # Green for filtered
                filtered_indices.append(i)
            else:
                colors.append((1, 0, 0))  # Red for unfiltered/rejected
                unfiltered_indices.append(i)
                
        print(f"Showing {len(filtered_indices)} filtered grasps (green) and {len(unfiltered_indices)} unfiltered grasps (red)")
        
        # Visualize
        draw_scene(
            pc=None,
            grasps=all_transforms,
            grasp_scores=all_quality,
            mesh=mesh,
            show_gripper_mesh=not args.grasp_shape_only,
            plasma_coloring=False,
            gripper_color=colors,
            save_png=args.save_png,
            render=args.render
        )
    except FileNotFoundError as e:
        print(f"Error: Could not find one of the required grasp files. Make sure both filtered and unfiltered files exist.")
        print(f"Exception: {e}")
else:
    # Original behavior
    if args.filtered:
        extra = '_filtered'
    else:
        extra = ''
    
    # Check new structure first, then fallback to old structure
    json_file = os.path.abspath(f"output/{args.object_name}/{args.object_name}_grasps{extra}.json")
    if not os.path.exists(json_file):
        json_file = os.path.abspath(f"output/{args.object_name}_grasps{extra}.json")
    
    # Load saved grasp data
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Load object mesh
    mesh = trimesh.load(data['object'])

    # Apply scale
    mesh.apply_scale(data['object_scale'])

    pose = tra.quaternion_matrix(data['object_rotation'])
    pose[:3, 3] = data['object_position']
    pose[3,3] = 1.0
    mesh.apply_transform(pose)

    # Extract grasp info
    transforms = np.array(data['transforms'])
    quality = np.array(data.get('quality_antipodal', data.get('quality_number_of_contacts', [1.0]*len(transforms))))
    grasp_widths = np.array(data.get('grasp_widths', [0.0]*len(transforms)))

    top_k = 2000
    # top_indices = np.argsort(quality)[-top_k:][::-1]
    # transforms = [transforms[i] for i in top_indices]
    # quality = [quality[i] for i in top_indices]
    # grasp_widths = [grasp_widths[i] for i in top_indices]

    transforms = transforms[:top_k]
    quality = quality[:top_k]
    grasp_widths = grasp_widths[:top_k]

    # Visualize
    draw_scene( 
        pc=None,
        grasps=transforms,
        grasp_scores=quality,
        mesh=mesh,
        show_gripper_mesh=not args.grasp_shape_only,  # Use the new option
        plasma_coloring=True,
        save_png=args.save_png,
        render=args.render,
        grasp_widths=grasp_widths
    )

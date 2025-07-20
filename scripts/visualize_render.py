from __future__ import print_function

import json
import trimesh
import numpy as np
import open3d as o3d
import math
import numpy as np
import tensorflow as tf
import time
import trimesh.transformations as tra
import os
import argparse
import json
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

def plot_mesh(mesh, color=None):
    """Convert trimesh to open3d mesh and return it."""
    assert type(mesh) == trimesh.base.Trimesh
    
    # Create open3d mesh
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
    
    # Compute normals
    o3d_mesh.compute_vertex_normals()
    
    # Set color if provided
    if color is not None:
        o3d_mesh.paint_uniform_color(color)
    else:
        o3d_mesh.paint_uniform_color([0.7, 0.7, 1.0])  # Light blue default
    
    return o3d_mesh

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
    render=True):
    """
    Draws the 3D scene for the object and the scene using Open3D.
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
    
    # Create list to hold all geometries
    geometries = []
    
    max_grasps = 200
    grasps = np.array(grasps)

    if grasp_scores is not None:
        grasp_scores = np.array(grasp_scores)

    if len(grasps) > max_grasps:
        print('Downsampling grasps, there are too many')
        chosen_ones = np.random.randint(low=0, high=len(grasps), size=max_grasps)
        grasps = grasps[chosen_ones]
        if grasp_scores is not None:
            grasp_scores = grasp_scores[chosen_ones]

    # Add mesh to scene
    if mesh is not None:
        if type(mesh) == list:
            for elem in mesh:
                geometries.append(plot_mesh(elem))
        else:
            geometries.append(plot_mesh(mesh))

    # Add point cloud to scene
    if pc is not None:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc[:, :3])
        
        if pc_color is None:
            if plasma_coloring:
                # Create plasma-like coloring based on z-coordinate
                z_values = pc[:, 2]
                z_normalized = (z_values - np.min(z_values)) / (np.max(z_values) - np.min(z_values))
                colors = np.zeros((len(pc), 3))
                colors[:, 0] = z_normalized  # Red channel
                colors[:, 1] = 1 - z_normalized  # Green channel
                colors[:, 2] = 0.5  # Blue channel
                pcd.colors = o3d.utility.Vector3dVector(colors)
            else:
                pcd.paint_uniform_color([0.1, 0.1, 1])
        else:
            pcd.colors = o3d.utility.Vector3dVector(pc_color)
        
        geometries.append(pcd)

    # Create grasp visualization
    grasp_pc = np.squeeze(get_control_point_tensor(1, False), 0)
    print(grasp_pc.shape)
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

    grasp_pc = np.asarray(modified_grasp_pc)

    def transform_grasp_pc(g):
        output = np.matmul(grasp_pc, g[:3, :3].T)
        output += np.expand_dims(g[:3, 3], 0)
        return output

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
        top5 = np.array(grasp_scores).argsort()[-5:][::-1]

    for ii in range(len(grasps)):
        i = indexes[ii]
        if grasps_selection is not None:
            if grasps_selection[i] == False:
                continue
        
        g = grasps[i]
        is_diverse = True
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
        if isinstance(gripper_color, list):
            current_gripper_color = gripper_color[i]
        elif grasp_scores is not None:
            normalized_score = (grasp_scores[i] - min_score) / (max_score - min_score + 0.0001)
            if grasp_color is not None:
                current_gripper_color = grasp_color[ii]
            else:
                current_gripper_color = get_color_plasma(normalized_score)

            if min_score == 1.0:
                current_gripper_color = (0.0, 1.0, 0.0)

        # Create gripper visualization
        if show_gripper_mesh:
            # Load and transform gripper mesh
            object = Object('assets/gripper_models/rum_gripper/model.obj')
            gripper_mesh = object.mesh
            gripper_mesh.apply_transform(g)
            o3d_gripper = plot_mesh(gripper_mesh, color=current_gripper_color)
            geometries.append(o3d_gripper)
        else:
            # Create line set for grasp visualization
            pts = np.matmul(grasp_pc, g[:3, :3].T)
            pts += np.expand_dims(g[:3, 3], 0)
            
            # Create line set connecting the grasp points
            lines = []
            for j in range(len(pts) - 1):
                lines.append([j, j + 1])
            
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(pts)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            
            # Set color for all lines
            colors = [current_gripper_color] * len(lines)
            line_set.colors = o3d.utility.Vector3dVector(colors)
            
            geometries.append(line_set)

    # Create coordinate frame
    # coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    # geometries.append(coord_frame)

    # Visualize
    if render or save_png:
        if save_png:
            print(f"Creating 3x3 (9-shot) collage and saving to {save_png}")
            # Create 9 different views for collage
            import cv2
            
            images = []
            
            # First, create one working view to get the baseline
            vis = o3d.visualization.Visualizer()
            vis.create_window(width=800, height=800, visible=False)
            
            for geom in geometries:
                vis.add_geometry(geom)
            
            # Set render options for high quality
            render_option = vis.get_render_option()
            render_option.background_color = np.array([0.1, 0.1, 0.1])
            render_option.point_size = 4.0  # Increased point size
            render_option.line_width = 4.0  # Increased line width
            
            # Get the view control and let it auto-fit
            view_control = vis.get_view_control()
            
            # Define 9 diverse camera positions
            # Row 1: Top views (looking down from different angles)
            # Row 2: Eye level views (horizontal rotations)
            # Row 3: Bottom views (looking up from different angles)
            camera_setups = [
                # Row 1: Top views
                (0, -30, 0),      # Top-front
                (90, -30, 0),     # Top-right
                (180, -30, 0),    # Top-back
                
                # Row 2: Eye level views
                (0, 0, 0),        # Front
                (90, 0, 0),       # Right
                (180, 0, 0),      # Back
                
                # Row 3: Bottom views
                (0, 30, 0),       # Bottom-front
                (90, 30, 0),      # Bottom-right
                (270, 30, 0),     # Bottom-left
            ]
            
            for i, (azimuth, elevation, roll) in enumerate(camera_setups):
                # Reset view to default
                view_control.reset_camera_local_rotate()
                
                # Apply rotations for diverse views
                # First rotate around Y-axis (azimuth)
                if azimuth != 0:
                    view_control.rotate(azimuth * 400 / 90, 0)  # Scale rotation
                
                # Then rotate around X-axis (elevation - up/down)
                if elevation != 0:
                    view_control.rotate(0, elevation * 400 / 90)  # Scale rotation
                
                # Update and render
                vis.poll_events()
                vis.update_renderer()
                
                # Capture high-resolution image
                temp_filename = f"temp_view_{i}.png"
                vis.capture_screen_image(temp_filename)
                
                # Read image
                img = cv2.imread(temp_filename)
                if img is not None:
                    images.append(img)
                else:
                    print(f"Warning: Failed to capture view {i}")
                
                # Clean up
                import os
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
            
            vis.destroy_window()
            
            # Create collage (3x3 grid)
            if len(images) == 9:
                # Resize images to high resolution if needed
                target_size = (800, 800)  # High resolution per image
                resized_images = []
                for img in images:
                    if img.shape[:2] != target_size:
                        img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_CUBIC)
                    else:
                        img_resized = img
                    resized_images.append(img_resized)
                
                # Create 3x3 collage (2400x2400 final resolution)
                row1 = np.hstack([resized_images[0], resized_images[1], resized_images[2]])
                row2 = np.hstack([resized_images[3], resized_images[4], resized_images[5]])
                row3 = np.hstack([resized_images[6], resized_images[7], resized_images[8]])
                collage = np.vstack([row1, row2, row3])
                
                # Save high-quality collage
                cv2.imwrite(save_png, collage, [cv2.IMWRITE_PNG_COMPRESSION, 1])  # Minimal compression
                print(f"High-quality 3x3 collage saved to {save_png} (2400x2400 pixels)")
            else:
                print(f"Warning: Could not create all 9 views (got {len(images)}), falling back to single view")
                # Fallback to single view
                vis = o3d.visualization.Visualizer()
                vis.create_window(width=800, height=600)
                
                for geom in geometries:
                    vis.add_geometry(geom)
                
                render_option = vis.get_render_option()
                render_option.background_color = np.array([0.1, 0.1, 0.1])
                render_option.point_size = 2.0
                render_option.line_width = 2.0
                
                vis.run()
                vis.capture_screen_image(save_png)
                vis.destroy_window()
        
        if render:
            vis = o3d.visualization.Visualizer()
            vis.create_window(width=800, height=600)
            
            for geom in geometries:
                vis.add_geometry(geom)
            
            # Set view options
            render_option = vis.get_render_option()
            render_option.background_color = np.array([0.1, 0.1, 0.1])
            render_option.point_size = 2.0
            render_option.line_width = 2.0
            
            vis.run()
            vis.destroy_window()

    print('removed {} similar grasps'.format(removed))  

def get_axis():
    """Create coordinate frame for open3d"""
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.10)


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

    top_k = 2000
    top_indices = np.argsort(quality)[-top_k:][::-1]
    transforms = [transforms[i] for i in top_indices]
    quality = [quality[i] for i in top_indices]

    # Visualize
    draw_scene( 
        pc=None,
        grasps=transforms,
        grasp_scores=quality,
        mesh=mesh,
        show_gripper_mesh=not args.grasp_shape_only,  # Use the new option
        plasma_coloring=True,
        save_png=args.save_png,
        render=args.render
    )
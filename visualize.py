from __future__ import print_function

import matplotlib
matplotlib.use('TkAgg')
import json
import trimesh
import numpy as np
import mayavi.mlab as mlab
import numpy as np
import trimesh
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
parser.add_argument('--verified', action='store_true')

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

def plot_mesh(mesh):
    assert type(mesh) == trimesh.base.Trimesh
    mlab.triangular_mesh (
        mesh.vertices[:, 0],
        mesh.vertices[:, 1],
        mesh.vertices[:, 2],
        mesh.faces,
        colormap='Blues'
    )

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
    plasma_coloring=False):
    """
    Draws the 3D scene for the object and the scene.
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

    if grasp_scores is not None:
        grasp_scores = np.array(grasp_scores)

    if len(grasps) > max_grasps:

        print('Downsampling grasps, there are too many')
        chosen_ones = np.random.randint(low=0, high=len(grasps), size=max_grasps)
        grasps = grasps[chosen_ones]
        if grasp_scores is not None:
            grasp_scores = grasp_scores[chosen_ones]

    
    if mesh is not None:
        if type(mesh) == list:
            for elem in mesh:
                plot_mesh(elem)
        else:
            plot_mesh(mesh)

    if pc_color is None and pc is not None:
        if plasma_coloring:
            mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], pc[:, 2], colormap='plasma')
        else:
            mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], color=(0.1,0.1,1),scale_factor=0.01)
    elif pc is not None:
        if plasma_coloring:
            mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], pc_color[:, 0], colormap='plasma')
        else:
            rgba = np.zeros((pc.shape[0], 4), dtype=np.uint8)
            rgba[:, :3] = np.asarray(pc_color)
            rgba[:, 3] = 255
            src = mlab.pipeline.scalar_scatter(pc[:, 0], pc[:, 1], pc[:, 2])
            src.add_attribute(rgba, 'colors')
            src.data.point_data.set_active_scalars('colors')
            g = mlab.pipeline.glyph(src)
            g.glyph.scale_mode = "data_scaling_off"
            g.glyph.glyph.scale_factor = 0.01


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

        if isinstance(gripper_color, list):
            pass
        elif grasp_scores is not None:
            normalized_score = (grasp_scores[i] - min_score) / (max_score - min_score + 0.0001)
            if grasp_color is not None:
                gripper_color = grasp_color[ii]
            else:
                gripper_color = get_color_plasma(normalized_score)

            if min_score == 1.0:
                gripper_color = (0.0, 1.0, 0.0)

    
        if show_gripper_mesh:
            object = Object('new_rum.obj')
            # object.rescale(0.001)
            gripper_mesh = object.mesh
            gripper_mesh.apply_transform(g)
            mlab.triangular_mesh(
                gripper_mesh.vertices[:, 0],
                gripper_mesh.vertices[:, 1],
                gripper_mesh.vertices[:, 2],
                gripper_mesh.faces,
                color=gripper_color,
                opacity=1 if visualize_diverse_grasps else 0.5
            )
        # else:
        pts = np.matmul(grasp_pc, g[:3, :3].T)
        pts += np.expand_dims(g[:3, 3], 0)
        if isinstance(gripper_color, list):
            mlab.plot3d(pts[:, 0], pts[:, 1], pts[:, 2], color=gripper_color[i], tube_radius=0.003, opacity=1)
        else:
            tube_radius = 0.001                    
            mlab.plot3d(pts[:, 0], pts[:, 1], pts[:, 2], color=gripper_color, tube_radius=tube_radius, opacity=1)
    mlab.show()

    print('removed {} similar grasps'.format(removed))  

def get_axis():
    # hacky axis for mayavi
    axis = np.array([[1,0,0], [0,1,0], [0,0,1]])
    axis_x = np.array([np.linspace(0, 0.10, 50), np.zeros(50), np.zeros(50)]).T
    axis_y = np.array([np.zeros(50), np.linspace(0, 0.10, 50), np.zeros(50)]).T
    axis_z = np.array([np.zeros(50), np.zeros(50), np.linspace(0, 0.10, 50)]).T
    axis = np.concatenate([axis_x, axis_y, axis_z], axis=0)
    return axis


if parser.parse_args().verified:
    extra = '_verified'
else:
    extra = ''
json_file = os.path.abspath(f"output/{parser.parse_args().object_name}_grasps{extra}.json")
# Load saved grasp data
with open(json_file, 'r') as f:
    data = json.load(f)

# Load object mesh
mesh = trimesh.load(data['object'])

# Apply scale
mesh.apply_scale(data['object_scale'])

# Extract grasp info
transforms = np.array(data['transforms'])
quality = np.array(data.get('quality_antipodal', data.get('quality_number_of_contacts', [1.0]*len(transforms))))

top_k = 10
top_indices = np.argsort(quality)[-top_k:][::-1]
transforms = [transforms[i] for i in top_indices]
quality = [quality[i] for i in top_indices]


# Visualize
draw_scene( 
    pc=None,
    grasps=transforms,
    grasp_scores=quality,
    mesh=mesh,
    show_gripper_mesh=True,  # Set to True if you want full gripper meshes
    plasma_coloring=True
)
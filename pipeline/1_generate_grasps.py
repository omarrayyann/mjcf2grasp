# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
# -*- coding: utf-8 -*-
"""Helper classes and functions to sample grasps for a given object mesh."""

from __future__ import print_function

import argparse
from collections import OrderedDict
import errno
import json
import os
import numpy as np
import multiprocessing as mp
from functools import partial

from tqdm import tqdm

import trimesh
import trimesh.transformations as tra


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
        rotation = [rotation[3], rotation[0], rotation[1], rotation[2]]
        matrix = tra.quaternion_matrix(rotation)
        matrix[3, 3] = 1.0
        matrix[:3, 3] = position
        self.position = position
        self.rotation = rotation
        self.mesh.apply_transform(matrix)
\
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


class PandaGripper(object):
    """An object representing a Franka Panda gripper."""

    def __init__(self, q=None, num_contact_points_per_finger=10, root_folder=''):
        """Create a Franka Panda parallel-yaw gripper object.

        Keyword Arguments:
            q {list of int} -- configuration (default: {None})
            num_contact_points_per_finger {int} -- contact points per finger (default: {10})
            root_folder {str} -- base folder for model files (default: {''})
        """
        self.joint_limits = [0.0, 0.04]
        self.default_pregrasp_configuration = 0.04

        if q is None:
            q = self.default_pregrasp_configuration

        self.q = q
        fn_base = root_folder + 'assets/gripper_models/panda_gripper/hand.stl'
        fn_finger = root_folder + 'assets/gripper_models/panda_gripper/finger.stl'
        self.base = trimesh.load(fn_base)
        self.finger_l = trimesh.load(fn_finger)
        self.finger_r = self.finger_l.copy()

        # transform fingers relative to the base
        self.finger_l.apply_transform(tra.euler_matrix(0, 0, np.pi))
        self.finger_l.apply_translation([+q, 0, 0.0584])
        self.finger_r.apply_translation([-q, 0, 0.0584])

        self.fingers = trimesh.util.concatenate([self.finger_l, self.finger_r])
        self.hand = trimesh.util.concatenate([self.fingers, self.base])

        self.ray_origins = []
        self.ray_directions = []
        for i in np.linspace(-0.01, 0.02, num_contact_points_per_finger):
            self.ray_origins.append(
                np.r_[self.finger_l.bounding_box.centroid + [0, 0, i], 1])
            self.ray_origins.append(
                np.r_[self.finger_r.bounding_box.centroid + [0, 0, i], 1])
            self.ray_directions.append(
                np.r_[-self.finger_l.bounding_box.primitive.transform[:3, 0]])
            self.ray_directions.append(
                np.r_[+self.finger_r.bounding_box.primitive.transform[:3, 0]])

        self.ray_origins = np.array(self.ray_origins)
        self.ray_directions = np.array(self.ray_directions)

        self.standoff_range = np.array([max(self.finger_l.bounding_box.bounds[0, 2],
                                            self.base.bounding_box.bounds[1, 2]),
                                        self.finger_l.bounding_box.bounds[1, 2]])
        self.standoff_range[0] += 0.001

    def get_obbs(self):
        """Get list of obstacle meshes.

        Returns:
            list of trimesh -- bounding boxes used for collision checking
        """
        return [self.finger_l.bounding_box, self.finger_r.bounding_box, self.base.bounding_box]

    def get_meshes(self):
        """Get list of meshes that this gripper consists of.

        Returns:
            list of trimesh -- visual meshes
        """
        return [self.finger_l, self.finger_r, self.base]

    def get_closing_rays(self, transform):
        """Get an array of rays defining the contact locations and directions on the hand.

        Arguments:
            transform {[nump.array]} -- a 4x4 homogeneous matrix

        Returns:
            numpy.array -- transformed rays (origin and direction)
        """
        return transform[:3, :].dot(
            self.ray_origins.T).T, transform[:3, :3].dot(self.ray_directions.T).T


class RumGripper(object):
    """An object representing a Franka Panda gripper."""

    def __init__(self, q=None, num_contact_points_per_finger=10, root_folder=''):
        
        self.default_pregrasp_configuration = 0.06

        if q is None:
            q = self.default_pregrasp_configuration

        self.q = q

        fn_base = root_folder + 'assets/gripper_models/rum_gripper/meshes/simple_body.stl'
        fn_finger_l = root_folder + 'assets/gripper_models/rum_gripper/meshes/simple_left.stl'
        fn_finger_r = root_folder + 'assets/gripper_models/rum_gripper/meshes/simple_right.stl'
        self.base = trimesh.load(fn_base)
        self.finger_l = trimesh.load(fn_finger_l)
        self.finger_r = trimesh.load(fn_finger_r)

        self.fingers = trimesh.util.concatenate([self.finger_l, self.finger_r])
        self.hand = trimesh.util.concatenate([self.fingers, self.base])

        self.standoff_range = np.array([
            max(self.finger_l.bounding_box.bounds[0, 2],
                self.base.bounding_box.bounds[1, 2]),
            self.finger_l.bounding_box.bounds[1, 2]
        ])
        self.standoff_range[0] += 0.001

        self.ray_origins = []
        self.ray_directions = []

        for i in np.linspace(-0.01, 0.04, num_contact_points_per_finger):
            self.ray_origins.append(np.r_[self.finger_l.bounding_box.centroid + [0, 0, i], 1] + [i/np.sqrt(2) + 0.01, 0, 0, 0])
            self.ray_origins.append(np.r_[self.finger_r.bounding_box.centroid + [0, 0, i], 1] - [i/np.sqrt(2) + 0.01, 0, 0, 0])
            self.ray_directions.append(np.r_[-self.finger_l.bounding_box.primitive.transform[:3, 0]])
            self.ray_directions.append(np.r_[+self.finger_r.bounding_box.primitive.transform[:3, 0]])

        self.ray_origins = np.array(self.ray_origins)
        self.ray_directions = np.array(self.ray_directions)

    def get_finger_meshes(self):
        return [self.finger_l, self.finger_r]

    def get_base_mesh(self):
        return self.base

    def get_base_obb(self):
        return self.base.bounding_box
    
    def get_obbs(self):
        return [self.finger_l.bounding_box, self.finger_r.bounding_box]

    def get_closing_rays(self, transform):
        """Get an array of rays defining the contact locations and directions on the hand.

        Arguments:
            transform {[nump.array]} -- a 4x4 homogeneous matrix

        Returns:
            numpy.array -- transformed rays (origin and direction)
        """
        return transform[:3, :].dot(
            self.ray_origins.T).T, transform[:3, :3].dot(self.ray_directions.T).T


def compute_grasp_widths(transforms, object_mesh, gripper_name='panda'):
    from trimesh.ray.ray_triangle import RayMeshIntersector
    import trimesh

    gripper = create_gripper(gripper_name)
    widths = []

    # Use Embree if available
    if trimesh.ray.has_embree:
        intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(object_mesh)
    else:
        intersector = RayMeshIntersector(object_mesh)

    for tf in transforms:
        ray_origins, ray_directions = gripper.get_closing_rays(tf)

        # Intersect rays with mesh
        locations, index_ray, index_tri = intersector.intersects_location(
            ray_origins, ray_directions, multiple_hits=False)

        if len(locations) < 2:
            widths.append(0.0)
            continue

        # Finger ray indices: even = left, odd = right
        left_hits = [(i, loc) for i, loc in zip(index_ray, locations) if i % 2 == 0]
        right_hits = [(i, loc) for i, loc in zip(index_ray, locations) if i % 2 == 1]

        if not left_hits or not right_hits:
            widths.append(0.0)
            continue

        # Choose the nearest contact point (shortest distance from ray origin)
        def closest_hit(hits):
            return min(hits, key=lambda x: np.linalg.norm(ray_origins[x[0]][:3] - x[1]))[1]

        contact_left = closest_hit(left_hits)
        contact_right = closest_hit(right_hits)

        # Compute Euclidean distance between contacts
        width = np.linalg.norm(contact_left - contact_right)
        widths.append(width)

    return widths

def get_available_grippers():
    """Get list of names of all available grippers.

    Returns:
        list of str -- a list of names for the gripper factory
    """
    available_grippers = OrderedDict({
        'panda': PandaGripper,
        'rum': RumGripper,
    })
    return available_grippers


def create_gripper(name, configuration=None, root_folder=''):
    """Create a gripper object.

    Arguments:
        name {str} -- name of the gripper

    Keyword Arguments:
        configuration {list of float} -- configuration (default: {None})
        root_folder {str} -- base folder for model files (default: {''})

    Raises:
        Exception: If the gripper name is unknown.

    Returns:
        [type] -- gripper object
    """
    if name.lower() == 'panda':
        return PandaGripper(q=configuration, root_folder=root_folder)
    elif name.lower() == 'rum':
        return RumGripper(q=configuration, root_folder=root_folder)
    else:
        raise Exception("Unknown gripper: {}".format(name))


def _check_collision_worker(object_mesh, gripper_mesh, transform_batch):
    """Worker function to check collisions for a batch of transforms.
    
    Arguments:
        object_mesh {trimesh} -- mesh of object
        gripper_mesh {trimesh} -- mesh of gripper
        transform_batch {list} -- batch of transforms to check
        
    Returns:
        list -- minimum distances for each transform
    """
    manager = trimesh.collision.CollisionManager()
    manager.add_object('object', object_mesh)
    min_distances = []
    
    for tf in transform_batch:
        min_distances.append(manager.min_distance_single(gripper_mesh, transform=tf))
        
    return min_distances


def _quality_point_contacts_worker(batch_data):
    """Worker function for processing point contact quality assessment in parallel.
    
    Arguments:
        batch_data {tuple} -- (transform_batch, collision_batch, object_mesh, gripper_name)
        
    Returns:
        list -- quality scores for each grasp in the batch
    """
    transform_batch, collision_batch, object_mesh, gripper_name = batch_data
    
    res = []
    gripper = create_gripper(gripper_name)
    
    if trimesh.ray.has_embree:
        intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(
            object_mesh, scale_to_box=True)
    else:
        intersector = trimesh.ray.ray_triangle.RayMeshIntersector(object_mesh)
        
    for p, colliding in zip(transform_batch, collision_batch):
        if colliding:
            res.append(-1)
        else:
            ray_origins, ray_directions = gripper.get_closing_rays(p)
            locations, index_ray, index_tri = intersector.intersects_location(
                ray_origins, ray_directions, multiple_hits=False)

            if len(locations) == 0:
                res.append(0)
            else:
                # this depends on the width of the gripper
                valid_locations = np.linalg.norm(
                    ray_origins[index_ray]-locations, axis=1) < 2.0*gripper.q

                if sum(valid_locations) == 0:
                    res.append(0)
                else:
                    contact_normals = object_mesh.face_normals[index_tri[valid_locations]]
                    motion_normals = ray_directions[index_ray[valid_locations]]
                    dot_prods = (motion_normals * contact_normals).sum(axis=1)
                    res.append(np.cos(dot_prods).sum() / len(ray_origins))
    
    return res


def _quality_antipodal_worker(batch_data):
    """Worker function for processing antipodal quality assessment in parallel.
    
    Arguments:
        batch_data {tuple} -- (transform_batch, collision_batch, object_mesh, gripper_name)
        
    Returns:
        list -- quality scores for each grasp in the batch
    """
    transform_batch, collision_batch, object_mesh, gripper_name = batch_data
    
    res = []
    gripper = create_gripper(gripper_name)
    
    if trimesh.ray.has_embree:
        intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(
            object_mesh, scale_to_box=True)
    else:
        intersector = trimesh.ray.ray_triangle.RayMeshIntersector(object_mesh)
        
    for p, colliding in zip(transform_batch, collision_batch):
        if colliding:
            res.append(0)
            continue
        
        ray_origins, ray_directions = gripper.get_closing_rays(p)
        locations, index_ray, index_tri = intersector.intersects_location(
            ray_origins, ray_directions, multiple_hits=False)

        if locations.size == 0:
            res.append(0)
            continue
            
        # chose contact points for each finger [they are stored in an alternating fashion]
        index_ray_left = np.array([i for i, num in enumerate(
            index_ray) if num % 2 == 0 and np.linalg.norm(ray_origins[num]-locations[i]) < 2.0*gripper.q])
        index_ray_right = np.array([i for i, num in enumerate(
            index_ray) if num % 2 == 1 and np.linalg.norm(ray_origins[num]-locations[i]) < 2.0*gripper.q])

        if index_ray_left.size == 0 or index_ray_right.size == 0:
            res.append(0)
            continue
            
        # select the contact point closest to the finger (which would be hit first during closing)
        left_contact_idx = np.linalg.norm(
            ray_origins[index_ray[index_ray_left]] - locations[index_ray_left], axis=1).argmin()
        right_contact_idx = np.linalg.norm(
            ray_origins[index_ray[index_ray_right]] - locations[index_ray_right], axis=1).argmin()
        left_contact_point = locations[index_ray_left[left_contact_idx]]
        right_contact_point = locations[index_ray_right[right_contact_idx]]

        left_contact_normal = object_mesh.face_normals[index_tri[index_ray_left[left_contact_idx]]]
        right_contact_normal = object_mesh.face_normals[
            index_tri[index_ray_right[right_contact_idx]]]

        l_to_r = (right_contact_point - left_contact_point) / \
            np.linalg.norm(right_contact_point -
                          left_contact_point)
        r_to_l = (left_contact_point - right_contact_point) / \
            np.linalg.norm(left_contact_point -
                          right_contact_point)

        qual_left = np.dot(left_contact_normal, r_to_l)
        qual_right = np.dot(right_contact_normal, l_to_r)
        if qual_left < 0 or qual_right < 0:
            qual = 0
        else:
            qual = min(qual_left, qual_right)
        
        # Always append the quality score
        res.append(qual)
    
    return res


def in_collision_with_gripper(object_mesh, gripper_transforms, gripper_name, silent=False, num_workers=None):
    """Check collision of object with gripper.

    Arguments:
        object_mesh {trimesh} -- mesh of object
        gripper_transforms {list of numpy.array} -- homogeneous matrices of gripper
        gripper_name {str} -- name of gripper

    Keyword Arguments:
        silent {bool} -- verbosity (default: {False})
        num_workers {int} -- number of parallel workers (default: {None}, uses CPU count)

    Returns:
        [list of bool] -- Which gripper poses are in collision with object mesh
    """
    if num_workers is None:
        num_workers = mp.cpu_count()
    
    # For small numbers of transforms, it's faster to use the sequential version
    if len(gripper_transforms) < 100 or num_workers <= 1:
        manager = trimesh.collision.CollisionManager()
        manager.add_object('object', object_mesh)
        gripper_meshes = [create_gripper(gripper_name).hand]
        min_distance = []
        for tf in tqdm(gripper_transforms, disable=silent):
            min_distance.append(np.min([manager.min_distance_single(
                gripper_mesh, transform=tf) for gripper_mesh in gripper_meshes]))

        return [d == 0 for d in min_distance], min_distance
    
    # Use parallel processing for larger numbers of transforms
    gripper_mesh = create_gripper(gripper_name).hand
    
    # Split transforms into batches for each worker
    num_transforms = len(gripper_transforms)
    batch_size = max(1, num_transforms // num_workers)
    batches = [gripper_transforms[i:i+batch_size] for i in range(0, num_transforms, batch_size)]
    
    # Create a partial function with fixed arguments
    worker_func = partial(_check_collision_worker, object_mesh, gripper_mesh)
    
    # Use a pool of workers to process batches in parallel
    min_distances = []
    
    # Setup progress bar to track total transforms, not batches
    pbar = tqdm(
        total=num_transforms, 
        disable=silent,
        desc=f"Checking collisions (using {num_workers} workers)"
    )
    
    with mp.Pool(processes=num_workers) as pool:
        for batch_result in pool.imap(worker_func, batches):
            min_distances.extend(batch_result)
            pbar.update(len(batch_result))  # Update by actual number of transforms processed
    
    pbar.close()
    
    return [d == 0 for d in min_distances], min_distances


def grasp_quality_point_contacts(transforms, collisions, object_mesh, gripper_name='panda', silent=False, num_workers=None):
    """Grasp quality function

    Arguments:
        transforms {[type]} -- grasp poses
        collisions {[type]} -- collision information
        object_mesh {trimesh} -- object mesh

    Keyword Arguments:
        gripper_name {str} -- name of gripper (default: {'panda'})
        silent {bool} -- verbosity (default: {False})
        num_workers {int} -- number of parallel workers (default: {None})

    Returns:
        list of float -- quality of grasps [0..1]
    """
    if num_workers is None:
        num_workers = mp.cpu_count()
    
    # For small numbers, just use sequential processing
    if len(transforms) < 100 or num_workers <= 1:
        res = []
        gripper = create_gripper(gripper_name)
        if trimesh.ray.has_embree:
            intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(
                object_mesh, scale_to_box=True)
        else:
            intersector = trimesh.ray.ray_triangle.RayMeshIntersector(object_mesh)
            
        for p, colliding in tqdm(zip(transforms, collisions), total=len(transforms), disable=silent):
            if colliding:
                res.append(-1)
            else:
                ray_origins, ray_directions = gripper.get_closing_rays(p)
                locations, index_ray, index_tri = intersector.intersects_location(
                    ray_origins, ray_directions, multiple_hits=False)

                if len(locations) == 0:
                    res.append(0)
                else:
                    # this depends on the width of the gripper
                    valid_locations = np.linalg.norm(
                        ray_origins[index_ray]-locations, axis=1) < 2.0*gripper.q

                    if sum(valid_locations) == 0:
                        res.append(0)
                    else:
                        contact_normals = object_mesh.face_normals[index_tri[valid_locations]]
                        motion_normals = ray_directions[index_ray[valid_locations]]
                        dot_prods = (motion_normals * contact_normals).sum(axis=1)
                        res.append(np.cos(dot_prods).sum() / len(ray_origins))
        return res
    
    # Use parallel processing for larger numbers
    # Split into batches for each worker
    batch_size = max(1, len(transforms) // num_workers)
    transform_batches = [transforms[i:i+batch_size] for i in range(0, len(transforms), batch_size)]
    collision_batches = [collisions[i:i+batch_size] for i in range(0, len(collisions), batch_size)]
    
    # Create batch data
    batch_data = [(t_batch, c_batch, object_mesh, gripper_name) 
                 for t_batch, c_batch in zip(transform_batches, collision_batches)]
    
    # Process in parallel
    all_results = []
    with mp.Pool(processes=num_workers) as pool:
        pbar = tqdm(
            total=len(transforms), 
            disable=silent,
            desc=f"Computing point contact quality (using {num_workers} workers)"
        )
        
        for result in pool.imap(_quality_point_contacts_worker, batch_data):
            all_results.extend(result)
            pbar.update(len(result))
        
        pbar.close()
    
    return all_results


def grasp_quality_antipodal(transforms, collisions, object_mesh, gripper_name='panda', silent=False, num_workers=None):
    """Grasp quality function.

    Arguments:
        transforms {numpy.array} -- grasps
        collisions {list of bool} -- collision information
        object_mesh {trimesh} -- object mesh

    Keyword Arguments:
        gripper_name {str} -- name of gripper (default: {'panda'})
        silent {bool} -- verbosity (default: {False})
        num_workers {int} -- number of parallel workers (default: {None})

    Returns:
        list of float -- quality of grasps [0..1]
    """
    if num_workers is None:
        num_workers = mp.cpu_count()
    
    # For small numbers, just use sequential processing
    if len(transforms) < 100 or num_workers <= 1:
        res = []
        gripper = create_gripper(gripper_name)
        if trimesh.ray.has_embree:
            intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(
                object_mesh, scale_to_box=True)
        else:
            intersector = trimesh.ray.ray_triangle.RayMeshIntersector(object_mesh)
            
        for p, colliding in tqdm(zip(transforms, collisions), total=len(transforms), disable=silent):
            if colliding:
                res.append(0)
                continue
                
            ray_origins, ray_directions = gripper.get_closing_rays(p)
            locations, index_ray, index_tri = intersector.intersects_location(
                ray_origins, ray_directions, multiple_hits=False)

            if locations.size == 0:
                res.append(0)
                continue
                
            # chose contact points for each finger [they are stored in an alternating fashion]
            index_ray_left = np.array([i for i, num in enumerate(
                index_ray) if num % 2 == 0 and np.linalg.norm(ray_origins[num]-locations[i]) < 2.0*gripper.q])
            index_ray_right = np.array([i for i, num in enumerate(
                index_ray) if num % 2 == 1 and np.linalg.norm(ray_origins[num]-locations[i]) < 2.0*gripper.q])

            if index_ray_left.size == 0 or index_ray_right.size == 0:
                res.append(0)
                continue
                
            # select the contact point closest to the finger (which would be hit first during closing)
            left_contact_idx = np.linalg.norm(
                ray_origins[index_ray[index_ray_left]] - locations[index_ray_left], axis=1).argmin()
            right_contact_idx = np.linalg.norm(
                ray_origins[index_ray[index_ray_right]] - locations[index_ray_right], axis=1).argmin()
            left_contact_point = locations[index_ray_left[left_contact_idx]]
            right_contact_point = locations[index_ray_right[right_contact_idx]]

            left_contact_normal = object_mesh.face_normals[index_tri[index_ray_left[left_contact_idx]]]
            right_contact_normal = object_mesh.face_normals[
                index_tri[index_ray_right[right_contact_idx]]]

            l_to_r = (right_contact_point - left_contact_point) / \
                np.linalg.norm(right_contact_point -
                               left_contact_point)
            r_to_l = (left_contact_point - right_contact_point) / \
                np.linalg.norm(left_contact_point -
                               right_contact_point)

            qual_left = np.dot(left_contact_normal, r_to_l)
            qual_right = np.dot(right_contact_normal, l_to_r)
            if qual_left < 0 or qual_right < 0:
                qual = 0
            else:
                qual = min(qual_left, qual_right)
            
            # Always append the quality score
            res.append(qual)
        return res
    
    # Use parallel processing for larger numbers
    # Split into batches for each worker
    batch_size = max(1, len(transforms) // num_workers)
    transform_batches = [transforms[i:i+batch_size] for i in range(0, len(transforms), batch_size)]
    collision_batches = [collisions[i:i+batch_size] for i in range(0, len(collisions), batch_size)]
    
    # Create batch data
    batch_data = [(t_batch, c_batch, object_mesh, gripper_name) 
                 for t_batch, c_batch in zip(transform_batches, collision_batches)]
    
    # Process in parallel
    all_results = []
    with mp.Pool(processes=num_workers) as pool:
        pbar = tqdm(
            total=len(transforms), 
            disable=silent,
            desc=f"Computing antipodal quality (using {num_workers} workers)"
        )
        
        for result in pool.imap(_quality_antipodal_worker, batch_data):
            all_results.extend(result)
            pbar.update(len(result))
        
        pbar.close()
    
    return all_results


def _raycast_collision_worker(object_mesh, origins_batch, expected_points_batch):
    """Worker function for raycast collision checking.
    
    Arguments:
        object_mesh {trimesh} -- mesh to check collisions against
        origins_batch {np.array} -- batch of origins
        expected_points_batch {np.array} -- batch of expected hit points
        
    Returns:
        np.array -- boolean array of valid collisions
    """
    if trimesh.ray.has_embree:
        intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(
            object_mesh, scale_to_box=True)
    else:
        intersector = trimesh.ray.ray_triangle.RayMeshIntersector(object_mesh)
        
    locations, index_rays, _ = intersector.intersects_location(
        origins_batch[:, :3, 3], origins_batch[:, :3, 2], multiple_hits=False)
        
    res = np.array([False] * len(origins_batch))
    res[index_rays] = np.all(np.isclose(
        locations, expected_points_batch[index_rays]), axis=1)
        
    return res

def raycast_collisioncheck(origins, expected_hit_points, object_mesh, num_workers=None):
    """ Check whether a set of ray casts turn out as expected.

    :param origins: ray origins and directions as Nx4x4 homogenous matrices (use last two columns)
    :param expected_hit_points: 3d points Nx3
    :param object_mesh: trimesh mesh instance
    :param num_workers: number of workers for parallel processing (ignored to avoid nested pools)

    :return: boolean array of size N
    """
    assert len(origins) == len(expected_hit_points)
    
    # Always use sequential mode when called from a worker process
    # to avoid nested process pools
    if trimesh.ray.has_embree:
        intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(
            object_mesh, scale_to_box=True)
    else:
        intersector = trimesh.ray.ray_triangle.RayMeshIntersector(object_mesh)

    locations, index_rays, _ = intersector.intersects_location(
        origins[:, :3, 3], origins[:, :3, 2], multiple_hits=False)
    res = np.array([False] * len(origins))
    res[index_rays] = np.all(np.isclose(
        locations, expected_hit_points[index_rays]), axis=1)

    return res


def _process_points_batch(batch_data):
    """Process a batch of points for systematic sampling in parallel.
    
    Arguments:
        batch_data {tuple} -- (points, normals, rotation_samples, standoff_samples, gripper_name, mesh)
        
    Returns:
        tuple -- (points, normals, transforms, roll_angles, standoffs, position_idx)
    """
    points_batch, normals_batch, rotation_samples, standoff_samples, gripper_name, mesh = batch_data
    
    batch_position_idx = []
    batch_points = []
    batch_normals = []
    batch_roll_angles = []
    batch_standoffs = []
    batch_transforms = []
    
    # Pre-allocate for better efficiency
    total_combinations = len(points_batch) * len(rotation_samples) * len(standoff_samples)
    batch_transforms = np.zeros((total_combinations, 4, 4))
    all_points = np.zeros((total_combinations, 3))
    all_normals = np.zeros((total_combinations, 3))
    all_roll_angles = np.zeros(total_combinations)
    all_standoffs = np.zeros(total_combinations)
    all_position_idx = np.zeros(total_combinations, dtype=int)
    
    idx = 0
    for i, (point, normal) in enumerate(zip(points_batch, normals_batch)):
        for roll in rotation_samples:
            orientation = tra.quaternion_matrix(
                tra.quaternion_about_axis(roll, [0, 0, 1]))
            
            for standoff in standoff_samples:
                origin = point + normal * standoff
                transform = np.dot(np.dot(tra.translation_matrix(origin), 
                                         trimesh.geometry.align_vectors([0, 0, -1], normal)),
                                   orientation)
                
                all_points[idx] = point
                all_normals[idx] = normal
                all_roll_angles[idx] = roll
                all_standoffs[idx] = standoff
                all_position_idx[idx] = i
                batch_transforms[idx] = transform
                idx += 1
    
    # Filter by raycast collision check - use sequential mode to avoid nested process pools
    # When inside a worker process, we must not create another process pool
    if trimesh.ray.has_embree:
        intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(
            mesh, scale_to_box=True)
    else:
        intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)

    locations, index_rays, _ = intersector.intersects_location(
        batch_transforms[:, :3, 3], batch_transforms[:, :3, 2], multiple_hits=False)
    valid = np.array([False] * len(batch_transforms))
    valid[index_rays] = np.all(np.isclose(
        locations, all_points[index_rays]), axis=1)
    
    return (
        all_points[valid],
        all_normals[valid],
        batch_transforms[valid],
        all_roll_angles[valid],
        all_standoffs[valid],
        all_position_idx[valid]
    )


def _process_random_points(batch_data):
    """Process a batch of points for random sampling in parallel.
    
    Arguments:
        batch_data {tuple} -- (points, normals, gripper, mesh)
        
    Returns:
        tuple -- (points, normals, transforms, roll_angles, standoffs)
    """
    points_batch, normals_batch, gripper, mesh = batch_data
    
    num_points = len(points_batch)
    batch_points = np.array(points_batch)
    batch_normals = np.array(normals_batch)
    batch_transforms = np.zeros((num_points, 4, 4))
    batch_roll_angles = np.zeros(num_points)
    batch_standoffs = np.zeros(num_points)
    
    # Generate all transforms at once
    angles = np.random.rand(num_points) * 2 * np.pi
    batch_roll_angles[:] = angles
    
    # Compute standoffs - random value within range
    standoff_range = gripper.standoff_range
    standoffs = (standoff_range[1] - standoff_range[0]) * np.random.rand(num_points) + standoff_range[0]
    batch_standoffs[:] = standoffs
    
    # Compute origins
    origins = batch_points + batch_normals * standoffs[:, np.newaxis]
    
    # Create transformations
    for i, (origin, normal, angle) in enumerate(zip(origins, batch_normals, angles)):
        orientation = tra.quaternion_matrix(
            tra.quaternion_about_axis(angle, [0, 0, 1]))
        batch_transforms[i] = np.dot(
            np.dot(tra.translation_matrix(origin),
                  trimesh.geometry.align_vectors([0, 0, -1], normal)),
            orientation)
    
    # No need to perform raycast collision check for random sampling
    # This will be done later in the collision checking step
    
    return (
        batch_points,
        batch_normals,
        batch_transforms,
        batch_roll_angles,
        batch_standoffs
    )


def sample_multiple_grasps(number_of_candidates, mesh, gripper_name, systematic_sampling,
                           surface_density=0.005*0.005, standoff_density=0.01, roll_density=15,
                           type_of_quality='antipodal',
                           min_quality=-1.0,
                           silent=False,
                           num_workers=None):
    """Sample a set of grasps for an object.

    Arguments:
        number_of_candidates {int} -- Number of grasps to sample
        mesh {trimesh} -- Object mesh
        gripper_name {str} -- Name of gripper model
        systematic_sampling {bool} -- Whether to use grid sampling for roll

    Keyword Arguments:
        surface_density {float} -- surface density, in m^2 (default: {0.005*0.005})
        standoff_density {float} -- density for standoff, in m (default: {0.01})
        roll_density {float} -- roll density, in deg (default: {15})
        type_of_quality {str} -- quality metric (default: {'antipodal'})
        min_quality {float} -- minimum grasp quality (default: {-1})
        silent {bool} -- verbosity (default: {False})
        num_workers {int} -- number of parallel processes (default: {None})

    Raises:
        Exception: Unknown quality metric

    Returns:
        [type] -- points, normals, transforms, roll_angles, standoffs, collisions, quality
    """
    # Set up multiprocessing
    if num_workers is None:
        num_workers = mp.cpu_count()
    
    # Initialize empty lists
    transforms = []
    points = []
    normals = []
    roll_angles = []
    standoffs = []

    gripper = create_gripper(gripper_name)
    verboseprint = print if not silent else lambda *a, **k: None

    if systematic_sampling:
        # systematic sampling. input:
        # - Surface density:
        # - Standoff density:
        # - Rotation density:
        # Resulting number of samples:
        # (Area/Surface Density) * (Finger length/Standoff density) * (360/Rotation Density)
        surface_samples = int(np.ceil(mesh.area / surface_density))
        standoff_samples = np.linspace(
            gripper.standoff_range[0],
            gripper.standoff_range[1],
            max(1, int((gripper.standoff_range[1] - gripper.standoff_range[0]) / standoff_density))
        )

        rotation_samples = np.arange(0, 1 * np.pi, np.deg2rad(roll_density))

        # Sample points on mesh surface
        tmp_points, face_indices = mesh.sample(
            surface_samples, return_index=True)
        tmp_normals = mesh.face_normals[face_indices]

        estimated_candidates = len(tmp_points) * \
            len(standoff_samples) * len(rotation_samples)
            
        if not silent:
            verboseprint(f"Estimated number of samples: {estimated_candidates:,} ({len(tmp_points):,} points × {len(standoff_samples)} standoffs × {len(rotation_samples)} rotations)")

        # Prepare for parallel processing
        # Split points into batches for each worker
        batch_size = max(1, len(tmp_points) // num_workers)
        point_batches = [tmp_points[i:i+batch_size] for i in range(0, len(tmp_points), batch_size)]
        normal_batches = [tmp_normals[i:i+batch_size] for i in range(0, len(tmp_normals), batch_size)]
        
        # Create batch data for parallel processing
        batch_data = [(points_batch, normals_batch, rotation_samples, standoff_samples, gripper_name, mesh)
                      for points_batch, normals_batch in zip(point_batches, normal_batches)]
        
        # Process batches in parallel
        all_points = []
        all_normals = []
        all_transforms = []
        all_roll_angles = []
        all_standoffs = []
        all_position_idx = []
        
        verboseprint("Sampling grasps in parallel...")
        with mp.Pool(processes=num_workers) as pool:
            batch_total = sum(len(pb) for pb in point_batches) * len(rotation_samples) * len(standoff_samples)
            pbar = tqdm(
                total=batch_total,
                disable=silent,
                desc=f"Sampling grasps (using {num_workers} workers)"
            )
            
            valid_count = 0
            processed_count = 0
            
            for result in pool.imap(_process_points_batch, batch_data):
                batch_points, batch_normals, batch_transforms, batch_roll_angles, batch_standoffs, batch_position_idx = result
                
                if len(batch_points) > 0:  # Only add if we got valid results
                    all_points.extend(batch_points)
                    all_normals.extend(batch_normals)
                    all_transforms.extend(batch_transforms)
                    all_roll_angles.extend(batch_roll_angles)
                    all_standoffs.extend(batch_standoffs)
                    all_position_idx.extend(batch_position_idx)
                    valid_count += len(batch_points)
                
                # Estimate how many grasps were processed in this batch
                processed_count += len(point_batches[0]) * len(rotation_samples) * len(standoff_samples)
                pbar.update(len(point_batches[0]) * len(rotation_samples) * len(standoff_samples))
                pbar.set_postfix({"Valid": valid_count})
            
            pbar.close()
        
        points = np.array(all_points)
        normals = np.array(all_normals)
        transforms = np.array(all_transforms)
        roll_angles = np.array(all_roll_angles)
        standoffs = np.array(all_standoffs)
        position_idx = np.array(all_position_idx)
        
        verboseprint(f"Generated {len(transforms):,} valid grasps after sampling")
        
    else:
        # Random sampling
        points, face_indices = mesh.sample(
            number_of_candidates, return_index=True)
        normals = mesh.face_normals[face_indices]
        
        # Prepare for parallel processing
        batch_size = max(1, len(points) // num_workers)
        point_batches = [points[i:i+batch_size] for i in range(0, len(points), batch_size)]
        normal_batches = [normals[i:i+batch_size] for i in range(0, len(normals), batch_size)]
        
        # Create batch data
        batch_data = [(points_batch, normals_batch, gripper, mesh) 
                     for points_batch, normals_batch in zip(point_batches, normal_batches)]
        
        # Process in parallel
        all_points = []
        all_normals = []
        all_transforms = []
        all_roll_angles = []
        all_standoffs = []
        
        verboseprint("Sampling grasps in parallel...")
        with mp.Pool(processes=num_workers) as pool:
            total_points = sum(len(pb) for pb in point_batches)
            pbar = tqdm(
                total=total_points,
                disable=silent,
                desc=f"Sampling grasps (using {num_workers} workers)"
            )
            
            for result in pool.imap(_process_random_points, batch_data):
                batch_points, batch_normals, batch_transforms, batch_roll_angles, batch_standoffs = result
                
                all_points.extend(batch_points)
                all_normals.extend(batch_normals)
                all_transforms.extend(batch_transforms)
                all_roll_angles.extend(batch_roll_angles)
                all_standoffs.extend(batch_standoffs)
                
                pbar.update(len(batch_points))
            
            pbar.close()
        
        points = np.array(all_points)
        normals = np.array(all_normals)
        transforms = np.array(all_transforms)
        roll_angles = np.array(all_roll_angles)
        standoffs = np.array(all_standoffs)
        
        verboseprint(f"Generated {len(transforms):,} grasps with random sampling")

    verboseprint("Checking collisions...")
    collisions, _ = in_collision_with_gripper(
        mesh, transforms, gripper_name=gripper_name, silent=silent, num_workers=num_workers)

    verboseprint("Labelling grasps...")
    quality = {}
    quality_key = 'quality_' + type_of_quality
    if type_of_quality == 'antipodal':
        quality[quality_key] = grasp_quality_antipodal(
            transforms, collisions, object_mesh=mesh, gripper_name=gripper_name, 
            silent=silent, num_workers=num_workers)
    elif type_of_quality == 'number_of_contacts':
        quality[quality_key] = grasp_quality_point_contacts(
            transforms, collisions, object_mesh=mesh, gripper_name=gripper_name, 
            silent=silent, num_workers=num_workers)
    else:
        raise Exception("Quality metric unknown: ", quality)

    # Filter out by quality
    quality_np = np.array(quality[quality_key])
    collisions = np.array(collisions)

    f_points = []
    f_normals = []
    f_transforms = []
    f_roll_angles = []
    f_standoffs = []
    f_collisions = []
    f_quality = []

    for i, _ in enumerate(transforms):
        if quality_np[i] >= min_quality:
            f_points.append(points[i])
            f_normals.append(normals[i])
            f_transforms.append(transforms[i])
            f_roll_angles.append(roll_angles[i])
            f_standoffs.append(standoffs[i])
            f_collisions.append(int(collisions[i]))
            f_quality.append(quality_np[i])

    points = np.array(f_points)
    normals = np.array(f_normals)
    transforms = np.array(f_transforms)
    roll_angles = np.array(f_roll_angles)
    standoffs = np.array(f_standoffs)
    collisions = f_collisions
    quality[quality_key] = f_quality
    
    verboseprint(f"Final result: {len(transforms):,} valid grasps with quality >= {min_quality}")

    return points, normals, transforms, roll_angles, standoffs, collisions, quality


def make_parser():
    """Create program arguments and default values.

    Returns:
        argparse.ArgumentParser -- an argument parser
    """
    parser = argparse.ArgumentParser(description='Sample grasps for an object.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--object_file', type=str,
                        default='/home/arsalan/data/models_selected/03797390/1be6b2c84cdab826c043c2d07bb83fc8/model.obj',
                        help='Number of samples.')
    parser.add_argument('--dataset', type=str, default='UNKNOWN',
                        help='Metadata about the origin of the file.')
    parser.add_argument('--classname', type=str, default='UNKNOWN',
                        help='Metadata about the class of the object.')
    parser.add_argument('--scale', type=float, default=1.0,
                        help='Scale the object.')
    parser.add_argument('--resize', type=float,
                        help="""Resize the object, such that the longest of its \
                            bounding box dimensions is of length --resize.""")
    parser.add_argument('--use_stl', action='store_true',
                        help='Use STL instead of obj.')
    parser.add_argument('--gripper', choices=get_available_grippers().keys(), default='rum',
                        help='Type of gripper.')
    parser.add_argument('--quality', choices=['number_of_contacts', 'antipodal'],
                        default='number_of_contacts',
                        help='Which type of quality metric to evaluate.')
    
    parser.add_argument('--position', type=float, nargs=3, default=[0, 0, 0],
                        help='Position of the object in the world frame (x, y, z).')
    parser.add_argument('--rotation', type=float, nargs=4, default=[0, 0, 0, 1],
                        help='Rotation of the object in quaternion format (x, y, z, w).')

    parser.add_argument('--single_standoff', action='store_true',
                        help='Use the closest possible standoff.')

    parser.add_argument('--systematic_sampling', action='store_true',
                        help='Systematically sample stuff.')
    parser.add_argument('--systematic_surface_density', type=float, default=0.005*0.005,
                        help='Surface density used for systematic sampling (in square meters).')
    parser.add_argument('--systematic_standoff_density', type=float, default=0.01,
                        help='Standoff density used for systematic sampling (in meters).')
    parser.add_argument('--systematic_roll_density', type=float, default=15.,
                        help='Roll density used for systematic sampling (in degrees).')
    parser.add_argument('--filter_best_per_position', action='store_true',
                        help='Only store one grasp (highest quality) if there are multiple per with the same position.')

    parser.add_argument('--min_quality', type=float, default=0.5,
                        help="""Only store grasps whose quality is at least this value. \
                            Colliding grasps have quality -1, i.e. they are filtered out by default.""")

    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples.')
    parser.add_argument('--output', type=str, default="grasps.json",
                        help='File to store the results (json).')
    parser.add_argument('--add_quality_metric', nargs=2, type=str, default="",
                        help='File (json) to calculate additional quality metric for.')
    parser.add_argument('--silent', action='store_true',
                        help='No commandline output.')

    parser.add_argument('--force', action='store_true',
                        help='Do things my way.')
                        
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of parallel workers to use for collision checking. Default uses all available CPU cores.')

    return parser


def verboseprint(*args, **kwargs):
    """Helper function to print verbose output."""
    pass

if __name__ == "__main__":
    # This guard is important for multiprocessing to work correctly
    parser = make_parser()
    args = parser.parse_args()
    
    # Define global verboseprint function
    verboseprint = print if not args.silent else lambda *a, **k: None

    if args.add_quality_metric:
        with open(args.add_quality_metric[1], 'r') as f:
            grasps = json.load(f)
        obj = Object(grasps['object'].replace('.obj', '.stl')
                     if args.use_stl else grasps['object'])
        obj.rescale(grasps['object_scale'])

        grasp_tfs = np.array(grasps['transforms'])
        collisions = np.array(grasps['collisions'])

        key = 'quality_{}'.format(args.add_quality_metric[0])

        if key in grasps.keys() and not args.force:
            raise Exception(
                "Quality metric already part of json file! (Needs --force option) ", key)

        if key == 'quality_number_of_contacts':
            grasps[key] = grasp_quality_point_contacts(
                grasp_tfs,
                collisions,
                object_mesh=obj.mesh,
                gripper_name=grasps['gripper'],
                silent=args.silent,
                num_workers=args.num_workers)
        elif key == 'quality_antipodal':
            grasps[key] = grasp_quality_antipodal(
                grasp_tfs,
                collisions,
                object_mesh=obj.mesh,
                gripper_name=grasps['gripper'],
                silent=args.silent,
                num_workers=args.num_workers)
        else:
            raise Exception("Unknown quality metric: ", key)

        with open(args.add_quality_metric[1], 'w') as f:
            json.dump(grasps, f)

    else:
        if os.path.dirname(args.output) != '':
            try:
                os.makedirs(os.path.dirname(args.output))
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

        obj = Object(args.object_file.replace('.obj', '.stl')
                     if args.use_stl else args.object_file)

        if args.resize:
            obj.resize(args.resize)
        else:
            obj.rescale(args.scale)

        obj.set_transform(
            position=args.position,
            rotation=args.rotation,
        )
        gripper = create_gripper(args.gripper)

        points, normals, transforms, roll_angles, standoffs, collisions, qualities\
            = sample_multiple_grasps(args.num_samples,
                                     obj.mesh,
                                     gripper_name=args.gripper,
                                     systematic_sampling=args.systematic_sampling,
                                     roll_density=args.systematic_roll_density,
                                     standoff_density=args.systematic_standoff_density,
                                     surface_density=args.systematic_surface_density,
                                     type_of_quality=args.quality,
                                    #  filter_best_per_position=args.filter_best_per_position,
                                     min_quality=args.min_quality,
                                     silent=args.silent,
                                     num_workers=args.num_workers)
        
        grasp_widths = compute_grasp_widths(transforms, obj.mesh, gripper_name=args.gripper)
        
        # Sort all features by quality score (highest quality first)
        quality_key = 'quality_' + args.quality
        quality_scores = qualities[quality_key]
        sort_indices = np.argsort(quality_scores)[::-1]  # Descending order
        
        # Apply sorting to all arrays/lists
        transforms = transforms[sort_indices]
        points = points[sort_indices]
        normals = normals[sort_indices]
        roll_angles = roll_angles[sort_indices]
        standoffs = standoffs[sort_indices]
        collisions = [collisions[i] for i in sort_indices]
        grasp_widths = [grasp_widths[i] for i in sort_indices]
        sorted_quality_scores = [quality_scores[i] for i in sort_indices]
        
        verboseprint(f"Sorted grasps by quality. Best quality: {sorted_quality_scores[0]:.4f}, Worst: {sorted_quality_scores[-1]:.4f}")

        # save transforms
        grasps = {
            'object': obj.filename,
            'object_scale': obj.scale,
            'object_position': obj.position,
            'object_rotation': obj.rotation,
            'object_class': args.classname,
            'object_dataset': args.dataset,
            'gripper': args.gripper,
            'gripper_configuration': [gripper.q],
            'transforms': [t.tolist() for t in transforms],
            'roll_angles': roll_angles.tolist(),
            'standoffs': standoffs.tolist(),
            'mesh_points': [p.tolist() for p in points],
            'mesh_normals': [n.tolist() for n in normals],
            'collisions': collisions,
            'grasp_widths': grasp_widths,
            quality_key: sorted_quality_scores,
        }

        with open(args.output, 'w') as f:
            verboseprint("Writing results to:", args.output)
            json.dump(grasps, f)
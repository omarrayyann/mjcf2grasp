import numpy as np
import scipy.stats as stats
import trimesh
import trimesh.transformations as tra


def transforms_to_vec3d_quatd(transforms):
    """Convert transforms to (position, quaternion) pairs."""
    locations, quaternions = [], []
    for t in transforms:
        pos = t[:3, 3]
        quat = tra.quaternion_from_matrix(t)  # [w, x, y, z]
        locations.append(pos.tolist())
        quaternions.append(quat.tolist())
    return locations, quaternions


def sample_antipodal(object_mesh: trimesh.Trimesh) -> list[np.ndarray]:
    """Sample antipodal grasp transforms on the mesh."""
    # Hardcoded parameters
    num_candidates = 1000
    num_orientations = 1
    gripper_maximum_aperture = 0.1
    gripper_standoff_fingertips = 0.1
    gripper_approach_direction = np.array([0, 0, 0.2])
    grasp_align_axis = np.array([0, 1, 0])
    orientation_sample_axis = np.array([0, 1, 0])
    random_seed = 42

    np.random.seed(random_seed)
    # Sample surface
    n_surf = max(1, num_candidates // num_orientations)
    pts, fidx = object_mesh.sample(n_surf, return_index=True)
    normals = object_mesh.face_normals[fidx]
    # Ray cast
    dirs = -normals
    hits, idxs, _ = object_mesh.ray.intersects_location(pts, dirs, multiple_hits=True)

    centers, axes = [], []
    initial_grasp_widths = []
    for i in range(n_surf):
        h = hits[np.where(idxs == i)]
        if len(h) == 0:
            continue
        if len(h) > 1:
            d = np.linalg.norm(h - pts[i], axis=1)
            v = np.where(d <= gripper_maximum_aperture)[0]
            if not len(v):
                continue
            far = v[np.argmax(d[v])]
            opp = h[far]
        else:
            opp = h[0]
            if np.linalg.norm(opp - pts[i]) > gripper_maximum_aperture:
                continue
        vec = opp - pts[i]
        L = np.linalg.norm(vec)
        if trimesh.util.isclose(L, 0) or L > gripper_maximum_aperture:
            continue
        if L<0.013:
            continue
        axes.append(vec / L)
        centers.append(pts[i] + vec * 0.5)
        initial_grasp_widths.append(L)

    # Build transforms
    transforms = []
    grasp_widths = []
    angles = np.linspace(-np.pi, np.pi, num_orientations, False)
    standoff = gripper_approach_direction * -gripper_standoff_fingertips
    i = 0
    for c, ax in zip(centers, axes):
        try:
            align = trimesh.geometry.align_vectors(grasp_align_axis, ax)
        except ValueError:
            continue
        base = tra.translation_matrix(c)
        off = tra.translation_matrix(standoff)
        for a in angles:
            R = tra.rotation_matrix(a, orientation_sample_axis)
            transforms.append(base.dot(align).dot(R).dot(off))
            grasp_widths.append(initial_grasp_widths[i])
        i += 1
    return transforms, grasp_widths

if __name__ == "__main__":
    mesh = trimesh.load("models/pan.obj", force='mesh')
    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError("Loaded file is not a mesh")

    grasp_tfs, widths = sample_antipodal(mesh)    # Print
    locs, qs = transforms_to_vec3d_quatd(grasp_tfs)
    for i, (l, q) in enumerate(zip(locs, qs)):
        print(f"Grasp {i}: Pos={l}, Quat={q}")

    # Visualize
    scene = trimesh.Scene(mesh)
    L = 0.05  # C depth
    aperture = 0.08
    dash_length = 0.05  # line before C

    for tf, aperture in zip(grasp_tfs, widths):
        # C opening using dynamic aperture
        pts = np.array([
            [0, aperture/2, 0],
            [0, aperture/2, -L],
            [0, -aperture/2, -L],
            [0, -aperture/2, 0]
        ])
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        cp = trimesh.load_path(segs)
        cp.apply_transform(tf)
        scene.add_geometry(cp)

        # Approach dash
        ln = np.array([[0, 0, -0.05], [0, 0, -0.07]])
        seg = np.expand_dims(ln, 0)
        lp = trimesh.load_path(seg)
        lp.apply_transform(tf)
        scene.add_geometry(lp)

    scene.show()

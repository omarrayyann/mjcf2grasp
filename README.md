# mjcf2grasp
`mjcf2grasp` lets you generate and verify functional grasps from an [MJCF](https://mujoco.readthedocs.io/en/stable/XMLreference.html) file using [MuJoCo](https://mujoco.org/).
<p>
  <img src="https://github.com/user-attachments/assets/528f1dda-dc60-4434-b04b-c0b67b336bc4" alt="tennis" width="18%" height="auto" />
  <img src="https://github.com/user-attachments/assets/df1ec21f-c457-4ad6-a611-2f16d0ac4f61" alt="cup" width="18%" height="auto" />
  <img src="https://github.com/user-attachments/assets/03f89c0c-da8c-4a72-93bb-f9f24ba9874a" alt="pan_shake" width="18%" height="auto" />
  <img src="https://github.com/user-attachments/assets/0b7d68ed-6f1f-4855-a0e4-b17e3e1c2fb0" alt="pan" width="18%" height="auto" />
  <img src="https://github.com/user-attachments/assets/936c10b1-f8f6-4701-b608-62c3529fdb85" alt="pan" width="18%" height="auto" />
</p>


## Installation

```bash
conda env create -f conda_env.yaml
conda activate mjcf2grasp
git submodule update --init --recursive
```

Build the Manifold library (required for mesh processing):

```bash
cd external_src/Manifold
mkdir build && cd build
cmake ..
make
```

## Usage

### Rigid Objects

#### 1. Prepare Object List

Create a JSON file containing a list of MJCF objects to process. Each object needs a `name` and `xml` path:

```json
[
    {
        "name": "Tennis_Racquet_5",
        "xml": "/path/to/Tennis_Racquet_5/Tennis_Racquet_5.xml"
    },
    {
        "name": "Pen_1",
        "xml": "/path/to/Pen_1/Pen_1.xml"
    },
  ]
```

See `examples/rigid_objects_list.json` for a working example.

#### 2. Run the Pipeline

```bash
python run_rigid.py --objects_list examples/rigid_objects_list.json
```

#### 3. Results

Results are saved to `results/rigid_objects/<object_name>/`:

```
results/rigid_objects/
└── Tennis_Racquet_5/
    └── Tennis_Racquet_5_grasps_filtered.npz
```

The `.npz` file contains the validated grasp transforms.

#### 4. Visualize Results

```bash
python scripts/visualize_meshcat.py --objects_list examples/rigid_objects_list.json
```

Use arrow keys (or `n`/`p`) to navigate between objects, `q` to quit.

### Articulable Objects

#### 1. Prepare Object List

Create a JSON file containing a list of articulable MJCF objects to process. Each object needs a `name` and `xml` path:

```json
[
    {
        "name": "Doorway_1",
        "xml": "/path/to/Doorway_1/Doorway_1.xml"
    }
]
```

See `examples/articulable_objects_list.json` for a working example.

#### 2. Run the Pipeline

```bash
python run_articulable.py --objects_list examples/articulable_objects_list.json
```

#### 3. Results

Results are saved to `results/articulable_objects/<object_name>/`:

```
results/articulable_objects/
└── Doorway_1/
    ├── main.obj
    ├── joint_meshes_info.json
    ├── joint_meshes_info_filtered.json
    └── Doorway_1_doorway_handle_1_joint_0_grasps_filtered.npz
```

The filtered JSON and NPZ files contain per-joint validated grasp transforms.

#### 4. Visualize Results

```bash
python scripts/visualize_meshcat.py \
    --objects_list examples/articulable_objects_list.json \
    --results_dir results/articulable_objects \
    --articulable \
    --max_grasps_per_joint 10
```

Use arrow keys (or `n`/`p`) to navigate between objects, `q` to quit. Different joints are shown in different colors.
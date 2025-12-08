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

### Build Manifold

The Manifold library is required for mesh processing. Build it with:

```bash
cd external_src/Manifold
mkdir build && cd build
cmake ..
make
```

## Usage

### Static Objects

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
    }
]
```

See `examples/example_objects_list.json` for a working example.

#### 2. Run the Pipeline

```bash
python run_static.py --objects_list path/to/objects_list.json
```

The pipeline will:
1. Combine meshes from the MJCF file
2. Process with Manifold for watertight mesh
3. Generate grasp candidates
4. Filter and validate grasps using MuJoCo simulation

#### 3. Results

Results are saved to `results/static_objects/<object_name>/`:

```
results/static_objects/
└── Tennis_Racquet_5/
    └── Tennis_Racquet_5_grasps_filtered.npz
```

The `.npz` file contains the validated grasp transforms.
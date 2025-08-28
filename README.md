# Thor Grasp
A pipeline that generates grasp samples for Thor objects, with evaluation and filtering in MuJoCo

<p>
  <img src="https://github.com/user-attachments/assets/528f1dda-dc60-4434-b04b-c0b67b336bc4" alt="tennis" width="18%" height="auto" />
  <img src="https://github.com/user-attachments/assets/df1ec21f-c457-4ad6-a611-2f16d0ac4f61" alt="cup" width="18%" height="auto" />
  <img src="https://github.com/user-attachments/assets/03f89c0c-da8c-4a72-93bb-f9f24ba9874a" alt="pan_shake" width="18%" height="auto" />
  <img src="https://github.com/user-attachments/assets/0b7d68ed-6f1f-4855-a0e4-b17e3e1c2fb0" alt="pan" width="18%" height="auto" />
  <img src="https://github.com/user-attachments/assets/936c10b1-f8f6-4701-b608-62c3529fdb85" alt="pan" width="18%" height="auto" />
</p>




## Installation

```bash
pip install -r requirements.txt
```

**External dependencies**:
   - Install [Manifold library](https://github.com/hjwdzh/Manifold) under external_src
   - Install [PyMeshLab](https://github.com/cnr-isti-vclab/PyMeshLab) under external_src
## Steps

1. **Find Objects**: Run `scripts/find_objects.py` to scan for valid objects and generate `matched_objs.json`
   
   **For static objects:**
   ```bash
   python scripts/find_objects.py <directory> --output matched_objs.json
   ```
   
   **For articulated objects (furniture with movable joints):**
   ```bash
   python scripts/find_objects.py <directory> --output articulated_matched_objs.json --check-joints
   ```
   
   The `--check-joints` flag adds a filtering requirement for XML files to include articulated joints (non-free joints) 
   
   Example output format:
   ```json
   [
       {
           "name": "Tennis_Racquet_1",
           "xml": "/path/to/assets/Tennis_Racquet/Prefabs/Tennis_Racquet_1/Tennis_Racquet_1.xml"
       },
       {
           "name": "Microwave_12", 
           "xml": "/path/to/assets/Microwave/Prefabs/Microwave_12/Microwave_12.xml"
       }
   ]
   ```

2. **Run Pipeline**: Run the appropriate pipeline based on object type:
   
   **For static objects:**
   ```bash
   python run_pipeline.py
   ```
   
   **For articulated objects (for fuctional grasps):**
   ```bash
   python run_pipeline_articulate.py
   ```

**Pipeline stages:**
- `pipeline/0_generate_mesh.py` - Combines meshes from MuJoCo XML into a single OBJ file
- `pipeline/1_generate_grasps.py` - Samples grasp poses for the object mesh  
- `pipeline/2_mesh_colliders.py` - Converts primitive colliders to mesh colliders
- `pipeline/3_filter_mujoco.py` - Filters grasps using MuJoCo simulation

(`pipeline_articulate` is similar to  `pipeline` but has additional filtering requirement in the filter_mujoco process. It ensures the grasp allows a later movement to generate motion in one of the non-free joints of the asset.)

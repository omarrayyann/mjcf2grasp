# Thor Grasp
Pipeline for generating grasps samples for Thor objects then evaluating and filtering them.

<p>
  <img src="https://github.com/user-attachments/assets/528f1dda-dc60-4434-b04b-c0b67b336bc4" alt="tennis" width="15%" height="auto" />
  <img src="https://github.com/user-attachments/assets/df1ec21f-c457-4ad6-a611-2f16d0ac4f61" alt="cup" width="15%" height="auto" />
  <img src="https://github.com/user-attachments/assets/03f89c0c-da8c-4a72-93bb-f9f24ba9874a" alt="pan_shake" width="15%" height="auto" />
  <img src="https://github.com/user-attachments/assets/0b7d68ed-6f1f-4855-a0e4-b17e3e1c2fb0" alt="pan" width="15%" height="auto" />
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
   ```bash
   python scripts/find_objects.py <directory> --output matched_objs.json
   ```
   
   Example output format:
   ```json
   [
       {
           "name": "Keychain_3",
           "xml": "/path/to/assets/KeyChain/Prefabs/Keychain_3/Keychain_3.xml"
       },
       {
           "name": "Laptop_12", 
           "xml": "/path/to/assets/Laptop_Full/Prefabs/Laptop_12/Laptop_12.xml"
       }
   ]
   ```

2. **Run Pipeline**: Run `run_pipeline.py` to process each object through the pipeline stages
   ```bash
   python run_pipeline.py
   ```
- `pipeline/0_generate_mesh.py` - Combines meshes from MuJoCo XML into a single OBJ file
- `pipeline/1_generate_grasps.py` - Samples grasp poses for the object mesh  
- `pipeline/2_filter_mujoco.py` - Filters grasps using MuJoCo

# Thor Grasp

Pipeline for generating grasps samples for Thor objects then evaluating and filtering them.

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

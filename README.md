# Thor Grasp

A pipeline for generating grasps samples for objects then evaluating and filtering them.

## Steps

1. **Find Objects**: Run `scripts/find_objects.py` to scan for valid objects and generate `matched_objs.json`
   ```bash
   python scripts/find_objects.py <directory> --output matched_objs.json
   ```

2. **Run Pipeline**: Run `run_pipeline.py` to process each object through the pipeline stages
   ```bash
   python run_pipeline.py
   ```

## Pipeline Scripts

- `0_generate_mesh.py` - Combines meshes from MuJoCo XML into a single OBJ file
- `1_generate_grasps.py` - Samples grasp poses for the object mesh  
- `2_filter_mujoco.py` - Filters grasps using MuJoCo

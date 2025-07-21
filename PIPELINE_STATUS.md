# Articulated Object Grasping Pipeline - Current Status

## Pipeline Structure (Reordered)

The pipeline now runs in the following logical order:

### Stage 0: Handle Detection and Mesh Generation
- **Script**: `pipeline_articulate/0_generate_mesh.py`
- **Purpose**: Uses LLM to identify handle components in MuJoCo XML files
- **Output**: Handle-only mesh (`.obj`) and full object mesh
- **Status**: ✅ Working

### Stage 1: Joint Axis Analysis  
- **Script**: `pipeline_articulate/1_find_axis_joint.py`
- **Purpose**: Uses LLM to identify the primary articulation joint (e.g., door hinge)
- **Output**: Joint axis information (`.json`) with joint type, axis, range, etc.
- **Status**: ⚠️ Implemented but needs OpenAI API key

### Stage 2: Grasp Generation
- **Script**: `pipeline_articulate/2_generate_grasps.py`
- **Purpose**: Generates grasps specifically for handle meshes
- **Output**: Grasp poses with quality metrics (`.json`)
- **Status**: ✅ Working

### Stage 3: Grasp Filtering
- **Script**: `pipeline_articulate/3_filter_mujoco.py`
- **Purpose**: Filters grasps using MuJoCo simulation for feasibility
- **Output**: Filtered grasps that pass simulation tests (`.json`)
- **Status**: ✅ Working

## Key Features

1. **LLM Integration**: Both handle detection and joint analysis use OpenAI GPT-4 for intelligent analysis
2. **Visualization Support**: Grasp visualization with `--articulated` flag for articulated pipeline outputs
3. **Resumable Pipeline**: Skips stages that have already been completed
4. **Comprehensive Reporting**: Detailed progress and statistics for each stage

## Output Structure

```
output_articulate/
└── ObjectName/
    ├── ObjectName_handles.obj           # Handle-only mesh
    ├── ObjectName_handles_handle_info.json  # Handle detection results
    ├── ObjectName_full.obj              # Full object mesh
    ├── ObjectName_joint_axis.json       # Joint analysis results
    ├── ObjectName_grasps.json           # Generated grasps
    └── ObjectName_grasps_filtered.json  # Filtered grasps
```

## Next Steps

1. **Set OpenAI API Key** for joint axis analysis
2. **Implement Stage 4**: Articulation motion planning
3. **Implement Stage 5**: MuJoCo validation with joint constraints

## Usage

```bash
# Run full pipeline
python run_pipeline_articulate.py

# Visualize grasps for articulated objects
python scripts/visualize_render.py ObjectName --articulated
python scripts/visualize_render.py ObjectName --filtered --articulated
```

The pipeline now processes articulated objects in a logical sequence that builds understanding progressively: handles → joints → grasps → filtering.

## Building the Docker Image

> **Note:** This image is a **base environment only** - it contains dependencies and cuRobo but NO code. Mount your code at `/root/mujoco-thor` at runtime. This keeps the image small, reusable, and means code changes don't require rebuilding.

### Prerequisites

1. Install Beaker: https://beaker-docs.apps.allenai.org/start/install.html
2. Configure Beaker: https://beaker.org/user
3. Set GitHub token (for private cuRobo repo):
   ```bash
   export GITHUB_TOKEN=your_token_here
   ```

### Build Command

```bash
export GITHUB_TOKEN=your_token_here

BASE_BEAKER_IMAGE="mjthor-datagen-base-$(date +"%Y%m%d-%Hh%Mm%Ss")" \
  && docker build \
   -t $BASE_BEAKER_IMAGE:latest \
   --build-arg GITHUB_TOKEN=$GITHUB_TOKEN \
   --file beaker_scripts/beaker_base.Dockerfile \
   . \
  && beaker image create --name $BASE_BEAKER_IMAGE $BASE_BEAKER_IMAGE:latest
```

**To update cuRobo version:** Modify `CUROBO_COMMIT` in the Dockerfile or add `--build-arg CUROBO_COMMIT=<hash>`

## Running on Beaker

**Important:** The image does NOT include the code - you must mount it at runtime.

As of 08Oct2025, the latest base image is `roseh/mjthor-datagen-base-20251008-22h38m08s`.

### Step 1: Package Your Code

First, package your code as a Beaker dataset:

```bash
python beaker_scripts/package_code.py
```

This will output a dataset name like: `roseh/roseh_code_2025-05-08_23-07-37`

### Step 2: Create a Beaker Session

Set your code dataset name and create a session:

```bash
# Set the codebase (from step 1)
BEAKER_CODEBASE=roseh/roseh_code_2025-05-08_23-07-37

# Get your username in uppercase for secret access
BEAKER_USERNAME_CAPS=$(beaker account whoami | tail -1 | awk '{print $2}' | tr [:lower:] [:upper:])

# Create session with code mounted
beaker session create \
    -w ai2/robo-molmo \
    --bare \
    --gpus 8 \
    --image beaker://roseh/mjthor-datagen-base-20251008-22h38m08s \
    --mount beaker://$BEAKER_CODEBASE=/root/mujoco-thor \
    --mount weka://prior-default=/weka/prior \
    --secret-env BEAKER_TOKEN="${BEAKER_USERNAME_CAPS}_BEAKER_TOKEN" \
    --secret-env AWS_ACCESS_KEY_ID="${BEAKER_USERNAME_CAPS}_AWS_ACCESS_KEY_ID" \
    --secret-env AWS_SECRET_ACCESS_KEY="${BEAKER_USERNAME_CAPS}_AWS_SECRET_ACCESS_KEY" \
    --secret-env GITHUB_TOKEN="${BEAKER_USERNAME_CAPS}_GITHUB_TOKEN" \
    --secret-env WANDB_API_KEY="${BEAKER_USERNAME_CAPS}_WANDB_API_KEY" \
    --env AWS_DEFAULT_REGION=us-west-2 \
    --env PYTHONPATH=/root/mujoco-thor/ \
    --env BEAKER_CODEBASE=$BEAKER_CODEBASE \
    --env MJCTHOR_ASSETS_DIR=/weka/prior/datasets/robomolmo/mjthor_resources/ \
    --budget ai2/oe-mm
```

### Step 3: Run Your Experiments

Inside the container:
```bash
conda activate mjthor
cd /root/mujoco-thor
python -m mujoco_thor.data_generation.main DoorOpeningDataGenConfig
```

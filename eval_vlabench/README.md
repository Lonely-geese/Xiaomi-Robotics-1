# VLABench Evaluation

This guide walks through evaluating **Xiaomi-Robotics-1** on the [VLABench](https://github.com/OpenMOSS/VLABench) benchmark: setting up the simulation environment, launching the inference server, running the evaluation client, and reading the results.

> **Architecture.** Evaluation uses a client–server split: the **server** loads the model and serves actions over a socket; the **client** runs the VLABench simulator, builds inputs with the Hugging Face `AutoProcessor`, and decodes returned actions. They run in **two separate conda environments**.

---

## Prerequisites

### 1. Deploy environment (`mibot`)

The server runs here. Install it once by following **[docs/DEPLOYMENT.md](../docs/DEPLOYMENT.md)**.

### 2. VLABench simulation environment (`vlabench`)

```bash
conda create -n vlabench python=3.10 -y
conda activate vlabench

# VLABench — installed under the current directory (repository root)
export VLABENCH_ROOT="$PWD/VLABench"
git clone https://github.com/OpenMOSS/VLABench.git "$VLABENCH_ROOT"
cd "$VLABENCH_ROOT"
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python scripts/download_assets.py

# XR-1 evaluation client dependencies.
# transformers must be exactly 4.57.1; other versions are unverified.
python -m pip install \
  torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install transformers==4.57.1 tyro tqdm pillow openai

# Pinned MuJoCo / dm_control
python -m pip install --force-reinstall --no-deps \
  mujoco==3.2.2 mujoco-mjx==3.2.2 dm_control==1.0.22

# RRT dependency (pinned commit)
mkdir -p "$VLABENCH_ROOT/third_party"
cd "$VLABENCH_ROOT/third_party"
git clone https://github.com/motion-planning/rrt-algorithms.git
cd rrt-algorithms
git checkout e51d95ee489a225220d6ae2a764c4111f6ba7d85
python -m pip install -e . --no-deps
cd - >/dev/null
```

Set the remaining runtime variables before starting evaluation (headless EGL, no X11). `VLABENCH_ROOT` was already set above:

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=0
export TOKENIZERS_PARALLELISM=false
```

The evaluation reads the official track files from `$VLABENCH_ROOT/configs/evaluation/tracks/`. Do not import `dm_control.viewer` on a headless server — VLABench uses EGL rendering and does not require an X11 display.

### 3. Checkpoint

Download the fine-tuned checkpoint from the [VLABench collection](https://huggingface.co/XiaomiRobotics/Xiaomi-Robotics-1-VLABench). The same model directory must be readable by both the server and the client:

```bash
export MODEL_PATH="$PWD/checkpoints/Xiaomi-Robotics-1-VLABench"
hf download XiaomiRobotics/Xiaomi-Robotics-1-VLABench --local-dir "$MODEL_PATH"
```

The processor must support the `vlabench_choice` robot key. The released checkpoint uses a raw action shape of `[10, 60]` with the first 7 dimensions executable: dims `0:3` position delta, `3:6` Euler rotation delta, `6` gripper; action chunk size 10, replanning every 5 steps.

---

## Evaluation

### Step 1 — Start the inference server

In a terminal with the **deploy** environment, from the repository root:

```bash
conda activate mibot
bash scripts/deploy.sh "$MODEL_PATH" 8 8   # <model_path> <num_ports> <num_gpus>
```

This starts 8 servers on ports `10086`–`10093` inside the `model_servers` tmux session. Inspect with `tmux attach -t model_servers`. **Start the server before launching the client.**

### Step 2 — Run the evaluation client

In another terminal. Activate the **VLABench** environment first (the launcher does not switch environments for you):

```bash
conda activate vlabench
bash scripts/launch_vlabench.sh 8 ./eval_vlabench/eval_logs "$MODEL_PATH"
```

| Arg | Meaning |
| :--- | :--- |
| `8` | Number of client workers (should match the number of servers) |
| `./eval_vlabench/eval_logs` | Output directory for results |
| `"$MODEL_PATH"` | Checkpoint path (used by the client to load the processor) |

The launcher evaluates five tracks sequentially — `track_1_in_distribution`, `track_2_cross_category`, `track_3_common_sense`, `track_4_semantic_instruction`, `track_6_unseen_texture` — distributing episodes within each track across the workers, then merges results at the end.

**Smoke test** (one worker, one episode per task):

```bash
conda activate vlabench
NUM_EVAL_EPISODES=1 \
    bash scripts/launch_vlabench.sh \
    1 ./eval_vlabench/eval_logs_smoke "$MODEL_PATH"
```

**Configuration knobs.** Override launcher behavior with environment variables: `BASE_PORT`, `NUM_EVAL_EPISODES`, `ROBOT_TYPE`, `ACTION_CHUNK_SIZE`, `REPLAN_STEPS`, `VISUALIZATION` (1/0), and `COT` (1/0).

**Run a single track or selected tasks** directly via `eval_vlabench/main.py`:

```bash
python -u eval_vlabench/main.py \
  --model-path "$MODEL_PATH" \
  --eval-track track_1_in_distribution \
  --n-episode 50 \
  --host 127.0.0.1 --port 10086 \
  --robot-type vlabench_choice \
  --state-dim 60 --action-dim 7 \
  --action-chunk-size 10 --replan-steps 5 \
  --visualization --no-cot \
  --save-dir ./eval_vlabench/eval_logs/track_1_in_distribution
```

Use `--tasks "task_a task_b"` to evaluate a subset of tasks.

---

## Results

When using `scripts/launch_vlabench.sh`, the output layout is:

```text
<output_directory>/
  track_1_in_distribution/
    dispatch.log
    metrics.json
    dispatch_manifest.json
    <task_name>/
      detail_info.json
      videos/
      worker_logs/
  track_2_cross_category/
  track_3_common_sense/
  track_4_semantic_instruction/
  track_6_unseen_texture/
  merged_results.json
  results.md
```

Reported metrics are **success rate (SR)**, **intention score (IS)**, and **progress score (PS)**. After all tracks finish, the launcher writes `merged_results.json` and `results.md`. Track values are macro averages across tasks; overall values are macro averages across all task entries (not episode-weighted).

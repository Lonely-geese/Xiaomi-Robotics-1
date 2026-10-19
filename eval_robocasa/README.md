# RoboCasa Evaluation

This guide walks through evaluating **Xiaomi-Robotics-1** on the [RoboCasa](https://robocasa.ai/) benchmark: setting up the simulation environment, launching the inference server, running the evaluation client, and reading the results.

> **Architecture.** Evaluation uses a client–server split: the **server** loads the model and serves actions over a socket; the **client** runs the RoboCasa simulator, builds inputs with the Hugging Face `AutoProcessor`, and decodes returned actions. They run in **two separate conda environments**.

---

## Prerequisites

### 1. Deploy environment (`mibot`)

The server runs here. Install it once by following **[docs/DEPLOYMENT.md](../docs/DEPLOYMENT.md)**.

### 2. RoboCasa simulation environment (`robocasa`)

We use RoboCasa **v0.2** (not v1.0, which is for RoboCasa365) with Python 3.10 and a pinned PyTorch version:

```bash
conda create -n robocasa python=3.10 -y
conda activate robocasa

# robosuite (master) + RoboCasa (v0.2)
git clone https://github.com/ARISE-Initiative/robosuite
cd robosuite && pip install -e . && cd ..
git clone https://github.com/robocasa/robocasa --branch v0.2
cd robocasa && pip install -e . && cd ..

# If numpy/numba install fails, try conda instead:
# conda install -c numba numba=0.56.4 -y

# XR-1 evaluation client dependencies (torch/numpy/mujoco come with robocasa).
# transformers must be exactly 4.57.1; other versions are unverified.
pip install transformers==4.57.1 imageio[ffmpeg] scipy tyro tqdm

# Set up RoboCasa system variables and download kitchen assets (~10 GB)
python -m robocasa.scripts.setup_macros
python -m robocasa.scripts.download_kitchen_assets
```

### 3. Checkpoint

Download the fine-tuned checkpoint from the [RoboCasa collection](https://huggingface.co/XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa). The same model directory must be readable by both the server and the client:

```bash
export MODEL_PATH="$PWD/checkpoints/Xiaomi-Robotics-1-RoboCasa"
hf download XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa --local-dir "$MODEL_PATH"
```

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

In another terminal. Activate the **RoboCasa** environment first (the launcher does not switch environments for you):

```bash
conda activate robocasa
bash scripts/launch_robocasa.sh 8 ./eval_robocasa/eval_logs "$MODEL_PATH"
```

| Arg | Meaning |
| :--- | :--- |
| `8` | Number of client workers (must match the number of servers) |
| `./eval_robocasa/eval_logs` | Output directory for logs and results |
| `"$MODEL_PATH"` | Checkpoint path (used by the client to load the processor) |

Each worker independently records success rates and rollout videos, and the launcher merges the per-task results at the end.

---

## Results

Our reference results (24 tasks, 100 episodes each):

| Task | Success Rate | Task | Success Rate |
| :--- | :---: | :--- | :---: |
| PnPCounterToCab | 56% | OpenDrawer | 77% |
| PnPCabToCounter | 64% | CloseDrawer | 96% |
| PnPCounterToSink | 59% | TurnOnStove | 82% |
| PnPSinkToCounter | 71% | TurnOffStove | 29% |
| PnPCounterToMicrowave | 30% | TurnOnSinkFaucet | 91% |
| PnPMicrowaveToCounter | 37% | TurnOffSinkFaucet | 97% |
| PnPCounterToStove | 73% | TurnSinkSpout | 86% |
| PnPStoveToCounter | 62% | CoffeeSetupMug | 42% |
| OpenSingleDoor | 87% | CoffeeServeMug | 77% |
| CloseSingleDoor | 100% | CoffeePressButton | 82% |
| OpenDoubleDoor | 99% | TurnOnMicrowave | 91% |
| CloseDoubleDoor | 94% | TurnOffMicrowave | 99% |
| **Average** | **74.2%** | | |

Results may vary across GPU machines. We ship our evaluation logs in `eval_logs/` — including detailed per-rank results and the final merged summary — to aid comparison and debugging.

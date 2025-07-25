# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Installation and Setup
- Install LeRobot: `pip install -e .`
- Install with simulation environments: `pip install -e ".[aloha, pusht, xarm]"`
- Install with specific robot support: `pip install -e ".[dynamixel, gamepad, intelrealsense]"`
- Install development dependencies: `pip install -e ".[dev, test]"`
- Setup WandB for logging: `wandb login`

### Training and Evaluation
- Train a policy: `python -m lerobot.scripts.train --policy.type=act --env.type=aloha --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human`
- Evaluate a pretrained policy: `python -m lerobot.scripts.eval --policy.path=lerobot/diffusion_pusht --env.type=pusht --eval.batch_size=10 --eval.n_episodes=10`
- Resume training from checkpoint: `python -m lerobot.scripts.train --config_path=path/to/train_config.json --resume=true`

### Data and Visualization
- Visualize dataset: `python -m lerobot.scripts.visualize_dataset --repo-id lerobot/pusht --episode-index 0`
- Load dataset in Python: `from lerobot.datasets.lerobot_dataset import LeRobotDataset; dataset = LeRobotDataset("lerobot/pusht")`

### Testing
- Run all tests: `pytest`
- Run end-to-end tests: `make test-end-to-end`
- Run specific policy tests: `make test-act-ete-train` (or diffusion, tdmpc, smolvla)

### Code Quality
- Format code: `ruff format`
- Lint code: `ruff check`
- Run type checking: `mypy` (when configured)

### Hardware Setup (for real robots)
- Find cameras: `lerobot-find-cameras`
- Find motor ports: `lerobot-find-port`
- Setup motors: `lerobot-setup-motors`
- Calibrate robot: `lerobot-calibrate`
- Record demonstrations: `lerobot-record`
- Replay demonstrations: `lerobot-replay`
- Teleoperate robot: `lerobot-teleoperate`

## Architecture Overview

LeRobot is a comprehensive robotics library with the following main components:

### Core Components
- **Policies** (`src/lerobot/policies/`): Implement various ML algorithms for robot control
  - ACT (Action Chunking with Transformers): Transformer-based imitation learning
  - Diffusion Policy: Diffusion models for action prediction
  - TDMPC: Temporal Difference Model Predictive Control
  - VQ-BeT: Vector Quantized Behavior Transformer
  - SmolVLA: Vision-Language-Action models for affordable robotics
  - Pi0: Large-scale policy pretraining models

- **Datasets** (`src/lerobot/datasets/`): Handle robotics datasets in LeRobotDataset format
  - Supports both simulation and real-world datasets
  - Video encoding/decoding for efficient storage
  - Temporal data relationships with delta_timestamps
  - HuggingFace Hub integration for dataset sharing

- **Robots** (`src/lerobot/robots/`): Physical robot implementations
  - SO-100/SO-101: Low-cost robotic arms
  - Koch: Teleoperation setup
  - HopeJR: Humanoid robot arm and hand
  - LeKiwi: Mobile robot platform
  - Stretch3, ViperX: Research platforms

- **Motors** (`src/lerobot/motors/`): Motor control interfaces
  - Dynamixel servos
  - Feetech servos
  - Generic motor bus abstraction

- **Cameras** (`src/lerobot/cameras/`): Camera interfaces
  - OpenCV cameras
  - Intel RealSense cameras
  - Configurable camera pipelines

- **Environments** (`src/lerobot/envs/`): Simulation environments
  - ALOHA: Bimanual manipulation tasks
  - PushT: Object pushing tasks
  - XArm: Single-arm manipulation

### Key Design Patterns

**Configuration System**: Uses dataclass-based configs with Draccus for command-line interface generation. Configuration files are in `src/lerobot/configs/`.

**Factory Pattern**: Most components use factory functions (e.g., `make_policy`, `make_env`, `make_dataset`) for instantiation with string identifiers.

**Dataset Format**: LeRobotDataset is the standard format using:
- HuggingFace datasets (Arrow/Parquet) for metadata
- MP4 videos for camera data (efficient storage)
- Temporal relationships via delta_timestamps
- Episode-aware data organization

**Device Management**: Automatic device detection and placement for CPU/CUDA operations throughout the codebase.

**Plugin System**: Extensible architecture for adding new policies, robots, and environments through configuration files.

## Common Workflows

### Adding a New Policy
1. Create policy directory in `src/lerobot/policies/`
2. Implement configuration dataclass (inheriting from base config)
3. Implement policy class with required methods (forward, select_action)
4. Update `available_policies` in `src/lerobot/__init__.py`
5. Add configuration YAML file
6. Add tests in `tests/policies/`

### Adding a New Robot
1. Create robot directory in `src/lerobot/robots/`
2. Implement robot class inheriting from base Robot class
3. Create configuration file
4. Update `available_robots` in `src/lerobot/__init__.py`
5. Add calibration and setup scripts if needed

### Working with Datasets
- Datasets follow LeRobotDataset format with standardized keys
- Use `delta_timestamps` for temporal relationships
- Camera data stored as VideoFrame references to MP4 files
- Support for both local and HuggingFace Hub storage

### Training Workflow
1. Prepare dataset (local or from hub)
2. Configure policy, environment, and training parameters
3. Run training script with appropriate arguments
4. Monitor with WandB (optional)
5. Evaluate checkpoints periodically
6. Upload best models to hub

## Important Files

- `src/lerobot/__init__.py`: Available components registry
- `src/lerobot/configs/`: All configuration dataclasses
- `src/lerobot/scripts/train.py`: Main training pipeline
- `src/lerobot/scripts/eval.py`: Policy evaluation pipeline
- `pyproject.toml`: Package configuration and dependencies
- `Makefile`: Test commands and end-to-end workflows

## Development Notes

- Python 3.10+ required
- Uses Ruff for formatting and linting
- Tests use pytest with timeout support
- Supports both CPU and CUDA devices
- Optional dependencies for different robot/simulation platforms
- WandB integration for experiment tracking
- Docker support for containerized development
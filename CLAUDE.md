# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

### Testing
- **Run all tests**: `python -m pytest -sv ./tests`
- **Run specific test file**: `pytest tests/<TEST_TO_RUN>.py`
- **Run end-to-end tests**: `make test-end-to-end`
- **Individual policy tests**: `make test-act-ete-train`, `make test-diffusion-ete-train`, etc.

### Code Quality
- **Format and lint**: `pre-commit run --all-files`
- **Install pre-commit hooks**: `pre-commit install`
- **Run ruff linting**: `ruff check .`
- **Run ruff formatting**: `ruff format .`

### Training and Evaluation
- **Train a policy**: `lerobot-train --config_path=<config_path>`
- **Evaluate a policy**: `lerobot-eval --policy.path=<model_path>`
- **Reproduce SOTA results**: `lerobot-train --config_path=lerobot/diffusion_pusht`

### Development Setup
- **Using poetry**: `poetry sync --extras "dev test"` or `poetry sync --all-extras`
- **Using uv**: `uv sync --extra dev --extra test` or `uv sync --all-extras`
- **Install as editable**: `pip install -e .`

## Code Architecture

### Core Structure
- **src/lerobot/**: Main source code directory
- **policies/**: Policy implementations (ACT, Diffusion, TDMPC, SmolVLA, etc.)
- **datasets/**: Dataset loading and processing utilities
- **robots/**: Robot hardware interfaces and configurations
- **teleoperators/**: Teleoperation interfaces (keyboard, gamepad, haptic devices)
- **cameras/**: Camera interfaces (OpenCV, RealSense)
- **motors/**: Motor control (Dynamixel, Feetech)
- **envs/**: Environment wrappers and utilities
- **scripts/**: Training and evaluation scripts

### Key Components

#### Dataset System
- **LeRobotDataset**: Core dataset class supporting temporal indexing with `delta_timestamps`
- Videos stored as MP4 with metadata in parquet format
- Episodes indexed with start/end frame indices
- Statistics computed for normalization

#### Policy Architecture
Each policy implements:
- Configuration class (inherits from policy config base)
- Model class with forward pass
- Training and evaluation logic
- Supports different action chunking and prediction horizons

#### Robot Integration
- Abstract Robot class with standardized interface
- Teleoperation through leader-follower setups
- Motor calibration and configuration utilities
- Camera integration for visual observations

### Development Patterns

#### Adding New Policies
1. Create policy directory under `src/lerobot/policies/`
2. Implement configuration and modeling classes
3. Update `available_policies` in `lerobot/__init__.py`
4. Add tests in `tests/policies/`

#### Adding New Environments
1. Update `available_tasks_per_env` and `available_datasets_per_env` in `lerobot/__init__.py`
2. Follow gymnasium interface standards
3. Provide environment configuration

#### Adding New Datasets
1. Update `available_datasets_per_env` in `lerobot/__init__.py`
2. Follow LeRobotDataset format standards
3. Include episode indexing and statistics

## Testing Notes
- Tests require git-lfs for artifacts: `git lfs pull`
- Simulation environments are tested in CI even if not installed locally
- End-to-end tests cover full training/evaluation pipelines
- Mock objects available for hardware testing without physical devices

## Hardware Integration
- Motor buses support Dynamixel and Feetech servos
- Camera support includes OpenCV and Intel RealSense
- Robot configurations stored as YAML files
- Calibration utilities for joint limits and motor settings

## Logging and Monitoring
- WandB integration for experiment tracking (`--wandb.enable=true`)
- Rerun.io for dataset visualization
- Comprehensive logging utilities in `src/lerobot/utils/`

## Dependencies and Environment
- Python 3.10+ required
- PyTorch 2.2+ with torchvision
- Optional dependencies defined in pyproject.toml for specific features
- Development uses poetry or uv for dependency management
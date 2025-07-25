# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This script demonstrates how to train Diffusion Policy on the ARX dual-arm carpet folding dataset.

The dataset contains demonstrations of bimanual carpet folding with multi-camera observations
(base, left_wrist, right_wrist) and 14-dimensional dual-arm actions (7 per arm).
"""

from pathlib import Path

import torch

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy


def main():
    # Create a directory to store the training checkpoint.
    output_directory = Path("outputs/train/arx_bimanual_diffusion_carpet_fold")
    output_directory.mkdir(parents=True, exist_ok=True)

    # Select your device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Training configuration
    training_steps = 100000  # Increase for full training
    log_freq = 100
    batch_size = 32
    learning_rate = 1e-4

    # Load dataset metadata for the ARX dual-arm carpet folding dataset
    dataset_repo_id = "kelvinzhaozg/arx_dual_arm_carpet_fold_combined"
    print(f"Loading dataset metadata from {dataset_repo_id}")

    dataset_metadata = LeRobotDatasetMetadata(dataset_repo_id)
    print(f"Dataset metadata loaded: {dataset_metadata}, features={dataset_metadata.features}")
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    print(f"Input features: {list(input_features.keys())}")
    print(f"Output features: {list(output_features.keys())}")

    # Configure Diffusion Policy for dual-arm setup with multi-camera observations
    cfg = DiffusionConfig(
        input_features=input_features,
        output_features=output_features,
        # Multi-camera configuration
        vision_backbone="resnet18",
        crop_shape=(224, 224),  # Crop from 240x424 to 224x224
        use_separate_rgb_encoder_per_camera=True,  # Important for multi-camera setup
        # Diffusion model configuration
        n_obs_steps=2,  # Use 1 observation steps (matches default delta indices)
        horizon=32,  # 32-step action horizon
        n_action_steps=16,  # Execute 16 actions per policy call
        # U-Net architecture
        down_dims=(512, 1024, 2048),
        kernel_size=5,
        n_groups=8,
        diffusion_step_embed_dim=128,
        use_film_scale_modulation=True,
        # Noise scheduler
        noise_scheduler_type="DDIM",
        num_train_timesteps=100,
        beta_schedule="squaredcos_cap_v2",
        beta_start=0.0001,
        beta_end=0.02,
        prediction_type="epsilon",
        clip_sample=True,
        clip_sample_range=1.0,
        # Training parameters
        optimizer_lr=learning_rate,
        optimizer_betas=(0.95, 0.999),
        optimizer_eps=1e-8,
        optimizer_weight_decay=1e-6,
    )

    print(f"Policy configuration created with horizon={cfg.horizon}, n_action_steps={cfg.n_action_steps}")

    # Instantiate policy with config and dataset stats
    policy = DiffusionPolicy(cfg, dataset_stats=dataset_metadata.stats)
    policy.train()
    policy.to(device)

    print(f"Policy instantiated and moved to {device}")

    # Configure delta timestamps for multi-step observations and actions
    # Based on 50 FPS from the dataset conversion script
    fps = dataset_metadata.fps
    print(f"Dataset FPS: {fps}")

    delta_timestamps = {
        # Multi-camera observations: load previous and current frames
        "base_image": [i / fps for i in cfg.observation_delta_indices],
        "left_wrist_image": [i / fps for i in cfg.observation_delta_indices],
        "right_wrist_image": [i / fps for i in cfg.observation_delta_indices],
        # Robot state: load previous and current state
        "state": [i / fps for i in cfg.observation_delta_indices],
        # Actions: load action sequence for diffusion supervision
        "actions": [i / fps for i in cfg.action_delta_indices],
    }

    print("Delta timestamps configured:")
    for key, timestamps in delta_timestamps.items():
        print(f"  {key}: {timestamps}")

    # Instantiate dataset with delta timestamps configuration
    print("Loading dataset...")
    dataset = LeRobotDataset(dataset_repo_id, delta_timestamps=delta_timestamps, video_backend="pyav")
    print(f"Dataset loaded with {len(dataset)} samples")

    # Create optimizer and dataloader
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=4,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )

    print(f"Starting training for {training_steps} steps with batch_size={batch_size}")

    # Run training loop
    step = 0
    done = False
    total_loss = 0.0

    while not done:
        for batch in dataloader:
            # Move batch to device
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

            # Map dataset keys to policy expected format
            policy_batch = {}

            # Map state -> observation.state
            if "state" in batch:
                policy_batch["observation.state"] = batch["state"]

            # Map actions -> action
            if "actions" in batch:
                policy_batch["action"] = batch["actions"]

            # Create action_is_pad (all False since we have real actions)
            if "actions" in batch:
                policy_batch["action_is_pad"] = torch.zeros(
                    batch["actions"].shape[:-1], dtype=torch.bool, device=device
                )

            # Copy image observations (keep original key names)
            for key in batch:
                if "image" in key:
                    policy_batch[key] = batch[key]

            # Copy other necessary keys
            for key in ["timestamp", "frame_index", "episode_index", "index", "task_index"]:
                if key in batch:
                    policy_batch[key] = batch[key]

            # Forward pass
            loss, _ = policy.forward(policy_batch)

            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

            if step % log_freq == 0:
                avg_loss = total_loss / max(1, step + 1)
                print(f"step: {step:6d} | loss: {loss.item():.4f} | avg_loss: {avg_loss:.4f}")

                # Save checkpoint every 5000 steps
                if step > 0 and step % 5000 == 0:
                    checkpoint_dir = output_directory / f"checkpoint_{step}"
                    checkpoint_dir.mkdir(exist_ok=True)
                    policy.save_pretrained(checkpoint_dir)
                    print(f"Checkpoint saved at step {step}")

            step += 1
            if step >= training_steps:
                done = True
                break

    # Save final policy checkpoint
    print("Training completed. Saving final checkpoint...")
    policy.save_pretrained(output_directory)
    print(f"Final checkpoint saved to {output_directory}")

    # Print training summary
    final_avg_loss = total_loss / training_steps
    print("\nTraining Summary:")
    print(f"  Total steps: {training_steps}")
    print(f"  Final average loss: {final_avg_loss:.4f}")
    print(f"  Model saved to: {output_directory}")


if __name__ == "__main__":
    main()

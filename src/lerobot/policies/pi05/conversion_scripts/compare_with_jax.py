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

import json
import pickle
from pathlib import Path

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy


def display(tensor: torch.Tensor):
    if tensor.dtype == torch.bool:
        tensor = tensor.float()
    print(f"Shape: {tensor.shape}")
    print(f"Mean: {tensor.mean().item()}")
    print(f"Std: {tensor.std().item()}")
    print(f"Min: {tensor.min().item()}")
    print(f"Max: {tensor.max().item()}")


def main():
    num_motors = 8
    device = "cuda"
    # model_name = "pi0_aloha_towel"
    # model_name = "pi0_aloha_sim"

    # pi05_droid
    model_name = "pi05_droid"

    if model_name == "pi05_droid":
        dataset_repo_id = "lerobot/droid_100"
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    ckpt_torch_dir = Path.home() / f".cache/openpi/openpi-assets/checkpoints/{model_name}_lerobot"
    ckpt_jax_dir = Path.home() / f".cache/openpi/openpi-assets/checkpoints/{model_name}"
    save_dir = Path(f"../openpi/data/{model_name}/save")

    with open(save_dir / "example.pkl", "rb") as f:
        example = pickle.load(f)
    with open(save_dir / "outputs.pkl", "rb") as f:
        outputs = pickle.load(f)
    with open(save_dir / "noise.pkl", "rb") as f:
        noise = pickle.load(f)

    with open(ckpt_jax_dir / "assets/droid/norm_stats.json") as f:
        norm_stats = json.load(f)

    # Override stats
    dataset_meta = LeRobotDatasetMetadata(dataset_repo_id)
    dataset_meta.stats["observation.state"]["mean"] = torch.tensor(
        norm_stats["norm_stats"]["state"]["mean"][:num_motors], dtype=torch.float32
    )
    dataset_meta.stats["observation.state"]["std"] = torch.tensor(
        norm_stats["norm_stats"]["state"]["std"][:num_motors], dtype=torch.float32
    )

    # Create LeRobot batch from Jax
    batch = {}
    # for cam_key, uint_chw_array in example["images"].items():
    #     batch[f"observation.images.{cam_key}"] = torch.from_numpy(uint_chw_array) / 255.0
    # batch["observation.state"] = torch.from_numpy(example["state"])
    # batch["action"] = torch.from_numpy(outputs["actions"])
    # batch["task"] = example["prompt"]

    # ['observation/exterior_image_1_left', 'observation/wrist_image_left', 'observation/joint_position', 'observation/gripper_position', 'prompt']
    batch["observation.images.exterior_image_1_left"] = (
        torch.from_numpy(example["observation/exterior_image_1_left"]) / 255.0
    )  # .permute(2, 0, 1)
    batch["observation.images.wrist_image_left"] = (
        torch.from_numpy(example["observation/wrist_image_left"]) / 255.0
    )  # .permute(2, 0, 1)

    batch["observation.state"] = torch.cat(
        [
            torch.from_numpy(example["observation/joint_position"]),
            torch.from_numpy(example["observation/gripper_position"]),
        ],
        dim=-1,
    )
    # batch["action"] = torch.from_numpy(outputs["actions"])
    batch["task"] = example["prompt"]

    # if model_name == "pi0_aloha_towel":
    #     del batch["observation.images.cam_low"]
    # elif model_name == "pi0_aloha_sim":
    #     batch["observation.images.top"] = batch["observation.images.cam_high"]
    #     del batch["observation.images.cam_high"]

    # Batchify
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].unsqueeze(0)
        elif isinstance(batch[key], str):
            batch[key] = [batch[key]]
        else:
            raise ValueError(f"{key}, {batch[key]}")

    # To device
    for k in batch:
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(device=device, dtype=torch.float32)

    noise = torch.from_numpy(noise).to(device=device, dtype=torch.float32)

    # from lerobot import policies  # noqa

    cfg = PreTrainedConfig.from_pretrained(ckpt_torch_dir)
    cfg.pretrained_path = ckpt_torch_dir
    policy = make_policy(cfg, dataset_meta)

    pi_actions = torch.from_numpy(outputs["actions"]).to(device=device, dtype=torch.float32)

    actions = []
    for _ in range(pi_actions.shape[0]):
        action = policy.select_action(batch, noise=noise)
        actions.append(action)

    actions = torch.stack(actions, dim=1)
    actions = actions.squeeze(0)

    pi_actions = torch.from_numpy(outputs["actions"]).to(device=device, dtype=torch.float32)
    print("atol=3e-2", torch.allclose(actions, pi_actions[:, : actions.shape[1]], atol=3e-2))
    print("atol=2e-2", torch.allclose(actions, pi_actions, atol=2e-2))
    print("atol=1e-2", torch.allclose(actions, pi_actions, atol=1e-2))

    # # loss_dict = policy.forward(batch, noise=noise, time=time_beta)
    # # loss_dict["loss"].backward()
    # # print("losses")
    # # display(loss_dict["losses_after_forward"])
    # # print("pi_losses")
    # # display(pi_losses)

    # actions = []
    # for _ in range(50):
    #     action = policy.select_action(batch, noise=noise)
    #     actions.append(action)

    # actions = torch.stack(actions, dim=1)
    # pi_actions = batch["action"]
    # print("actions")
    # display(actions)
    # print()
    # print("pi_actions")
    # display(pi_actions)
    # print("atol=3e-2", torch.allclose(actions, pi_actions, atol=3e-2))
    # print("atol=2e-2", torch.allclose(actions, pi_actions, atol=2e-2))
    # print("atol=1e-2", torch.allclose(actions, pi_actions, atol=1e-2))


if __name__ == "__main__":
    main()

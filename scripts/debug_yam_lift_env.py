"""Debug script to initialize the YAM lift environment and capture rendered images.

Usage:
    ./isaaclab.sh -p scripts/debug_yam_lift_env.py --enable_cameras

    # Or headless with image saving only:
    ./isaaclab.sh -p scripts/debug_yam_lift_env.py --headless --enable_cameras
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Debug YAM lift environment rendering.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--output_dir", type=str, default="debug_output/yam_lift", help="Directory to save images.")
parser.add_argument("--save_every", type=int, default=10, help="Save an image every N steps.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os

import numpy as np
import torch

from isaaclab.envs import ManagerBasedRLEnv

from isaaclab_tasks.manager_based.manipulation.lift.config.yam.joint_pos_env_cfg import (  # noqa: F401
    YamCubeLiftEnvCfg,
)


def save_image(rgb_array: np.ndarray, path: str) -> None:
    """Save an RGB numpy array as a PNG image using PIL or fall back to raw save."""
    try:
        from PIL import Image

        img = Image.fromarray(rgb_array)
        img.save(path)
    except ImportError:
        # Fallback: save as raw .npy
        np.save(path.replace(".png", ".npy"), rgb_array)
        print(f"[WARN]: PIL not available, saved as .npy instead: {path}")


def main():
    """Initialize YAM lift env, step with random actions, and save rendered images."""
    # Create output directory
    os.makedirs(args_cli.output_dir, exist_ok=True)

    # Create environment config
    env_cfg = YamCubeLiftEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device

    # Set up viewer camera for a good viewpoint of the lift task
    env_cfg.viewer.eye = (1.5, 1.5, 1.5)
    env_cfg.viewer.lookat = (0.5, 0.0, 0.3)
    env_cfg.viewer.resolution = (1280, 720)

    # Create environment with rgb_array rendering
    env = ManagerBasedRLEnv(cfg=env_cfg, render_mode="rgb_array")

    # Reset environment
    print("[INFO]: Environment created. Resetting...")
    env.reset()

    # Main loop: step with random actions and save images
    step = 0
    print(f"[INFO]: Running loop. Saving images every {args_cli.save_every} steps to {args_cli.output_dir}/")

    while simulation_app.is_running():
        with torch.inference_mode():
            # Reset periodically
            if step > 0 and step % 300 == 0:
                env.reset()
                print(f"[INFO]: Reset at step {step}")

            # Random actions
            actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
            obs, rew, terminated, truncated, info = env.step(actions)

            # Render and save image
            if step % args_cli.save_every == 0:
                rgb = env.render()
                if rgb is not None and rgb.size > 0:
                    img_path = os.path.join(args_cli.output_dir, f"step_{step:06d}.png")
                    save_image(rgb, img_path)
                    print(f"[INFO]: Step {step:6d} | reward: {rew[0].item():.4f} | saved: {img_path}")
                else:
                    print(f"[INFO]: Step {step:6d} | reward: {rew[0].item():.4f} | render not ready")

            step += 1

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

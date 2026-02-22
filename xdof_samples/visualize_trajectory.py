"""
Trajectory Data Visualization Tool

Visualize robot trajectory data and camera feeds using Rerun.
"""

from pathlib import Path
from typing import List

import cv2
import numpy as np
import rerun as rr
import tyro
from rerun.datatypes import Mat3x3

from library.trajectory import (
    ArmSide,
    ArmTrajectory,
    CameraPerspective,
    FrameConvention,
    XmiTrajectory,
    load_trajectory,
)

# Constants
FPS = 30
CAM_OFFSET = [0, 0.09, -0.05]
CAM_ANGLE = np.pi / 4

TOP_CAMERA_NAME = "top_camera-images-rgb"
LEFT_CAMERA_NAME = "left_camera-images-rgb"
RIGHT_CAMERA_NAME = "right_camera-images-rgb"
MAX_WIDTH = 1280


def log_transform(name: str, matrix: np.ndarray):
    """Log 4x4 transformation matrix."""
    rr.log(
        name,
        rr.Transform3D(
            axis_length=0.1,
            translation=matrix[:3, 3],
            mat3x3=Mat3x3(matrix[:3, :3].flatten()),
        ),
    )


def log_video(name: str, video_path: Path, timestamps: List[int]):
    asset_video = rr.AssetVideo(path=str(video_path))
    # Log video to hierarchical path: name/camera/image
    topic = f"{name}/camera/image"

    rr.log(topic, asset_video, static=True)
    rr.send_columns(
        topic,
        indexes=[rr.TimeColumn("time", timestamp=[t * 1e-9 for t in timestamps])],
        columns=rr.VideoFrameReference.columns_nanos(timestamps),
    )


# Cache for video dimensions and scaled assets
_video_dimensions_cache = {}


def log_coordinate_with_video(pose, camera_relative_pose, video_path, entity_path, focal_length=20):
    """
    Log pose in world frame and camera with hierarchical structure.

    Args:
        pose: 4x4 transformation matrix for the coordinate frame
        camera_relative_pose: 4x4 camera pose relative to the coordinate frame
        video_path: Path to video file (for dimensions)
        entity_path: Rerun entity path (should match log_video name)
        focal_length: Camera focal length for visualization
    """
    width, height, scale = 640, 480, 1
    if video_path.exists():
        video_key = str(video_path)
        if video_key not in _video_dimensions_cache:
            cap = cv2.VideoCapture(str(video_path))
            if cap.isOpened():
                orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if orig_width > MAX_WIDTH or orig_height > MAX_WIDTH:
                    scale = min(MAX_WIDTH / orig_width, MAX_WIDTH / orig_height)
                else:
                    scale = 1
                cap.release()
                _video_dimensions_cache[video_key] = (orig_width, orig_height, scale)
            else:
                raise ValueError(f"Failed to open video at {video_path}")

        width, height, scale = _video_dimensions_cache[video_key]

    # Log main coordinate frame with smaller axis for better spacing
    rr.log(
        entity_path,
        rr.Transform3D(
            axis_length=0.15,
            translation=pose[:3, 3],
            mat3x3=Mat3x3(pose[:3, :3].flatten()),
        ),
    )

    # Log up arrow for visual reference
    rr.log(
        f"{entity_path}_arrow_up",
        rr.Arrows3D(
            origins=pose[:3, 3],
            vectors=np.array([0, 0, 0.02]),
            colors=[[0, 0, 255]],
        ),
    )

    # Log camera as child entity with relative transform
    camera_entity_path = f"{entity_path}/camera"
    rr.log(
        camera_entity_path,
        rr.Transform3D(
            axis_length=0.15,
            translation=camera_relative_pose[:3, 3],
            mat3x3=Mat3x3(camera_relative_pose[:3, :3].flatten()),
            scale=scale,
        ),
    )

    # Log camera pinhole model at camera entity (matches video location)
    rr.log(
        camera_entity_path,
        rr.Pinhole(focal_length=focal_length, width=width, height=height),
    )


def visualize_robot(
    trajectory: ArmTrajectory,
    frame: int,
    frame_convention: FrameConvention = FrameConvention.GRIPPER_RDF,
):
    """Visualize robot trajectory data."""
    rr.log("world_coordinate", rr.Transform3D(axis_length=0.1, translation=np.zeros(3)))

    left_ee_pose_obs = trajectory.get_ee_pose_obs(ArmSide.LEFT, frame_convention)[frame]
    log_transform("obs-left-pose", left_ee_pose_obs)

    right_ee_pose_obs = trajectory.get_ee_pose_obs(ArmSide.RIGHT, frame_convention)[frame]
    log_transform("obs-right-pose", right_ee_pose_obs)

    # Log joint positions as scalars
    for arm_side in ArmSide:
        joint_pos = trajectory.get_joint_pos_action(arm_side)[frame]
        for i, value in enumerate(joint_pos):
            rr.log(f"action-{arm_side.value}-pos/{i}", rr.Scalars(value))

        gripper_pos = trajectory.get_gripper_pos_action(arm_side)[frame]
        for i, value in enumerate(gripper_pos):
            rr.log(f"{arm_side.value}-gripper_pos/{i}", rr.Scalars(value))


def visualize_xmi(
    trajectory: XmiTrajectory,
    frame: int,
    frame_convention: FrameConvention = FrameConvention.GRIPPER_RDF,
):
    """Visualize XMI trajectory data."""

    rr.log("world_coordinate", rr.Transform3D(axis_length=0.1, translation=np.zeros(3)))

    # Head camera (camera pose directly, no relative transform needed)
    head_camera_pose = trajectory.get_head_camera_pose_action()[frame]
    log_coordinate_with_video(
        head_camera_pose,
        np.eye(4),  # Camera pose is already the camera, no relative transform
        # head_camera_pose,
        trajectory.get_video_path(CameraPerspective.TOP),
        TOP_CAMERA_NAME,
        focal_length=70,
    )

    # Left gripper and wrist camera
    left_ee_pose = trajectory.get_ee_pose_action(ArmSide.LEFT, frame_convention)[frame]
    left_camera_pose = trajectory.get_wrist_camera_pose_action(ArmSide.LEFT)[frame]
    # Calculate camera pose relative to gripper
    left_camera_relative = np.linalg.inv(left_ee_pose) @ left_camera_pose

    log_coordinate_with_video(
        left_ee_pose,
        left_camera_relative,
        trajectory.get_video_path(CameraPerspective.LEFT),
        LEFT_CAMERA_NAME,
        focal_length=70,
    )

    # Right gripper and wrist camera
    right_ee_pose = trajectory.get_ee_pose_action(ArmSide.RIGHT, frame_convention)[frame]
    right_camera_pose = trajectory.get_wrist_camera_pose_action(ArmSide.RIGHT)[frame]
    # Calculate camera pose relative to gripper
    right_camera_relative = np.linalg.inv(right_ee_pose) @ right_camera_pose

    log_coordinate_with_video(
        right_ee_pose,
        right_camera_relative,
        trajectory.get_video_path(CameraPerspective.RIGHT),
        RIGHT_CAMERA_NAME,
        focal_length=70,
    )

    # Gripper positions (scalar values)
    left_gripper = trajectory.get_gripper_pos_action(ArmSide.LEFT)[frame]
    rr.log("left_gripper_pos", rr.Scalars(left_gripper))

    right_gripper = trajectory.get_gripper_pos_action(ArmSide.RIGHT)[frame]
    rr.log("right_gripper_pos", rr.Scalars(right_gripper))


def main(data_path: str, frame_convention: FrameConvention = FrameConvention.GRIPPER_RDF):
    """Main visualization function."""
    rr.init("trajectory_viz", spawn=True)

    # Load trajectory using new library
    trajectory = load_trajectory(Path(data_path))

    frames = trajectory.n_frames
    timestamps = [int(t * (1e9 // FPS)) for t in range(frames)]

    print(f"Visualizing {frames} frames of {type(trajectory).__name__} trajectory")

    log_video(
        TOP_CAMERA_NAME,
        trajectory.get_video_path(CameraPerspective.TOP),
        timestamps,
    )

    log_video(
        LEFT_CAMERA_NAME,
        trajectory.get_video_path(CameraPerspective.LEFT),
        timestamps,
    )
    log_video(
        RIGHT_CAMERA_NAME,
        trajectory.get_video_path(CameraPerspective.RIGHT),
        timestamps,
    )

    # Visualize
    for frame, timestamp in enumerate(timestamps):
        rr.set_time("time", timestamp=1e-9 * timestamp)

        for label in trajectory.annotation_map[frame]:
            rr.log("annotations", rr.TextLog(label, level=rr.TextLogLevel.INFO))

        # Visualize data
        if isinstance(trajectory, XmiTrajectory):
            visualize_xmi(trajectory, frame, frame_convention)
        elif isinstance(trajectory, ArmTrajectory):
            visualize_robot(trajectory, frame, frame_convention)


if __name__ == "__main__":
    tyro.cli(main)

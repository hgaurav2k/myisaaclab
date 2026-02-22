from typing import Any

import numpy as np
import numpy.typing as npt
import quaternion

from .constants import ArmSide

# --- Constants and Helper Functions (Unchanged) ---

# see from the back of the camera lens                     / z
#       z                                                 /
#       ^    x                                            /
#       |   ^                                            |------> x
#       |  /   -> pin whole camera convention            |
#       | /                                              |
#    y--|/                                               |y
CAMERA_LOCAL_FRAME_WORLD_TO_CAMERA_CONVENTION = np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
CALIB_FRAME_TO_WORLD_FRAME = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])


def convert_left_handed_to_right_handed(quest_matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    y_flip_transform = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    intermediate_result = y_flip_transform @ quest_matrix
    final_result = intermediate_result @ y_flip_transform.T
    return final_result


def load_pose(input_dict: dict[str, np.ndarray]) -> np.ndarray:
    assert "position" in input_dict
    assert "rotation" in input_dict or "quaternion_wxyz" in input_dict
    position = np.array(input_dict["position"])
    if "rotation" in input_dict:
        rotation_wxyz = np.array(input_dict["rotation"])
    else:
        rotation_wxyz = np.array(input_dict["quaternion_wxyz"])
    rotation_matrix = quaternion.as_rotation_matrix(quaternion.from_float_array(rotation_wxyz))
    pose = np.eye(4)
    pose[:3, 3] = position
    pose[:3, :3] = rotation_matrix
    return pose


def movement_calib_frame_to_plot_world_frame(pose_in_calib_frame: np.ndarray) -> np.ndarray:
    return CALIB_FRAME_TO_WORLD_FRAME @ pose_in_calib_frame @ CALIB_FRAME_TO_WORLD_FRAME.T


def get_average_head_pose_collapose_to_z_up(head_poses_mat: np.ndarray) -> np.ndarray:
    Z_AXIS_OFFSET = -1.7
    head_poses_mat_in_calibration_frame = convert_left_handed_to_right_handed(head_poses_mat)
    head_poses_mat_in_world_frame = movement_calib_frame_to_plot_world_frame(head_poses_mat_in_calibration_frame)
    head_poses_mat_in_world_frame_average = np.mean(head_poses_mat_in_world_frame, axis=0)
    plot_world_frame = head_poses_mat_in_world_frame_average.copy()
    plot_world_frame[:3, 3] += np.array([0, 0, Z_AXIS_OFFSET])
    plot_world_frame[:3, 2] = np.array([0, 0, 1])
    x_axis = plot_world_frame[:3, 0]
    x_axis[2] = 0
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = -np.cross(x_axis, np.array([0, 0, 1]))
    plot_world_frame[:3, 1] = y_axis
    plot_world_frame[:3, 0] = x_axis
    return plot_world_frame


class XmiHelper:
    """
    This class manages the XMI episode: maintaining a fixed 'world frame' and converting the raw
    data stream and camera calibration data into that world frame.

    World frame: z-up, x-forward, y-left
    Camera/Gripper convention: z-forward, x-right, y-down
    """

    def __init__(self, metadata: dict[str, Any], head_poses_quest_world_frame: np.ndarray):
        if "station_metadata" in metadata and "data_version" in metadata["station_metadata"]:
            self.data_version = metadata["station_metadata"]["data_version"]
        else:
            self.data_version = "0.0"

        # Load raw extrinsic matrices from metadata
        top_camera_in_quest_head_raw = load_pose(
            metadata["station_metadata"]["extrinsics"]["top_camera_in_quest_head"]
        )
        left_gripper_in_controller_raw = load_pose(
            metadata["station_metadata"]["extrinsics"]["gripper_in_left_controller"]
        )
        right_gripper_in_controller_raw = load_pose(
            metadata["station_metadata"]["extrinsics"]["gripper_in_right_controller"]
        )
        self.wrist_camera_in_gripper_flange = load_pose(
            metadata["station_metadata"]["extrinsics"]["gripper_camera_in_gripper"]
        )
        self.left_intrinsics = np.array(
            metadata["camera_info"]["top_camera"]["intrinsic_data"]["cameras"]["left_rgb"]["intrinsics_matrix"]
        )

        # Establish the final, stable world frame based on average head pose
        self.final_world_frame = get_average_head_pose_collapose_to_z_up(head_poses_quest_world_frame)
        self.to_final_world_frame = np.linalg.inv(self.final_world_frame)

        # 1. Head camera relative to the head
        self.head_T_top_camera = (
            movement_calib_frame_to_plot_world_frame(top_camera_in_quest_head_raw)
            @ CAMERA_LOCAL_FRAME_WORLD_TO_CAMERA_CONVENTION.T
        )

        # 2. Grippers relative to their controllers
        self.controller_T_left_gripper = (
            movement_calib_frame_to_plot_world_frame(left_gripper_in_controller_raw)
            @ CAMERA_LOCAL_FRAME_WORLD_TO_CAMERA_CONVENTION.T
        )
        self.controller_T_right_gripper = (
            movement_calib_frame_to_plot_world_frame(right_gripper_in_controller_raw)
            @ CAMERA_LOCAL_FRAME_WORLD_TO_CAMERA_CONVENTION.T
        )

        # 3. Wrist cameras relative to their controllers (Hand -> Gripper -> Camera)
        self.controller_T_left_wrist_camera = self.controller_T_left_gripper @ self.wrist_camera_in_gripper_flange
        self.controller_T_right_wrist_camera = self.controller_T_right_gripper @ self.wrist_camera_in_gripper_flange

    def get_head_data(self, head_poses_mat_quest_world_frame: np.ndarray) -> npt.NDArray[np.float64]:
        """
        Gets the head pose in the final world frame and the constant transform from the
        head to its camera.

        Returns:
            head_poses_in_world_frame: Head poses in the final world frame. Shape (N, 4, 4).
        """
        head_poses_in_calibration_frame = convert_left_handed_to_right_handed(head_poses_mat_quest_world_frame)
        head_poses_in_plot_world_frame = movement_calib_frame_to_plot_world_frame(head_poses_in_calibration_frame)
        head_poses_in_world_frame = self.to_final_world_frame @ head_poses_in_plot_world_frame

        return head_poses_in_world_frame

    def get_controller_data(
        self, hand_poses_mat_quest_world_frame: np.ndarray, arm_side: ArmSide
    ) -> npt.NDArray[np.float64]:
        """
        Gets controller poses in the final world frame.

        Returns:
            controller_poses_in_world_frame: Controller poses in the final world frame. Shape (N, 4, 4).
        """
        hand_poses_in_calibration_frame = convert_left_handed_to_right_handed(hand_poses_mat_quest_world_frame)
        hand_poses_in_plot_world_frame = movement_calib_frame_to_plot_world_frame(hand_poses_in_calibration_frame)
        hand_poses_in_world_frame = self.to_final_world_frame @ hand_poses_in_plot_world_frame

        return hand_poses_in_world_frame

import os
from typing import Any, Optional

import mink
import mujoco
import numpy as np
from mink import SE3
from numpy.typing import NDArray

YAM_XML_PATH = os.path.join(os.path.dirname(__file__), "models/yam.xml")
FRANKA_XML_PATH = os.path.join(os.path.dirname(__file__), "models/panda.xml")


class MuJoCoFK:
    def __init__(self, model: Any, site_name: Optional[str] = None):
        self.model = model
        self.configuration = mink.Configuration(model)
        self._site_name = site_name

    def forward_kinematics(
        self, q: np.ndarray, site_name: Optional[str] = None, body_name: Optional[str] = None
    ) -> mink.SE3:
        self.configuration.update(q)
        if body_name:
            return self.configuration.get_transform_frame_to_world(body_name, "body")

        site_name = site_name or self._site_name
        assert site_name is not None, "site_name must be provided if body_name is not provided"
        return self.configuration.get_transform_frame_to_world(site_name, "site")


class RobotFK:
    def __init__(self, metadata: dict[Any, Any]):
        robot_type = metadata.get("station_metadata", {}).get("arm_type", None)

        if robot_type in ("yam", "arx"):
            model = mujoco.MjModel.from_xml_path(YAM_XML_PATH)
            self._mj_model = MuJoCoFK(model, "grasp_site")
            self._num_joints = 6
        elif robot_type == "franka":
            model = mujoco.MjModel.from_xml_path(FRANKA_XML_PATH)
            self._mj_model = MuJoCoFK(model, "grasp_site")
            self._num_joints = 7
        else:
            raise NotImplementedError(f"Robot type {robot_type} not supported for FK computation")

        right_arm_extrinsic = (
            metadata.get("station_metadata", {}).get("extrinsics", {}).get("right_arm_extrinsic", None)
        )
        if right_arm_extrinsic:
            self.right_T_left = SE3(
                wxyz_xyz=np.concatenate([right_arm_extrinsic["rotation"], right_arm_extrinsic["position"]])
            ).as_matrix()
        else:
            self.right_T_left = np.array(
                [
                    [1, 0, 0, 0.0],
                    [0, 1, 0, -0.61],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
            )

    def fk(
        self,
        q: NDArray[np.float64],
        extrinsics: NDArray[np.float64] | None = None,
        site_name: str | None = None,
        body_name: str | None = None,
    ) -> NDArray[np.float64]:
        mujoco_fk = np.array(
            [
                self._mj_model.forward_kinematics(q[i, : self._num_joints], site_name, body_name).as_matrix()
                for i in range(q.shape[0])
            ]
        )

        if extrinsics is not None:
            return extrinsics @ np.array(mujoco_fk)

        return np.array(mujoco_fk)

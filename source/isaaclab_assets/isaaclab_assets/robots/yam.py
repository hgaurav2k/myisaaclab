# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the YAM robot.

The following configurations are available:

* :obj:`YAM_CFG`: YAM single-arm robot with parallel-jaw gripper
* :obj:`YAM_HIGH_PD_CFG`: YAM single-arm robot with stiffer PD control
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# Path to the YAM single-arm MJCF file
_YAM_MJCF_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "xdof_samples", "station_mjcf", "yam_single_arm.xml"
)

##
# Configuration
##

YAM_CFG = ArticulationCfg(
    spawn=sim_utils.MjcfFileCfg(
        asset_path=_YAM_MJCF_PATH,
        fix_base=True,
        import_sites=True,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint1": 0.0,
            "joint2": 1.0,
            "joint3": 1.5,
            "joint4": 0.0,
            "joint5": 0.0,
            "joint6": 0.0,
            "left_finger": 0.02,
            "right_finger": -0.02,
        },
    ),
    actuators={
        "yam_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-3]"],
            effort_limit_sim=28.0,
            stiffness=40.0,
            damping=2.5,
        ),
        "yam_wrist": ImplicitActuatorCfg(
            joint_names_expr=["joint[4-6]"],
            effort_limit_sim=10.0,
            stiffness=10.0,
            damping=1.0,
        ),
        "yam_gripper": ImplicitActuatorCfg(
            joint_names_expr=["left_finger", "right_finger"],
            effort_limit_sim=100.0,
            stiffness=100.0,
            damping=10.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of YAM single-arm robot."""


YAM_HIGH_PD_CFG = YAM_CFG.copy()
YAM_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
YAM_HIGH_PD_CFG.actuators["yam_shoulder"].stiffness = 200.0
YAM_HIGH_PD_CFG.actuators["yam_shoulder"].damping = 40.0
YAM_HIGH_PD_CFG.actuators["yam_wrist"].stiffness = 100.0
YAM_HIGH_PD_CFG.actuators["yam_wrist"].damping = 20.0
"""Configuration of YAM robot with stiffer PD control.

This configuration is useful for task-space control using differential IK.
"""

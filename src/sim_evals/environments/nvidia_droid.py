import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any
from isaaclab.utils import configclass
from isaaclab.assets import AssetBaseCfg, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sensors import CameraCfg
import isaaclab.sim as sim_utils
import numpy as np
import math
import torch

ASSET_PATH = Path(__file__).parent / "../../../assets"

NVIDIA_DROID = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(ASSET_PATH / "franka_robotiq_2f_85_flattened.usd"),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=64,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0, 0, 0),
            rot=(1, 0, 0, 0),
            joint_pos={
                "panda_joint1": 0.0,
                "panda_joint2": -1 / 5 * np.pi,
                "panda_joint3": 0.0,
                "panda_joint4": -4 / 5 * np.pi,
                "panda_joint5": 0.0,
                "panda_joint6": 3 / 5 * np.pi,
                "panda_joint7": 0,
                "finger_joint": 0.0,
                "right_outer.*": 0.0,
                "left_inner.*": 0.0,
                "right_inner.*": 0.0,
            },
        ),
        soft_joint_pos_limit_factor=1,
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit=87.0,
                velocity_limit=2.175,
                stiffness=400.0,
                damping=80.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit=12.0,
                velocity_limit=2.61,
                stiffness=400.0,
                damping=80.0,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["finger_joint"],
                stiffness=1000.0,
                damping=None,
                # PhysX max joint velocity (rad/s) for the finger. For IMPLICIT actuators `velocity_limit`
                # is IGNORED by the sim -- it must be `velocity_limit_sim` (which writes the PhysX
                # set_dof_max_velocities). Without an enforced cap the stiff finger snaps shut in ~1 frame,
                # so the recorded gripper proprioception/action was a binary {0,1} signal. Capping it makes
                # the finger ramp over ~0.67 s (~10 frames @ 15 Hz), matching the DROID gripper's continuous
                # transition distribution: swept 0.3-2.0 rad/s, 0.9 minimizes the 1-Wasserstein distance to
                # DROID's per-step slew (0.043 vs 0.92 uncapped) and matches its ~10-frame median transition.
                # The tiptop datagen holds the arm 20 frames per gripper action, so the finger fully
                # actuates before the arm moves on.
                # NOTE: 0.9 rad/s was TOO SLOW for the TAMP grasps — the gently-closing finger nudges the
                # light photogrammetry toys away before trapping them, collapsing scene-6 success to ~0/32.
                # Tuning for BOTH a continuous (non-binary) gripper AND reliable grasps -> see _sweep below;
                # this value is the chosen tradeoff (still a multi-frame ramp, far from the 1-frame snap).
                velocity_limit_sim=2.0,
            ),
        },
    )


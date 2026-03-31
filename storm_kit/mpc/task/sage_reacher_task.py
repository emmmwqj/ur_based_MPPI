#
# MIT License
#
# Copyright (c) 2020-2021 NVIDIA CORPORATION.
#

import torch

from ..rollout.arm_reacher import ArmReacher
from .sage_arm_task import SageArmTask


class SageReacherTask(SageArmTask):
    """
    Reaching task assembly for SAGE-MPPI.
    """

    def __init__(
        self,
        task_file="ur10.yml",
        robot_file="ur10_reacher.yml",
        world_file="collision_env.yml",
        tensor_args={"device": "cpu", "dtype": torch.float32},
    ):
        super().__init__(
            task_file=task_file,
            robot_file=robot_file,
            world_file=world_file,
            tensor_args=tensor_args,
        )

    def get_rollout_fn(self, **kwargs):
        return ArmReacher(**kwargs)

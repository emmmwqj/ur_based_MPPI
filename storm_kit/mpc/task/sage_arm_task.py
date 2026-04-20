#
# MIT License
#
# Copyright (c) 2020-2021 NVIDIA CORPORATION.
#

"""
Canonical SAGE arm task entry.

The latest clean SAGE task implementation lives in ``sage_arm_task_impl.py``.
This file preserves the stable public task name ``SageArmTask``.
"""

from .sage_arm_task_impl import SageArmTaskV3


class SageArmTask(SageArmTaskV3):
    """Latest canonical SAGE arm task."""

    pass

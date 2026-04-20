#
# MIT License
#
# Copyright (c) 2020-2021 NVIDIA CORPORATION.
#

"""
Canonical SAGE reaching task entry.

This file preserves the stable public task name ``SageReacherTask`` while the
implementation remains separated from the canonical wrapper.
"""

from .sage_reacher_task_v3 import SageReacherTaskV3


class SageReacherTask(SageReacherTaskV3):
    """Latest canonical SAGE reaching task."""

    pass

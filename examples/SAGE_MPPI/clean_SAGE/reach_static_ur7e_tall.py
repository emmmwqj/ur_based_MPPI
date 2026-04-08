#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clean SAGE tall-scene launcher.

This wrapper exists so the clean pipeline has its own standalone project
directory under ``examples/SAGE_MPPI/clean_SAGE`` while reusing the verified
clean example entry implemented in ``examples/sim_gazebo``.

Behavioral intent:
- same tall scene and Gazebo/ROS2 flow as the original SAGE tall project
- controller path is the independent clean path:
    SageReacherTaskV3 -> SAGE_MPPI_CORE
"""

from __future__ import annotations

import os
import sys

STORM_ROOT = os.path.expanduser("~/storm")
if STORM_ROOT not in sys.path:
    sys.path.insert(0, STORM_ROOT)

from examples.sim_gazebo.reach_static_ur7e_sage_clean import main


if __name__ == "__main__":
    sys.exit(main())

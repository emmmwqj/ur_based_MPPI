# Joint Smoothness Report

Lower acceleration, jerk, and integrated squared jerk indicate smoother motion.
`joint_state` is the executed Gazebo trajectory. `command` is the target position stream sent to `/forward_position_controller/commands`.

## Global Metrics

| stream | samples | duration_s | median_dt_s | mean_joint_jerk_rms_rad_s3 | max_joint_jerk_abs_rad_s3 | mean_integrated_squared_jerk |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| joint_state | 5348 | 53.47 | 0.01 | 422.445 | 6818.17 | 1.49401e+07 |
| command | 800 | 47.5085 | 0.0527744 | 0.424553 | 7.33449 | 9.58694 |

## Per-Joint Executed Trajectory

| joint | vel_rms | acc_rms | jerk_rms | max_abs_jerk | integrated_squared_jerk |
| --- | ---: | ---: | ---: | ---: | ---: |
| shoulder_pan_joint | 0.000324445 | 11.5349 | 1035.78 | 6818.17 | 5.73759e+07 |
| shoulder_lift_joint | 0.00733819 | 2.88278 | 254.156 | 1412.78 | 3.45456e+06 |
| elbow_joint | 0.0199659 | 1.74728 | 152.153 | 1357.21 | 1.2381e+06 |
| wrist_1_joint | 0.0234107 | 2.85062 | 248.998 | 1888.83 | 3.31575e+06 |
| wrist_2_joint | 0.00671347 | 2.24813 | 200.715 | 1090.22 | 2.15452e+06 |
| wrist_3_joint | 0.000754003 | 7.45031 | 642.865 | 4134.07 | 2.2102e+07 |

## Per-Joint Command Trajectory

| joint | vel_rms | acc_rms | jerk_rms | max_abs_jerk | integrated_squared_jerk |
| --- | ---: | ---: | ---: | ---: | ---: |
| shoulder_pan_joint | 0.101581 | 0.10344 | 0.801417 | 7.33449 | 27.3859 |
| shoulder_lift_joint | 0.0225735 | 0.0358772 | 0.3696 | 3.93732 | 6.12066 |
| elbow_joint | 0.012346 | 0.0233716 | 0.245145 | 3.10389 | 2.5619 |
| wrist_1_joint | 0.0210806 | 0.0408267 | 0.31675 | 4.42116 | 4.50841 |
| wrist_2_joint | 0.0191759 | 0.0237915 | 0.27074 | 2.89174 | 3.08847 |
| wrist_3_joint | 0.0498971 | 0.0537299 | 0.543666 | 5.45507 | 13.8563 |

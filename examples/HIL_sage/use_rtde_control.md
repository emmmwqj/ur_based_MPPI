终端 1，启动 URSim/UR7e 的 ROS driver，让它发布 /joint_states：

  cd /home/wqj/storm/examples/HIL_sage
  ./run_ur_driver.sh

  终端 2，启动 RTDE servoJ 版控制器：

  cd /home/wqj/storm/examples/HIL_sage
  ./run_hil_sage_mpc_rtde.sh \
    --robot-ip 192.168.56.100 \
    --servo-frequency 500 \
    --lookahead-time 0.10 \
    --gain 300 \
    --max-joint-speed 0.5


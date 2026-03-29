# Recovery Validation

## 目标
在不改原始高墙主脚本的前提下，验证一套可跳出局部 basin 的恢复策略。

## 我在 debug 版里做的修复
文件：
- `~/storm/examples/sim_gazebo/debug/reach_static_ur7e_tall_debug.py`

修复内容：
1. 目标切换时不再重启 `ControlProcess`
2. 目标切换时只重置时间基准
3. 目标切换时重置采样分布，并把协方差放大到 `9x`
4. 当接近目标时恢复默认采样分布
5. 当检测到“末端远离目标且几乎不动”时，再把协方差放大到 `16x` 做一次恢复尝试

## 第二轮实际验证
日志：
- `~/storm/examples/sim_gazebo/debug/logs/run_20260328_204627.log`

验证序列：
1. 从当前卡住状态开始
2. 切换到 `world=[0.403, 0.400, 0.500]`
3. 再切回 `world=[0.500, -0.450, 0.400]`

## 验证结果
### 1. 目标切换时的控制流程 bug 消失了
第二轮没有再出现：
- `后台 MPC 进程未在超时内退出，强制终止...`
- `index 0 is out of bounds for dimension 0 with size 0`

这说明：
- 去掉 `_restart_control_process(...)`
- 改成 `_reset_control_process_timing(...)`
是对的。

### 2. 跨墙目标切换后，控制器明显更容易离开原 basin
切到 `world=[0.403, 0.400, 0.500]` 后：
- `ee_error` 从 `0.6172` 很快降到 `0.1954`
- 然后继续降到 `0.0654`

说明放大协方差后的重置分布确实让控制器跳到了新的可行绕障模式。

### 3. 回切到 `world=[0.500, -0.450, 0.400]` 时，虽然仍会短暂停滞，但能被恢复策略拉出来
回切后先出现：
- `ee_error ≈ 0.8467`

随后 debug 恢复触发，协方差被放大到：
- `cov_action = 0.045`（原来是 `0.005`）
- `scale_tril ≈ 0.2121`

之后 `ee_error` 逐步下降：
- `0.8459`
- `0.7449`
- `0.4919`
- `0.2777`
- `0.2032`
- `0.1674`
- `0.1499`
- `0.1065`
- `0.0723`
- `0.0515`
- `0.0474`

这说明：
- 原来的停滞并不是不可恢复
- 真正缺的是“目标切换时的重新探索能力”

## 结论
这套修复策略是有效的。

最关键的两个点：
1. 不要在目标切换时重启 `ControlProcess`
2. 目标切换或停滞时，要显式重置采样分布并放大协方差

## 建议
下一步最合理的是把这套已经验证过的策略移植到：
- `~/storm/examples/sim_gazebo/reach_static_ur7e_tall.py`

建议移植内容：
1. 去掉 `_restart_control_process(...)`
2. 使用 `_reset_control_process_timing(...)`
3. 增加目标切换时的 distribution reset
4. 增加简化版 stall recovery

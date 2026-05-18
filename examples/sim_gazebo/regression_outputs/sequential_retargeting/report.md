# Baseline STORM MPPI Sequential Retargeting Report

- Config: `/home/wqj/storm/examples/sim_gazebo/config/ur7e_reacher_gazebo_tall.yml`
- Seeds: `[0, 1, 2]`

## Target Sequence

- p0_default: (0.4, -0.5, 0.4)
- p1: (0.4, -0.5, 0.4)
- p2: (0.33, 0.65, 0.3)
- p3: (0.54, 0.0, 0.5)
- p4: (0.33, 0.65, 0.3)
- p5: (0.36, -0.57, 0.43)
- p6: (0.4, 0.1, 0.1)

## Overall

- Runs: `3`
- Segments: `21`
- <2cm success rate: `9.5%`
- <5mm success rate: `9.5%`
- Rebound count: `4`
- Stalled count: `13`

## Per-Target Summary

| Target | Mean Final EE Error | <2cm | <5mm | Rebound Count | Stalled Count | Worst Final |
|---|---:|---:|---:|---:|---:|---:|
| p0_default | 0.345067 | 0.0% | 0.0% | 0 | 2 | 0.362800 |
| p1 | 0.343300 | 0.0% | 0.0% | 0 | 3 | 0.358400 |
| p2 | 0.101500 | 33.3% | 33.3% | 1 | 0 | 0.159800 |
| p3 | 0.566233 | 0.0% | 0.0% | 1 | 3 | 0.576400 |
| p4 | 0.025700 | 33.3% | 33.3% | 2 | 2 | 0.042900 |
| p5 | 0.089433 | 0.0% | 0.0% | 0 | 3 | 0.183100 |
| p6 | 0.577267 | 0.0% | 0.0% | 0 | 0 | 0.746800 |

## Worst Segment

- Seed: `1`
- Target: `p6`
- Final EE error: `0.746800`
- Peak EE error: `0.746800`
- <2cm time: `None`
- <5mm time: `None`
- Rebound: `False`
- Stalled: `False`
- Stall reason: `controller_exited`

## Detailed Per-Run Segments

### Seed 0

| Target | Peak EE Error | Min EE Error | Final EE Error | <2cm Time | <5mm Time | Rebound | Stalled | Stall Reason |
|---|---:|---:|---:|---:|---:|---|---|---|
| p0_default | 0.363100 | 0.360800 | 0.362800 | - | - | False | False | - |
| p1 | 0.365500 | 0.358400 | 0.358400 | - | - | False | True | segment_timeout |
| p2 | 0.841200 | 0.141300 | 0.141300 | - | - | False | False | - |
| p3 | 0.702200 | 0.055500 | 0.576400 | - | - | True | True | segment_timeout |
| p4 | 0.576400 | 0.001500 | 0.001500 | 6.63s | 7.66s | False | False | - |
| p5 | 1.231800 | 0.183100 | 0.183100 | - | - | False | True | segment_timeout |
| p6 | 0.741600 | 0.444200 | 0.444200 | - | - | False | True | controller_exited |

### Seed 1

| Target | Peak EE Error | Min EE Error | Final EE Error | <2cm Time | <5mm Time | Rebound | Stalled | Stall Reason |
|---|---:|---:|---:|---:|---:|---|---|---|
| p0_default | 0.583700 | 0.336100 | 0.336100 | - | - | False | True | no_progress |
| p1 | 0.336100 | 0.335200 | 0.335200 | - | - | False | True | no_progress |
| p2 | 0.913900 | 0.134000 | 0.159800 | - | - | True | False | - |
| p3 | 0.699900 | 0.560800 | 0.562300 | - | - | False | True | no_progress |
| p4 | 0.562300 | 0.020800 | 0.042900 | - | - | True | True | segment_timeout |
| p5 | 1.198500 | 0.030700 | 0.030700 | - | - | False | True | segment_timeout |
| p6 | 0.746800 | 0.746800 | 0.746800 | - | - | False | True | controller_exited |

### Seed 2

| Target | Peak EE Error | Min EE Error | Final EE Error | <2cm Time | <5mm Time | Rebound | Stalled | Stall Reason |
|---|---:|---:|---:|---:|---:|---|---|---|
| p0_default | 0.579000 | 0.336300 | 0.336300 | - | - | False | True | no_progress |
| p1 | 0.336300 | 0.336300 | 0.336300 | - | - | False | True | no_progress |
| p2 | 0.900400 | 0.002100 | 0.003400 | 7.26s | 7.26s | False | False | - |
| p3 | 0.710900 | 0.559200 | 0.560000 | - | - | False | True | no_progress |
| p4 | 0.560000 | 0.018200 | 0.032700 | 5.98s | - | True | True | segment_timeout |
| p5 | 1.201400 | 0.054500 | 0.054500 | - | - | False | True | segment_timeout |
| p6 | 0.752600 | 0.540800 | 0.540800 | - | - | False | True | controller_exited |


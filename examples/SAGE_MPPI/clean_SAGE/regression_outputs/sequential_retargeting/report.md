# Sequential Retargeting Regression Report

- Config: `/home/wqj/storm/examples/SAGE_MPPI/clean_SAGE/config/ur7e_reacher_gazebo_tall_sage_clean.yml`
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
- <2cm success rate: `100.0%`
- <5mm success rate: `100.0%`
- Rebound count: `0`
- Local refinement triggered rate: `100.0%`

## Per-Target Summary

| Target | Mean Final EE Error | <2cm | <5mm | Rebound Count | LR Triggered | Worst Final |
|---|---:|---:|---:|---:|---:|---:|
| p0_default | 0.000000 | 100.0% | 100.0% | 0 | 3/3 | 0.000000 |
| p1 | 0.000000 | 100.0% | 100.0% | 0 | 3/3 | 0.000000 |
| p2 | 0.000067 | 100.0% | 100.0% | 0 | 3/3 | 0.000200 |
| p3 | 0.000000 | 100.0% | 100.0% | 0 | 3/3 | 0.000000 |
| p4 | 0.000000 | 100.0% | 100.0% | 0 | 3/3 | 0.000000 |
| p5 | 0.000167 | 100.0% | 100.0% | 0 | 3/3 | 0.000500 |
| p6 | 0.000000 | 100.0% | 100.0% | 0 | 3/3 | 0.000000 |

## Worst Segment

- Seed: `2`
- Target: `p5`
- Final EE error: `0.000500`
- Peak EE error: `1.225100`
- <2cm time: `11.129999999999995`
- <5mm time: `11.129999999999995`
- Rebound: `False`
- Local refinement triggered: `True`

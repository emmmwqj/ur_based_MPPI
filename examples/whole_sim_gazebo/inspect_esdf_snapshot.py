#!/usr/bin/env python3
import argparse
import os
import sys

import numpy as np
import torch

STORM_ROOT = os.path.expanduser('~/storm')
if STORM_ROOT not in sys.path:
    sys.path.insert(0, STORM_ROOT)

from examples.whole_sim_gazebo.esdf_snapshot import ESDFSnapshot


def main() -> int:
    parser = argparse.ArgumentParser(description='Inspect and sanity-check an ESDF snapshot.')
    parser.add_argument(
        '--snapshot',
        type=str,
        default='/home/wqj/perception_D435i/src/sim_nvblox/result/latest_esdf_snapshot.npz',
    )
    parser.add_argument('--samples', type=int, default=5)
    args = parser.parse_args()

    tensor_args = {'device': torch.device('cpu'), 'dtype': torch.float32}
    snapshot = ESDFSnapshot(args.snapshot, tensor_args=tensor_args, interpolation='trilinear')

    valid_idx = torch.nonzero(snapshot.valid_mask, as_tuple=False)
    if valid_idx.numel() == 0:
        print('No valid ESDF voxels found.')
        return 1

    sample_count = min(args.samples, valid_idx.shape[0])
    sample_idx = valid_idx[:sample_count]
    points_world = snapshot.origin_world.unsqueeze(0) + sample_idx.to(dtype=torch.float32) * snapshot.voxel_size
    queried, queried_valid = snapshot.query(points_world)
    reference = snapshot.esdf[
        sample_idx[:, 0],
        sample_idx[:, 1],
        sample_idx[:, 2],
    ]

    print('Sampled valid voxel centers:')
    for i in range(sample_count):
        print(
            '  idx=%s point=%s ref=%.6f query=%.6f valid=%s'
            % (
                sample_idx[i].tolist(),
                [round(float(v), 4) for v in points_world[i].tolist()],
                float(reference[i].item()),
                float(queried[i].item()),
                bool(queried_valid[i].item()),
            )
        )

    max_err = torch.max(torch.abs(queried - reference)).item()
    print('max_abs_error_at_voxel_centers=%.8f' % max_err)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

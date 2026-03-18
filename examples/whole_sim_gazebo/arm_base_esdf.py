import copy

from storm_kit.mpc.rollout.arm_base import ArmBase

from examples.whole_sim_gazebo.esdf_collision_cost import ESDFCollisionCost


def _log(message: str) -> None:
    print(f"[ArmBaseESDF] {message}", flush=True)


class ArmBaseESDF(ArmBase):
    """Arm rollout that keeps STORM's base costs but replaces env collision with ESDF."""

    def __init__(self, exp_params, tensor_args=None, world_params=None):
        if tensor_args is None:
            tensor_args = {"device": "cpu", "dtype": None}

        exp_params_local = copy.deepcopy(exp_params)
        primitive_weight = float(exp_params_local["cost"]["primitive_collision"].get("weight", 0.0))
        voxel_weight = float(exp_params_local["cost"]["voxel_collision"].get("weight", 0.0))
        exp_params_local["cost"]["primitive_collision"]["weight"] = 0.0
        exp_params_local["cost"]["voxel_collision"]["weight"] = 0.0

        super().__init__(exp_params=exp_params_local, tensor_args=tensor_args, world_params=world_params)

        self.esdf_collision_cost = None
        self.self_collision_enabled = bool(exp_params_local["cost"]["robot_self_collision"]["weight"] > 0.0)
        if exp_params_local["cost"]["esdf_collision"]["weight"] > 0.0:
            self.esdf_collision_cost = ESDFCollisionCost(
                world_params=world_params,
                robot_params=exp_params_local["robot_params"],
                tensor_args=self.tensor_args,
                **exp_params_local["cost"]["esdf_collision"],
            )
        if self.self_collision_enabled:
            robot_nn = getattr(self.robot_self_collision_cost.coll, "robot_nn", None)
            if robot_nn is None or not hasattr(robot_nn, "norm_dict"):
                self.self_collision_enabled = False
                self.exp_params["cost"]["robot_self_collision"]["weight"] = 0.0
                _log("Self-collision weights are unavailable; robot_self_collision disabled for this example.")

        _log(
            "Environment collision source set to ESDF snapshot. primitive_collision=%.1f voxel_collision=%.1f esdf_collision=%.1f"
            % (
                primitive_weight,
                voxel_weight,
                float(exp_params_local["cost"]["esdf_collision"]["weight"]),
            )
        )

    def cost_fn(self, state_dict, action_batch, no_coll=False, horizon_cost=True):
        cost = super().cost_fn(state_dict, action_batch, no_coll=True, horizon_cost=horizon_cost)

        if no_coll:
            return cost

        state_batch = state_dict["state_seq"]
        link_pos_batch = state_dict["link_pos_seq"]
        link_rot_batch = state_dict["link_rot_seq"]

        if self.self_collision_enabled:
            cost += self.robot_self_collision_cost.forward(state_batch[:, :, : self.n_dofs])

        if self.esdf_collision_cost is not None:
            cost += self.esdf_collision_cost.forward(link_pos_batch, link_rot_batch)

        return cost

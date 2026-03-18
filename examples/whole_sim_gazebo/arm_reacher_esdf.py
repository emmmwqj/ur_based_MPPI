import torch

from storm_kit.differentiable_robot_model.coordinate_transform import (
    matrix_to_quaternion,
    quaternion_to_matrix,
)
from storm_kit.mpc.cost import DistCost, PoseCost

from examples.whole_sim_gazebo.arm_base_esdf import ArmBaseESDF


class ArmReacherESDF(ArmBaseESDF):
    """Reacher rollout with ESDF-based environment collision."""

    def __init__(self, exp_params, tensor_args=None, world_params=None):
        super().__init__(exp_params=exp_params, tensor_args=tensor_args, world_params=world_params)
        self.goal_state = None
        self.goal_ee_pos = None
        self.goal_ee_rot = None

        device = self.tensor_args["device"]
        float_dtype = self.tensor_args["dtype"]
        self.dist_cost = DistCost(
            **self.exp_params["cost"]["joint_l2"],
            device=device,
            float_dtype=float_dtype,
        )
        self.goal_cost = PoseCost(**exp_params["cost"]["goal_pose"], tensor_args=self.tensor_args)

    def cost_fn(self, state_dict, action_batch, no_coll=False, horizon_cost=True, return_dist=False):
        cost = super().cost_fn(state_dict, action_batch, no_coll=no_coll, horizon_cost=horizon_cost)
        ee_pos_batch = state_dict["ee_pos_seq"]
        ee_rot_batch = state_dict["ee_rot_seq"]
        state_batch = state_dict["state_seq"]

        goal_cost, rot_err_norm, goal_dist = self.goal_cost.forward(
            ee_pos_batch,
            ee_rot_batch,
            self.goal_ee_pos,
            self.goal_ee_rot,
        )
        cost += goal_cost

        if self.exp_params["cost"]["joint_l2"]["weight"] > 0.0 and self.goal_state is not None:
            disp_vec = state_batch[:, :, : self.n_dofs] - self.goal_state[:, : self.n_dofs]
            cost += self.dist_cost.forward(disp_vec)

        if return_dist:
            return cost, rot_err_norm, goal_dist

        if self.exp_params["cost"]["zero_acc"]["weight"] > 0:
            cost += self.zero_acc_cost.forward(
                state_batch[:, :, self.n_dofs * 2 : self.n_dofs * 3],
                goal_dist=goal_dist,
            )

        if self.exp_params["cost"]["zero_vel"]["weight"] > 0:
            cost += self.zero_vel_cost.forward(
                state_batch[:, :, self.n_dofs : self.n_dofs * 2],
                goal_dist=goal_dist,
            )

        return cost

    def update_params(
        self,
        retract_state=None,
        goal_state=None,
        goal_ee_pos=None,
        goal_ee_rot=None,
        goal_ee_quat=None,
    ):
        super().update_params(retract_state=retract_state)

        if goal_ee_pos is not None:
            self.goal_ee_pos = torch.as_tensor(goal_ee_pos, **self.tensor_args).unsqueeze(0)
            self.goal_state = None
        if goal_ee_rot is not None:
            self.goal_ee_rot = torch.as_tensor(goal_ee_rot, **self.tensor_args).unsqueeze(0)
            self.goal_ee_quat = matrix_to_quaternion(self.goal_ee_rot)
            self.goal_state = None
        if goal_ee_quat is not None:
            self.goal_ee_quat = torch.as_tensor(goal_ee_quat, **self.tensor_args).unsqueeze(0)
            self.goal_ee_rot = quaternion_to_matrix(self.goal_ee_quat)
            self.goal_state = None
        if goal_state is not None:
            self.goal_state = torch.as_tensor(goal_state, **self.tensor_args).unsqueeze(0)
            self.goal_ee_pos, self.goal_ee_rot = self.dynamics_model.robot_model.compute_forward_kinematics(
                self.goal_state[:, : self.n_dofs],
                self.goal_state[:, self.n_dofs : 2 * self.n_dofs],
                link_name=self.exp_params["model"]["ee_link_name"],
            )
            self.goal_ee_quat = matrix_to_quaternion(self.goal_ee_rot)

        return True

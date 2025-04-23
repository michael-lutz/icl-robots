# mypy: disable-error-code="override"
"""Defines simple task for training a walking policy for the default humanoid using an SSM actor."""

from dataclasses import dataclass
from typing import Generic, Literal, TypeVar, cast, get_args

from abc import ABC, abstractmethod
import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
import xax
from jaxtyping import Array, PRNGKeyArray

import ksim

from iclrobot.dh_walking.dh_walking import (
    NUM_JOINTS,
    HumanoidWalkingTask,
    HumanoidWalkingTaskConfig,
)

SSMBlockType = Literal["diagonal", "discrete_diagonal", "full_rank", "dplr", "gated", "discrete_dplr"]


def glorot(key: PRNGKeyArray, shape: tuple[int, ...]) -> Array:
    return jax.random.uniform(key, shape, minval=-1.0, maxval=1.0) * jnp.sqrt(2 / sum(shape))


class BaseSSMBlock(eqx.Module, ABC):
    @abstractmethod
    def forward(self, h: Array, x: Array) -> Array: ...


class SSMBlock(BaseSSMBlock):
    a_mat: Array
    b_mat: Array

    def __init__(self, hidden_size: int, *, key: PRNGKeyArray) -> None:
        key_a, key_b = jax.random.split(key)
        self.a_mat = glorot(key_a, (hidden_size, hidden_size))
        self.b_mat = glorot(key_b, (hidden_size, hidden_size))

    def forward(self, h: Array, x: Array) -> Array:
        """Performs a single forward pass."""
        h = self.a_mat @ h + self.b_mat.T @ x
        return h


class DiagSSMBlock(BaseSSMBlock):
    a_diag: Array
    b_mat: Array

    def __init__(self, hidden_size: int, *, key: PRNGKeyArray) -> None:
        keys = jax.random.split(key, 2)
        self.a_diag = glorot(keys[0], (hidden_size,))
        self.b_mat = glorot(keys[1], (hidden_size, hidden_size))

    def forward(self, h: Array, x: Array) -> Array:
        """Performs a single forward pass."""
        h = self.a_diag * h + self.b_mat.T @ x
        return h


class DiscretizedDiagSSMBlock(DiagSSMBlock):
    delta: Array

    def __init__(
        self,
        hidden_size: int,
        *,
        key: PRNGKeyArray,
        init_delta: float = 1.0,
        init_scale: float = 10.0,
    ) -> None:
        super().__init__(hidden_size, key=key)
        self.delta = jnp.array(init_delta)
        self.a_diag = jax.random.uniform(key, (hidden_size,), minval=-1.0, maxval=0.0) * init_scale

    def get_a_mat(self, x: Array) -> Array:
        """Discretize the diagonal matrix using zero-order hold."""
        a_diag_discrete = jnp.exp(self.a_diag * self.delta)
        return a_diag_discrete

    def get_b_mat(self, x: Array) -> Array:
        """Discretize the input matrix using zero-order hold."""
        delta_a_diag = self.a_diag * self.delta
        exp_a_diag = jnp.exp(delta_a_diag)
        delta_a_inv = 1 / delta_a_diag
        delta_b_mat = self.delta * self.b_mat

        b_discrete = delta_a_inv * (exp_a_diag - 1) * delta_b_mat
        return b_discrete

    def forward(self, h: Array, x: Array) -> Array:
        """Performs a single forward pass."""
        a_diag = self.get_a_mat(x)
        b_mat = self.get_b_mat(x)
        return a_diag * h + b_mat.T @ x


class DPLRSSMBlock(BaseSSMBlock):
    a_diag: Array
    p_vec: Array
    q_vec: Array
    b_mat: Array

    def __init__(
        self,
        hidden_size: int,
        *,
        key: PRNGKeyArray,
        rank: int = 4,
    ) -> None:
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        self.a_diag = glorot(k1, (hidden_size,))
        self.p_vec = glorot(k2, (hidden_size, rank))
        self.q_vec = glorot(k3, (hidden_size, rank))
        self.b_mat = glorot(k4, (hidden_size, hidden_size))

    def forward(self, h: Array, x: Array) -> Array:
        """Performs a single forward pass."""
        a_dplr = jnp.diag(self.a_diag) + self.p_vec @ self.q_vec.T
        return a_dplr @ h + self.b_mat.T @ x


class DiscretizedDPLRSSMBlock(BaseSSMBlock):
    a_diag: Array
    p_vec: Array
    q_vec: Array
    b_mat: Array
    delta: float

    def __init__(
        self,
        hidden_size: int,
        *,
        key: PRNGKeyArray,
        rank: int = 4,
        delta: float = 1.0,
    ) -> None:
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.a_diag = glorot(k1, (hidden_size,))
        self.p_vec = glorot(k2, (hidden_size, rank))
        self.q_vec = glorot(k3, (hidden_size, rank))
        self.b_mat = glorot(k4, (hidden_size, hidden_size))
        self.delta = delta  # keeping constant since can be learned elsewhere

    def forward(self, h: Array, x: Array) -> Array:
        # A_dplr = diag(a) + P @ Q^T
        A_dplr = jnp.diag(self.a_diag) + self.p_vec @ self.q_vec.T

        # Discretized A and B
        A_disc = jnp.eye(h.shape[0]) + self.delta * A_dplr
        B_disc = self.delta * self.b_mat

        return A_disc @ h + B_disc.T @ x


class GatedSSMBlock(BaseSSMBlock):
    a_diag: Array
    p_vec: Array
    q_vec: Array
    b_mat: Array
    b_gate_proj: eqx.nn.Linear

    def __init__(
        self,
        hidden_size: int,
        *,
        key: PRNGKeyArray,
        rank: int = 4,
    ) -> None:
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        self.a_diag = glorot(k1, (hidden_size,))
        self.p_vec = glorot(k2, (hidden_size, rank))
        self.q_vec = glorot(k3, (hidden_size, rank))
        self.b_mat = glorot(k4, (hidden_size, hidden_size))
        self.b_gate_proj = eqx.nn.Linear(hidden_size, hidden_size, key=k5)

    def forward(self, h: Array, x: Array) -> Array:
        """Performs a single forward pass."""
        a_dplr = jnp.diag(self.a_diag) + self.p_vec @ self.q_vec.T

        # Doing gating in the output space saves memory.
        gate = jax.nn.sigmoid(self.b_gate_proj(x))
        b_proj = self.b_mat.T @ x
        return a_dplr @ h + gate * b_proj


class SSM(eqx.Module):
    input_proj: eqx.nn.Linear
    output_proj: eqx.nn.Linear
    ssm_blocks: list[BaseSSMBlock]
    num_layers: int = eqx.static_field()
    hidden_size: int = eqx.static_field()
    skip_connections: bool = eqx.static_field()

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        num_layers: int,
        block_type: SSMBlockType = "dplr",
        skip_connections: bool = False,
        discretize: bool = False,
        *,
        key: PRNGKeyArray,
    ) -> None:
        input_key, output_key, ssm_key = jax.random.split(key, 3)
        ssm_block_keys = jax.random.split(ssm_key, num_layers)
        self.input_proj = eqx.nn.Linear(input_size, hidden_size, key=input_key)
        self.output_proj = eqx.nn.Linear(hidden_size, output_size, key=output_key)

        def get_block(key: PRNGKeyArray) -> BaseSSMBlock:
            match block_type:
                case "diagonal":
                    return DiagSSMBlock(hidden_size, key=key)
                case "discrete_diagonal":
                    return DiscretizedDiagSSMBlock(hidden_size, key=key)
                case "full_rank":
                    if discretize:
                        raise ValueError("Full rank blocks do not support discretization due to instability.")
                    return SSMBlock(hidden_size, key=key)
                case "dplr":
                    return DPLRSSMBlock(hidden_size, key=key, rank=NUM_JOINTS)
                case "discrete_dplr":
                    return DiscretizedDPLRSSMBlock(hidden_size, key=key, rank=NUM_JOINTS)
                case "gated":
                    return GatedSSMBlock(hidden_size, key=key, rank=NUM_JOINTS)
                case _:
                    raise ValueError(f"Unknown block type: {block_type}")

        self.ssm_blocks = [get_block(ssm_block_keys[i]) for i in range(num_layers)]
        self.skip_connections = skip_connections
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def __call__(self, hs: Array, x: Array) -> tuple[Array, Array]:
        """Performs a single forward pass."""
        x = self.input_proj(x)
        new_hs = []
        for i, block in enumerate(self.ssm_blocks):
            h = block.forward(hs[i], x)
            new_hs.append(h)

            # We apply non-linearities in the vertical direction.
            xh = jax.nn.gelu(h)
            x = xh + x if self.skip_connections else xh
        y = self.output_proj(x)
        new_hs = jnp.stack(new_hs, axis=0)
        return new_hs, y


class DefaultHumanoidSSMActor(eqx.Module):
    """SSM-based actor for the walking task."""

    ssm: SSM
    num_inputs: int = eqx.static_field()
    num_outputs: int = eqx.static_field()
    min_std: float = eqx.static_field()
    max_std: float = eqx.static_field()
    var_scale: float = eqx.static_field()
    num_mixtures: int = eqx.static_field()

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        num_inputs: int,
        num_outputs: int,
        min_std: float,
        max_std: float,
        var_scale: float,
        hidden_size: int,
        depth: int,
        block_type: SSMBlockType = "dplr",
        discretize: bool = False,
        num_mixtures: int = 5,
    ) -> None:
        self.ssm = SSM(
            input_size=num_inputs,
            output_size=num_outputs * 3 * num_mixtures,
            hidden_size=hidden_size,
            num_layers=depth,
            block_type=block_type,
            discretize=discretize,
            key=key,
        )

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.min_std = min_std
        self.max_std = max_std
        self.var_scale = var_scale
        self.num_mixtures = num_mixtures

    def forward(self, obs_n: Array, carry: Array) -> tuple[distrax.Distribution, Array]:
        new_hs, out_n = self.ssm(carry, obs_n)

        # Splits the predictions into means, standard deviations, and logits.
        slice_len = NUM_JOINTS * self.num_mixtures
        mean_nm = out_n[:slice_len].reshape(NUM_JOINTS, self.num_mixtures)
        std_nm = out_n[slice_len : slice_len * 2].reshape(NUM_JOINTS, self.num_mixtures)
        logits_nm = out_n[slice_len * 2 :].reshape(NUM_JOINTS, self.num_mixtures)

        # Softplus and clip to ensure positive standard deviations.
        std_nm = jnp.clip((jax.nn.softplus(std_nm) + self.min_std) * self.var_scale, max=self.max_std)

        # Using mixture of gaussians to encourage exploration at the start.
        dist_n = distrax.MixtureSameFamily(
            mixture_distribution=distrax.Categorical(logits=logits_nm),
            components_distribution=distrax.Normal(mean_nm, std_nm),
        )

        return dist_n, new_hs


class DefaultHumanoidSSMCritic(eqx.Module):
    """SSM-based critic for the walking task."""

    ssm: SSM

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        num_inputs: int,
        hidden_size: int,
        depth: int,
        block_type: SSMBlockType = "dplr",
        discretize: bool = False,
    ) -> None:
        num_outputs = 1

        # Create SSM layer
        self.ssm = SSM(
            input_size=num_inputs,
            output_size=num_outputs,
            hidden_size=hidden_size,
            num_layers=depth,
            block_type=block_type,
            discretize=discretize,
            key=key,
        )

    def forward(self, obs_n: Array, carry: Array) -> tuple[Array, Array]:
        new_hs, out_n = self.ssm(carry, obs_n)
        return out_n, new_hs


class DefaultHumanoidSSMModel(eqx.Module):
    actor: DefaultHumanoidSSMActor
    critic: DefaultHumanoidSSMCritic

    def __init__(
        self,
        key: PRNGKeyArray,
        min_std: float,
        max_std: float,
        num_actor_inputs: int,
        num_critic_inputs: int,
        num_joints: int,
        hidden_size: int,
        depth: int,
        block_type: SSMBlockType = "dplr",
        discretize: bool = False,
        num_mixtures: int = 5,
    ) -> None:
        self.actor = DefaultHumanoidSSMActor(
            key,
            num_inputs=num_actor_inputs,
            num_outputs=num_joints,
            min_std=min_std,
            max_std=max_std,
            var_scale=0.5,
            hidden_size=hidden_size,
            depth=depth,
            block_type=block_type,
            discretize=discretize,
            num_mixtures=num_mixtures,
        )
        self.critic = DefaultHumanoidSSMCritic(
            key,
            num_inputs=num_critic_inputs,
            hidden_size=hidden_size,
            depth=depth,
            block_type=block_type,
            discretize=discretize,
        )


@dataclass
class HumanoidWalkingSSMTaskConfig(HumanoidWalkingTaskConfig):
    block_type: str = xax.field(
        value="dplr",
        help="The type of SSM block to use.",
    )
    discretize: bool = xax.field(
        value=False,
        help="Whether to discretize the SSM blocks.",
    )
    hidden_size: int = xax.field(
        value=128,
        help="The hidden size for the SSM.",
    )
    depth: int = xax.field(
        value=5,
        help="The number of SSM layers.",
    )
    num_mixtures: int = xax.field(
        value=5,
        help="The number of mixtures for the actor.",
    )


Config = TypeVar("Config", bound=HumanoidWalkingSSMTaskConfig)


class HumanoidWalkingSSMTask(HumanoidWalkingTask[Config], Generic[Config]):
    """SSM-based task for the walking task."""

    @property
    def actor_num_inputs(self) -> int:
        """The number of inputs to the actor network.

        The inputs are:
        - 2 (timestep sin/cos)
        - NUM_JOINTS (positions)
        - NUM_JOINTS (velocities)
        - 3 (imu_acc)
        - 3 (imu_gyro)
        - 4 (base_quat)
        - 3 (lin_vel_obs)
        - 3 (ang_vel_obs)
        """
        return 2 + NUM_JOINTS + NUM_JOINTS + 3 + 3 + 4 + 3 + 3

    @property
    def critic_num_inputs(self) -> int:
        """The number of inputs to the critic network.

        The inputs are:
        - 2 (timestep sin/cos)
        - NUM_JOINTS (positions)
        - NUM_JOINTS (velocities)
        - 160 (com_inertia)
        - 96 (com_vel)
        - 3 (imu_acc)
        - 3 (imu_gyro)
        - NUM_JOINTS (actuator_force)
        - 3 (base_pos)
        - 4 (base_quat)
        - 3 (lin_vel_obs)
        - 3 (ang_vel_obs)
        """
        return 2 + NUM_JOINTS + NUM_JOINTS + 160 + 96 + 3 + 3 + NUM_JOINTS + 3 + 4 + 3 + 3

    def get_model(self, key: PRNGKeyArray) -> DefaultHumanoidSSMModel:
        valid_types = get_args(SSMBlockType)
        if self.config.block_type not in valid_types:
            raise ValueError(f"Invalid block_type: {self.config.block_type}. Must be one of {valid_types}")

        block_type = cast(SSMBlockType, self.config.block_type)

        return DefaultHumanoidSSMModel(
            key,
            num_actor_inputs=self.actor_num_inputs,
            num_critic_inputs=self.critic_num_inputs,
            num_joints=NUM_JOINTS,
            min_std=0.01,
            max_std=1.0,
            num_mixtures=self.config.num_mixtures,
            hidden_size=self.config.hidden_size,
            depth=self.config.depth,
            block_type=block_type,
            discretize=self.config.discretize,
        )

    def get_initial_model_carry(self, rng: PRNGKeyArray) -> tuple[Array, Array]:
        """Gets initial carry state for the model.

        Args:
            rng: Random number generator key.

        Returns:
            Initial RNN hidden states for actor and critic.
        """
        return (
            jnp.zeros(shape=(self.config.depth, self.config.hidden_size)),
            jnp.zeros(shape=(self.config.depth, self.config.hidden_size)),
        )

    def run_actor(
        self,
        model: DefaultHumanoidSSMActor,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        carry: Array,
    ) -> tuple[distrax.Distribution, Array]:
        """Runs the actor network to get an action distribution.

        Args:
            model: The actor network.
            observations: Current observations.
            commands: Current commands.
            carry: RNN hidden state.

        Returns:
            A tuple of action distribution and next RNN hidden state.
        """
        timestep_1 = observations["timestep_observation"]
        dh_joint_pos_j = observations["joint_position_observation"]
        dh_joint_vel_j = observations["joint_velocity_observation"]
        imu_acc_3 = observations["sensor_observation_imu_acc"]
        imu_gyro_3 = observations["sensor_observation_imu_gyro"]
        base_quat_4 = observations["base_orientation_observation"]
        lin_vel_obs_3 = observations["base_linear_velocity_observation"]
        ang_vel_obs_3 = observations["base_angular_velocity_observation"]

        obs_n = jnp.concatenate(
            [
                jnp.cos(timestep_1),  # 1
                jnp.sin(timestep_1),  # 1
                dh_joint_pos_j,  # NUM_JOINTS
                dh_joint_vel_j / 10.0,  # NUM_JOINTS
                imu_acc_3 / 50.0,  # 3
                imu_gyro_3 / 3.0,  # 3
                base_quat_4,  # 4
                lin_vel_obs_3,  # 3
                ang_vel_obs_3,  # 3
            ],
            axis=-1,
        )

        return model.forward(obs_n, carry)

    def run_critic(
        self,
        model: DefaultHumanoidSSMCritic,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        carry: Array,
    ) -> tuple[Array, Array]:
        """Runs the critic network to get state values.

        Args:
            model: The critic network.
            observations: Current observations.
            commands: Current commands.
            carry: RNN hidden state.

        Returns:
            A tuple of state values and next RNN hidden state.
        """
        timestep_1 = observations["timestep_observation"]
        dh_joint_pos_j = observations["joint_position_observation"]
        dh_joint_vel_j = observations["joint_velocity_observation"]
        com_inertia_n = observations["center_of_mass_inertia_observation"]
        com_vel_n = observations["center_of_mass_velocity_observation"]
        imu_acc_3 = observations["sensor_observation_imu_acc"]
        imu_gyro_3 = observations["sensor_observation_imu_gyro"]
        act_frc_n = observations["actuator_force_observation"]
        base_pos_3 = observations["base_position_observation"]
        base_quat_4 = observations["base_orientation_observation"]
        lin_vel_obs_3 = observations["base_linear_velocity_observation"]
        ang_vel_obs_3 = observations["base_angular_velocity_observation"]

        obs_n = jnp.concatenate(
            [
                jnp.cos(timestep_1),  # 1
                jnp.sin(timestep_1),  # 1
                dh_joint_pos_j,  # NUM_JOINTS
                dh_joint_vel_j / 10.0,  # NUM_JOINTS
                com_inertia_n,  # 160
                com_vel_n,  # 96
                imu_acc_3 / 50.0,  # 3
                imu_gyro_3 / 3.0,  # 3
                act_frc_n / 100.0,  # NUM_JOINTS
                base_pos_3,  # 3
                base_quat_4,  # 4
                lin_vel_obs_3,  # 3
                ang_vel_obs_3,  # 3
            ],
            axis=-1,
        )

        return model.forward(obs_n, carry)

    def get_ppo_variables(
        self,
        model: DefaultHumanoidSSMModel,
        trajectory: ksim.Trajectory,
        model_carry: tuple[Array, Array],
        rng: PRNGKeyArray,
    ) -> tuple[ksim.PPOVariables, tuple[Array, Array]]:
        """Computes PPO variables for training.

        Args:
            model: The actor-critic model.
            trajectory: The trajectory to compute variables for.
            model_carry: Model carry state containing RNN hidden states.
            rng: Random number generator key.

        Returns:
            A tuple of PPO variables and the next model carry state.
        """

        def scan_fn(
            actor_critic_carry: tuple[Array, Array], transition: ksim.Trajectory
        ) -> tuple[tuple[Array, Array], ksim.PPOVariables]:
            actor_carry, critic_carry = actor_critic_carry
            actor_dist, next_actor_carry = self.run_actor(
                model=model.actor,
                observations=transition.obs,
                commands=transition.command,
                carry=actor_carry,
            )
            log_probs = actor_dist.log_prob(transition.action)
            assert isinstance(log_probs, Array)
            value, next_critic_carry = self.run_critic(
                model=model.critic,
                observations=transition.obs,
                commands=transition.command,
                carry=critic_carry,
            )

            transition_ppo_variables = ksim.PPOVariables(
                log_probs=log_probs,
                values=value.squeeze(-1),
            )

            initial_carry = self.get_initial_model_carry(rng)
            next_carry = jax.tree.map(
                lambda x, y: jnp.where(transition.done, x, y), initial_carry, (next_actor_carry, next_critic_carry)
            )

            return next_carry, transition_ppo_variables

        next_model_carry, ppo_variables = jax.lax.scan(scan_fn, model_carry, trajectory)

        return ppo_variables, next_model_carry

    def sample_action(
        self,
        model: DefaultHumanoidSSMModel,
        model_carry: tuple[Array, Array],
        physics_model: ksim.PhysicsModel,
        physics_state: ksim.PhysicsState,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        rng: PRNGKeyArray,
        argmax: bool,
    ) -> ksim.Action:
        """Samples an action from the policy.

        Args:
            model: The actor-critic model.
            model_carry: RNN hidden states.
            physics_model: The physics model.
            physics_state: The current physics state.
            observations: Current observations.
            commands: Current commands.
            rng: Random number generator key.
            argmax: Whether to take the mode of the distribution instead of sampling.

        Returns:
            A sampled action with its carry state and auxiliary outputs.
        """
        actor_carry_in, critic_carry_in = model_carry

        # Runs the actor model to get the action distribution.
        action_dist_j, actor_carry = self.run_actor(
            model=model.actor,
            observations=observations,
            commands=commands,
            carry=actor_carry_in,
        )

        action_j = action_dist_j.mode() if argmax else action_dist_j.sample(seed=rng)

        return ksim.Action(
            action=action_j,
            carry=(actor_carry, critic_carry_in),
            aux_outputs=None,
        )


if __name__ == "__main__":
    # To run training, use the following command:
    #   python -m iclrobot.dh_walking.dh_walking_ssm
    # To visualize the environment, use the following command:
    #   python -m iclrobot.dh_walking.dh_walking_ssm run_environment=True
    # On MacOS or other devices with less memory, you can change the number
    # of environments and batch size to reduce memory usage. Here's an example
    # from the command line:
    #   python -m iclrobot.dh_walking.dh_walking_ssm num_envs=8 rollouts_per_batch=4
    HumanoidWalkingSSMTask.launch(
        HumanoidWalkingSSMTaskConfig(
            # Model parameters.
            hidden_size=128,
            depth=5,
            block_type="discrete_diagonal",
            discretize=False,
            # Training parameters.
            num_envs=2048,
            batch_size=256,
            num_passes=4,
            epochs_per_log_step=1,
            rollout_length_seconds=5.0,
            # Simulation parameters.
            dt=0.005,
            ctrl_dt=0.02,
            max_action_latency=0.0,
            min_action_latency=0.0,
            randomize=True,
        ),
    )

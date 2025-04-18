# mypy: disable-error-code="override"
"""A simple RNN model for the humanoid walking task."""

from dataclasses import dataclass
from typing import Generic, TypeVar

import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
import ksim
import xax
from jaxtyping import Array, PRNGKeyArray

from iclrobot.dh_walking.dh_walking import (
    NUM_JOINTS,
    HumanoidWalkingTask,
    HumanoidWalkingTaskConfig,
)


class DefaultHumanoidRNNActor(eqx.Module):
    """RNN-based actor for the walking task.

    Attributes:
        input_proj: Linear projection for input features.
        rnns: Tuple of GRU cells.
        output_proj: Linear projection for output.
        min_std: Minimum standard deviation.
        max_std: Maximum standard deviation.
        var_scale: Scale factor for the variance.
    """

    input_proj: eqx.nn.Linear
    rnns: tuple[eqx.nn.GRUCell, ...]
    output_proj: eqx.nn.Linear
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
        num_mixtures: int,
    ) -> None:
        """Initialize the actor network.

        Args:
            key: Random key for initialization.
            num_inputs: Number of input features.
            num_outputs: Number of output dimensions.
            min_std: Minimum standard deviation.
            max_std: Maximum standard deviation.
            var_scale: Scale factor for the variance.
            hidden_size: Number of hidden units.
            depth: Number of RNN layers.
            num_mixtures: Number of mixtures.
        """
        # Project input to hidden size
        key, input_proj_key = jax.random.split(key)
        self.input_proj = eqx.nn.Linear(
            in_features=num_inputs,
            out_features=hidden_size,
            key=input_proj_key,
        )

        # Create RNN layers
        key, rnn_key = jax.random.split(key)
        self.rnns = tuple(
            [
                eqx.nn.GRUCell(
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    key=rnn_key,
                )
                for _ in range(depth)
            ]
        )

        # Project to output
        self.output_proj = eqx.nn.Linear(
            in_features=hidden_size,
            out_features=num_outputs * 3 * num_mixtures,
            key=key,
        )

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.min_std = min_std
        self.max_std = max_std
        self.var_scale = var_scale
        self.num_mixtures = num_mixtures

    def forward(self, obs_n: Array, carry: Array) -> tuple[distrax.Distribution, Array]:
        """Forward pass of the actor network.

        Args:
            obs_n: Batch of observations.
            carry: RNN hidden state.

        Returns:
            A tuple of action distribution and next RNN hidden state.
        """
        x_n = self.input_proj(obs_n)
        out_carries = []
        for i, rnn in enumerate(self.rnns):
            x_n = rnn(x_n, carry[i])
            out_carries.append(x_n)
        out_n = self.output_proj(x_n)

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

        return dist_n, jnp.stack(out_carries, axis=0)


class DefaultHumanoidRNNCritic(eqx.Module):
    """RNN-based critic for the walking task.

    Attributes:
        input_proj: Linear projection for input features.
        rnns: Tuple of GRU cells.
        output_proj: Linear projection for output.
    """

    input_proj: eqx.nn.Linear
    rnns: tuple[eqx.nn.GRUCell, ...]
    output_proj: eqx.nn.Linear

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        num_inputs: int,
        hidden_size: int,
        depth: int,
    ) -> None:
        """Initialize the critic network.

        Args:
            key: Random key for initialization.
            num_inputs: Number of input features.
            hidden_size: Number of hidden units.
            depth: Number of RNN layers.
        """
        num_outputs = 1

        # Project input to hidden size
        key, input_proj_key = jax.random.split(key)
        self.input_proj = eqx.nn.Linear(
            in_features=num_inputs,
            out_features=hidden_size,
            key=input_proj_key,
        )

        # Create RNN layers
        key, rnn_key = jax.random.split(key)
        self.rnns = tuple(
            [
                eqx.nn.GRUCell(
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    key=rnn_key,
                )
                for _ in range(depth)
            ]
        )

        # Project to output
        self.output_proj = eqx.nn.Linear(
            in_features=hidden_size,
            out_features=num_outputs,
            key=key,
        )

    def forward(self, obs_n: Array, carry: Array) -> tuple[Array, Array]:
        """Forward pass of the critic network.

        Args:
            obs_n: Batch of observations.
            carry: RNN hidden state.

        Returns:
            A tuple of state values and next RNN hidden state.
        """
        x_n = self.input_proj(obs_n)
        out_carries = []
        for i, rnn in enumerate(self.rnns):
            x_n = rnn(x_n, carry[i])
            out_carries.append(x_n)
        out_n = self.output_proj(x_n)

        return out_n, jnp.stack(out_carries, axis=0)


class DefaultHumanoidRNNModel(eqx.Module):
    """Combined actor-critic model for the humanoid walking task.

    Attributes:
        actor: Policy network that outputs action distributions.
        critic: Value network that estimates state values.
    """

    actor: DefaultHumanoidRNNActor
    critic: DefaultHumanoidRNNCritic

    def __init__(
        self,
        key: PRNGKeyArray,
        min_std: float,
        max_std: float,
        num_actor_inputs: int,
        num_actor_mixtures: int,
        num_critic_inputs: int,
        num_joints: int,
        hidden_size: int,
        depth: int,
    ) -> None:
        """Initialize the actor-critic model.

        Args:
            key: Random key for initialization.
            min_std: Minimum standard deviation.
            max_std: Maximum standard deviation.
            num_actor_inputs: Number of input features for the actor network.
            num_actor_mixtures: Number of mixtures for the actor network.
            num_critic_inputs: Number of input features for the critic network.
            num_joints: Number of joints to control.
            hidden_size: Number of hidden units.
            depth: Number of RNN layers.
        """
        self.actor = DefaultHumanoidRNNActor(
            key,
            num_inputs=num_actor_inputs,
            num_outputs=num_joints,
            min_std=min_std,
            max_std=max_std,
            var_scale=0.5,
            num_mixtures=num_actor_mixtures,
            hidden_size=hidden_size,
            depth=depth,
        )
        self.critic = DefaultHumanoidRNNCritic(
            key,
            num_inputs=num_critic_inputs,
            hidden_size=hidden_size,
            depth=depth,
        )


@dataclass
class HumanoidWalkingRNNTaskConfig(HumanoidWalkingTaskConfig):
    """Configuration for training a walking policy using an RNN."""

    hidden_size: int = xax.field(
        value=128,
        help="The hidden size for the RNN.",
    )
    depth: int = xax.field(
        value=5,
        help="The number of RNN layers.",
    )
    num_mixtures: int = xax.field(
        value=5,
        help="The number of mixtures for the actor.",
    )


Config = TypeVar("Config", bound=HumanoidWalkingRNNTaskConfig)


class HumanoidWalkingRNNTask(HumanoidWalkingTask[Config], Generic[Config]):
    """A task for training a walking policy using an RNN."""

    @property
    def actor_num_inputs(self) -> int:
        """The number of inputs to the actor network.

        The inputs are:
        - 2 (timestep sin/cos)
        - NUM_JOINTS (positions)
        - NUM_JOINTS (velocities)
        - 3 (imu_acc)
        - 4 (base_quat)
        - NUM_JOINTS * history_length (action history)
        """
        return 2 + NUM_JOINTS + NUM_JOINTS + 3 + 4

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

    def get_model(self, key: PRNGKeyArray) -> DefaultHumanoidRNNModel:
        """Creates the actor-critic model."""
        return DefaultHumanoidRNNModel(
            key,
            num_actor_inputs=self.actor_num_inputs,
            num_critic_inputs=self.critic_num_inputs,
            num_joints=NUM_JOINTS,
            min_std=0.01,
            max_std=1.0,
            num_actor_mixtures=self.config.num_mixtures,
            hidden_size=self.config.hidden_size,
            depth=self.config.depth,
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
        model: DefaultHumanoidRNNActor,
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
        base_quat_4 = observations["base_orientation_observation"]

        obs_n = jnp.concatenate(
            [
                jnp.cos(timestep_1),  # 1
                jnp.sin(timestep_1),  # 1
                dh_joint_pos_j,  # NUM_JOINTS
                dh_joint_vel_j / 10.0,  # NUM_JOINTS
                imu_acc_3 / 50.0,  # 3
                base_quat_4,  # 4
            ],
            axis=-1,
        )

        return model.forward(obs_n, carry)

    def run_critic(
        self,
        model: DefaultHumanoidRNNCritic,
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
        model: DefaultHumanoidRNNModel,
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
        model: DefaultHumanoidRNNModel,
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
    #   python -m iclrobot.dh_walking.dh_walking_rnn
    # To visualize the environment, use the following command:
    #   python -m iclrobot.dh_walking.dh_walking_rnn run_environment=True
    # On MacOS or other devices with less memory, you can change the number
    # of environments and batch size to reduce memory usage. Here's an example
    # from the command line:
    #   python -m iclrobot.dh_walking.dh_walking_rnn num_envs=8 rollouts_per_batch=4
    HumanoidWalkingRNNTask.launch(
        HumanoidWalkingRNNTaskConfig(
            # Training parameters.
            num_envs=2048,
            batch_size=256,
            num_passes=4,
            epochs_per_log_step=1,
            rollout_length_seconds=10.0,
            # Simulation parameters.
            dt=0.005,
            ctrl_dt=0.02,
            max_action_latency=0.0,
            min_action_latency=0.0,
            randomize_physics=True,
        ),
    )

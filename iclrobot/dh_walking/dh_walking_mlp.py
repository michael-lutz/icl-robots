"""A simple MLP model for the humanoid walking task."""

from dataclasses import dataclass
from typing import Generic, TypeVar
import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
import ksim
import xax
from jaxtyping import Array, PRNGKeyArray

from iclrobot.dh_walking.dh_walking import NUM_JOINTS, HumanoidWalkingTask, HumanoidWalkingTaskConfig


class DefaultHumanoidActor(eqx.Module):
    """Actor network for the walking task that outputs a mixture of Gaussians policy.

    Attributes:
        mlp: Multi-layer perceptron network.
        min_std: Minimum standard deviation for the Gaussian components.
        max_std: Maximum standard deviation for the Gaussian components.
        var_scale: Scale factor for the variance.
        num_mixtures: Number of Gaussian components in the mixture.
    """

    mlp: eqx.nn.MLP
    min_std: float = eqx.static_field()
    max_std: float = eqx.static_field()
    var_scale: float = eqx.static_field()
    num_mixtures: int = eqx.static_field()

    def __init__(
        self,
        key: PRNGKeyArray,
        num_inputs: int,
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
            min_std: Minimum standard deviation for the Gaussian components.
            max_std: Maximum standard deviation for the Gaussian components.
            var_scale: Scale factor for the variance.
            hidden_size: Number of hidden units in each layer.
            depth: Number of hidden layers.
            num_mixtures: Number of Gaussian components in the mixture.
        """
        num_outputs = NUM_JOINTS

        self.mlp = eqx.nn.MLP(
            in_size=num_inputs,
            out_size=num_outputs * 3 * num_mixtures,
            width_size=hidden_size,
            depth=depth,
            key=key,
            activation=jax.nn.relu,
        )
        self.min_std = min_std
        self.max_std = max_std
        self.var_scale = var_scale
        self.num_mixtures = num_mixtures

    def forward(self, obs_n: Array) -> distrax.Distribution:
        """Forward pass of the actor network.

        Args:
            obs_n: Batch of observations.

        Returns:
            A mixture of Gaussians distribution over actions.
        """
        prediction_n = self.mlp(obs_n)

        # Splits the predictions into means, standard deviations, and logits.
        slice_len = NUM_JOINTS * self.num_mixtures
        mean_nm = prediction_n[:slice_len].reshape(NUM_JOINTS, self.num_mixtures)
        std_nm = prediction_n[slice_len : slice_len * 2].reshape(NUM_JOINTS, self.num_mixtures)
        logits_nm = prediction_n[slice_len * 2 :].reshape(NUM_JOINTS, self.num_mixtures)

        # Softplus and clip to ensure positive standard deviations.
        std_nm = jnp.clip((jax.nn.softplus(std_nm) + self.min_std) * self.var_scale, max=self.max_std)

        # Using mixture of gaussians to encourage exploration at the start.
        dist_n = distrax.MixtureSameFamily(
            mixture_distribution=distrax.Categorical(logits=logits_nm),
            components_distribution=distrax.Normal(mean_nm, std_nm),
        )

        return dist_n


class DefaultHumanoidCritic(eqx.Module):
    """Critic network for the walking task that estimates state values.

    Attributes:
        mlp: Multi-layer perceptron network for value estimation.
    """

    mlp: eqx.nn.MLP

    def __init__(
        self,
        key: PRNGKeyArray,
        num_inputs: int,
        hidden_size: int,
        depth: int,
    ) -> None:
        """Initialize the critic network.

        Args:
            key: Random key for initialization.
            hidden_size: Number of hidden units in each layer.
            depth: Number of hidden layers.
        """
        num_outputs = 1

        self.mlp = eqx.nn.MLP(
            in_size=num_inputs,
            out_size=num_outputs,
            width_size=hidden_size,
            depth=depth,
            key=key,
            activation=jax.nn.relu,
        )

    def forward(self, obs_n: Array) -> Array:
        """Forward pass of the critic network.

        Args:
            obs_n: Batch of observations.

        Returns:
            Estimated state values.
        """
        return self.mlp(obs_n)


class DefaultHumanoidModel(eqx.Module):
    """Combined actor-critic model for the humanoid walking task.

    Attributes:
        actor: Policy network that outputs action distributions.
        critic: Value network that estimates state values.
    """

    actor: DefaultHumanoidActor
    critic: DefaultHumanoidCritic

    def __init__(
        self,
        key: PRNGKeyArray,
        actor_num_inputs: int,
        critic_num_inputs: int,
        hidden_size: int,
        depth: int,
        num_mixtures: int,
    ) -> None:
        """Initialize the actor-critic model.

        Args:
            key: Random key for initialization.
            hidden_size: Number of hidden units in each layer of both networks.
            depth: Number of hidden layers in both networks.
            num_mixtures: Number of Gaussian components in the actor's mixture policy.
        """
        self.actor = DefaultHumanoidActor(
            key,
            num_inputs=actor_num_inputs,
            min_std=0.01,
            max_std=1.0,
            var_scale=0.5,
            hidden_size=hidden_size,
            depth=depth,
            num_mixtures=num_mixtures,
        )
        self.critic = DefaultHumanoidCritic(
            key,
            num_inputs=critic_num_inputs,
            hidden_size=hidden_size,
            depth=depth,
        )


@dataclass
class HumanoidWalkingMLPTaskConfig(HumanoidWalkingTaskConfig):
    """A simple config for training a walking policy for the default humanoid."""

    hidden_size: int = xax.field(
        value=128,
        help="The hidden size for the MLPs.",
    )
    depth: int = xax.field(
        value=5,
        help="The depth for the MLPs.",
    )
    num_mixtures: int = xax.field(
        value=5,
        help="The number of mixtures for the actor.",
    )


Config = TypeVar("Config", bound=HumanoidWalkingMLPTaskConfig)


class HumanoidWalkingMLPTask(HumanoidWalkingTask[Config], Generic[Config]):
    """A simple task for training a walking policy for the default humanoid."""

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
        res = 2 + NUM_JOINTS + NUM_JOINTS + 3 + 3 + 4 + 3 + 3
        return res

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
        res = 2 + NUM_JOINTS + NUM_JOINTS + 160 + 96 + 3 + 3 + NUM_JOINTS + 3 + 4 + 3 + 3
        return res

    def get_model(self, key: PRNGKeyArray) -> DefaultHumanoidModel:
        """Creates the actor-critic model."""
        return DefaultHumanoidModel(
            key,
            actor_num_inputs=self.actor_num_inputs,
            critic_num_inputs=self.critic_num_inputs,
            hidden_size=self.config.hidden_size,
            depth=self.config.depth,
            num_mixtures=self.config.num_mixtures,
        )

    def get_initial_model_carry(self, rng: PRNGKeyArray) -> None:
        """Gets initial carry state for the model."""
        return None

    def run_actor(
        self,
        model: DefaultHumanoidActor,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        model_carry: None,
    ) -> distrax.Distribution:
        """Runs the actor network to get an action distribution.

        Args:
            model: The actor network.
            observations: Current observations.
            commands: Current commands.
            model_carry: Unused model carry state.

        Returns:
            A distribution over actions.
        """
        timestep_1 = observations["timestep_observation"]
        dh_joint_pos_j = observations["joint_position_observation"]
        dh_joint_vel_j = observations["joint_velocity_observation"]
        imu_acc_3 = observations["sensor_observation_imu_acc"]
        lin_vel_obs_3 = observations["base_linear_velocity_observation"]
        ang_vel_obs_3 = observations["base_angular_velocity_observation"]
        imu_gyro_3 = observations["sensor_observation_imu_gyro"]
        base_quat_4 = observations["base_orientation_observation"]

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

        return model.forward(obs_n)

    def run_critic(
        self,
        model: DefaultHumanoidCritic,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        model_carry: None,
    ) -> Array:
        """Runs the critic network to get state values.

        Args:
            model: The critic network.
            observations: Current observations.
            commands: Current commands.
            model_carry: Unused model carry state.

        Returns:
            Estimated state values.
        """
        timestep_1 = observations["timestep_observation"]
        dh_joint_pos_j = observations["joint_position_observation"]
        dh_joint_vel_j = observations["joint_velocity_observation"]
        com_inertia_n = observations["center_of_mass_inertia_observation"]
        com_vel_n = observations["center_of_mass_velocity_observation"]
        imu_acc_3 = observations["sensor_observation_imu_acc"]
        imu_gyro_3 = observations["sensor_observation_imu_gyro"]
        act_frc_obs_n = observations["actuator_force_observation"]
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
                act_frc_obs_n / 100.0,  # NUM_JOINTS
                base_pos_3,  # 3
                base_quat_4,  # 4
                lin_vel_obs_3,  # 3
                ang_vel_obs_3,  # 3
            ],
            axis=-1,
        )

        return model.forward(obs_n)

    def get_ppo_variables(
        self,
        model: DefaultHumanoidModel,
        trajectory: ksim.Trajectory,
        model_carry: None,
        rng: PRNGKeyArray,
    ) -> tuple[ksim.PPOVariables, None]:
        """Computes PPO variables for training.

        Args:
            model: The actor-critic model.
            trajectory: The trajectory to compute variables for.
            model_carry: Unused model carry state.
            rng: Random number generator key.

        Returns:
            A tuple of PPO variables and the next model carry state.
        """

        # Vectorize over the time dimensions.
        def get_log_prob(transition: ksim.Trajectory) -> Array:
            action_dist_tj = self.run_actor(model.actor, transition.obs, transition.command, None)
            log_probs_tj = action_dist_tj.log_prob(transition.action)
            assert isinstance(log_probs_tj, Array)
            return log_probs_tj

        log_probs_tj = jax.vmap(get_log_prob)(trajectory)
        assert isinstance(log_probs_tj, Array)

        # Vectorize over the time dimensions.
        values_tj = jax.vmap(self.run_critic, in_axes=(None, 0, 0, None))(
            model.critic, trajectory.obs, trajectory.command, None
        )
        ppo_variables = ksim.PPOVariables(
            log_probs=log_probs_tj,
            values=values_tj.squeeze(-1),
        )

        return ppo_variables, None

    def sample_action(
        self,
        model: DefaultHumanoidModel,
        model_carry: None,
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
            model_carry: Unused carry state.
            physics_model: The physics model.
            physics_state: The current physics state.
            observations: Current observations.
            commands: Current commands.
            rng: Random number generator key.
            argmax: Whether to take the mode of the distribution instead of sampling.

        Returns:
            A sampled action with its carry state and auxiliary outputs.
        """
        action_dist_j = self.run_actor(
            model=model.actor, observations=observations, commands=commands, model_carry=model_carry
        )
        action_j = action_dist_j.mode() if argmax else action_dist_j.sample(seed=rng)
        return ksim.Action(
            action=action_j,
            carry=None,
        )


if __name__ == "__main__":
    # To run training, use the following command:
    #   python -m iclrobot.dh_walking.dh_walking_mlp
    # To visualize the environment, use the following command:
    #   python -m iclrobot.dh_walking.dh_walking_mlp run_environment=True
    # On MacOS or other devices with less memory, you can change the number
    # of environments and batch size to reduce memory usage. Here's an example
    # from the command line:
    #   python -m iclrobot.dh_walking.dh_walking_mlp num_envs=8 rollouts_per_batch=4
    HumanoidWalkingMLPTask.launch(
        HumanoidWalkingMLPTaskConfig(
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
            randomize=False,
        ),
    )

# mypy: disable-error-code="override"
"""A simple Transformer model for the humanoid walking task."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

import chex
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


@jax.tree_util.register_dataclass
@dataclass
class HistoricalCarry:
    observations: xax.FrozenDict[str, Array]
    actions: Array


class MultiHeadAttention(eqx.Module):
    dim: int = eqx.static_field()
    num_heads: int = eqx.static_field()
    head_dim: int = eqx.static_field()
    w_qkv: eqx.nn.Linear

    def __init__(self, dim: int, num_heads: int, *, key: PRNGKeyArray):
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.w_qkv = eqx.nn.Linear(dim, dim * 3, key=key)

    def apply_attention(self, q: Array, k: Array, v: Array) -> Array:
        T_q, D = q.shape
        T_kv, D = k.shape
        chex.assert_shape(v, (T_kv, D))

        q = q.reshape(T_q, self.num_heads, self.head_dim).transpose(1, 0, 2)  # [H, T_q, D_h]
        k = k.reshape(T_kv, self.num_heads, self.head_dim).transpose(1, 0, 2)  # [H, T_kv, D_h]
        v = v.reshape(T_kv, self.num_heads, self.head_dim).transpose(1, 0, 2)  # [H, T_kv, D_h]

        attn_scores = q @ k.transpose(0, 2, 1) / jnp.sqrt(self.head_dim)  # [H, T_q, T_kv]
        attn_weights = jax.nn.softmax(attn_scores, axis=-1)

        res = attn_weights @ v  # [H, T_q, D_h]
        return res.transpose(1, 0, 2).reshape(T_q, D)  # [T_q, D]

    def forward_sequence(self, x_seq: Array) -> Array:
        T_seq = x_seq.shape[0]
        qkv_seq = jax.vmap(self.w_qkv)(x_seq)  # [T_seq, 3D]
        q_seq, k_seq, v_seq = jnp.split(qkv_seq, 3, axis=-1)

        mask = jnp.zeros((T_seq, T_seq), dtype=jnp.bool_)  # [T_seq, T_seq]
        attn_output = self.apply_attention(q_seq, k_seq, v_seq)  # [T_seq, D]

        res = x_seq + attn_output  # [T_seq, D]
        return res


class AttentionBlock(eqx.Module):
    dim: int = eqx.static_field()
    ln1: eqx.nn.LayerNorm
    ln2: eqx.nn.LayerNorm
    attn: MultiHeadAttention
    mlp: eqx.nn.MLP

    def __init__(self, dim: int, num_heads: int, key: PRNGKeyArray):
        self.dim = dim

        k1, k2 = jax.random.split(key, 2)
        self.ln1 = eqx.nn.LayerNorm(dim)
        self.ln2 = eqx.nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, key=k1)
        self.mlp = eqx.nn.MLP(dim, dim, width_size=dim, depth=1, key=k2, activation=jax.nn.gelu)

    def forward_sequence(self, x_seq: Array) -> Array:
        x_norm1 = jax.vmap(self.ln1)(x_seq)
        x_attn = self.attn.forward_sequence(x_norm1)
        x_res1 = x_seq + x_attn
        x_norm2 = jax.vmap(self.ln2)(x_res1)
        x_mlp = jax.vmap(self.mlp)(x_norm2)
        x_out = x_res1 + x_mlp
        return x_out


class Transformer(eqx.Module):
    hidden_size: int = eqx.static_field()
    cls_embedding: Array
    final_ln: eqx.nn.LayerNorm
    initial_ln: eqx.nn.LayerNorm
    learned_position_embedding: eqx.nn.Embedding
    context_length: int = eqx.static_field()
    blocks: list[AttentionBlock]

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        context_length: int,
        cls_init_scale: float,
        key: PRNGKeyArray,
    ):
        self.hidden_size = hidden_size
        self.context_length = context_length
        cls_key, learned_pos_key, block_key = jax.random.split(key, 3)
        block_keys = jax.random.split(block_key, num_layers)

        self.cls_embedding = jax.random.uniform(cls_key, (1, hidden_size)) * cls_init_scale
        self.learned_position_embedding = eqx.nn.Embedding(context_length + 1, hidden_size, key=learned_pos_key)
        self.blocks = [AttentionBlock(hidden_size, num_heads, key=block_keys[i]) for i in range(num_layers)]
        self.final_ln = eqx.nn.LayerNorm(hidden_size)
        self.initial_ln = eqx.nn.LayerNorm(hidden_size)

    def input_pos_embedding(self, seq_len: int, offset: int) -> Array:
        """Position embeddings added directly to the input sequence."""
        return jax.vmap(self.learned_position_embedding)(jnp.arange(offset, offset + seq_len))

    def forward_sequence(self, x_seq: Array) -> Array:
        chex.assert_shape(x_seq, (self.context_length, self.hidden_size))  # [T, D]

        # Concatenate cls_token to normalized sequence
        x_seq = jax.vmap(self.initial_ln)(x_seq)
        x_seq = jnp.concatenate([x_seq, self.cls_embedding], axis=0)
        x_seq += self.input_pos_embedding(x_seq.shape[0], offset=0)

        for block in self.blocks:
            x_seq = block.forward_sequence(x_seq)

        x_seq = jax.vmap(self.final_ln)(x_seq)
        return x_seq


class DefaultHumanoidTransformerActor(eqx.Module):
    """Transformer-based actor for the walking task."""

    obs_proj: eqx.nn.Linear
    act_proj: eqx.nn.Linear
    transformer: Transformer
    output_proj: eqx.nn.Linear
    num_obs: int = eqx.static_field()
    num_act: int = eqx.static_field()
    num_frames: int = eqx.static_field()
    min_std: float = eqx.static_field()
    max_std: float = eqx.static_field()
    var_scale: float = eqx.static_field()
    num_mixtures: int = eqx.static_field()

    def __init__(
        self,
        key: PRNGKeyArray,
        num_obs: int,
        num_act: int,
        min_std: float,
        max_std: float,
        var_scale: float,
        hidden_size: int,
        depth: int,
        num_heads: int,
        num_mixtures: int,
        num_frames: int,
        cls_init_scale: float,
    ) -> None:
        """Initialize the actor network."""
        # Project input to hidden size
        obs_key, act_key, out_key, transformer_key = jax.random.split(key, 4)
        self.obs_proj = eqx.nn.Linear(
            in_features=num_obs,
            out_features=hidden_size,
            key=obs_key,
        )
        self.act_proj = eqx.nn.Linear(
            in_features=num_act,
            out_features=hidden_size,
            key=act_key,
        )

        # Create Transformer layers
        self.transformer = Transformer(
            hidden_size=hidden_size,
            num_layers=depth,
            num_heads=num_heads,
            context_length=num_frames * 2,  # obs + act
            cls_init_scale=cls_init_scale,
            key=transformer_key,
        )

        # Project to output
        self.output_proj = eqx.nn.Linear(
            in_features=hidden_size,
            out_features=num_act * 3 * num_mixtures,
            key=out_key,
        )

        self.num_obs = num_obs
        self.num_act = num_act
        self.min_std = min_std
        self.max_std = max_std
        self.var_scale = var_scale
        self.num_mixtures = num_mixtures
        self.num_frames = num_frames

    def forward(self, obs_n: Array, act_n: Array) -> distrax.Distribution:
        """Forward pass of the actor network.

        Args:
            obs_n: Observation sequence [T, obs_dim]
            act_n: Action sequence [T, act_dim]
        """
        chex.assert_shape(obs_n, (self.num_frames, self.num_obs))
        chex.assert_shape(act_n, (self.num_frames, self.num_act))

        # Create and interleave obs and act embeddings.
        obs_emb = jax.vmap(self.obs_proj)(obs_n)  # [T, hidden_size]
        act_emb = jax.vmap(self.act_proj)(act_n)  # [T, hidden_size]
        interleaved_indices = jnp.arange(2 * obs_emb.shape[0]).reshape(2, -1).T.ravel()  # [0,L,1,L+1,2,L+2...]
        x_emb = jnp.concatenate([act_emb, obs_emb])[interleaved_indices]

        # Getting the CLS and predicting output.
        cls_emb = self.transformer.forward_sequence(x_emb)[-1]
        out_n = self.output_proj(cls_emb)

        # Splits the predictions into means, standard deviations, and logits.
        # mean_n = out_n[:NUM_JOINTS]
        # std_n = out_n[NUM_JOINTS:]
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

        return dist_n


class DefaultHumanoidTransformerCritic(eqx.Module):
    """Transformer-based critic for the walking task."""

    obs_proj: eqx.nn.Linear
    act_proj: eqx.nn.Linear
    transformer: Transformer
    output_proj: eqx.nn.Linear
    num_obs: int = eqx.static_field()
    num_act: int = eqx.static_field()
    num_frames: int = eqx.static_field()

    def __init__(
        self,
        key: PRNGKeyArray,
        num_obs: int,
        num_act: int,
        hidden_size: int,
        depth: int,
        num_heads: int,
        num_frames: int,
        cls_init_scale: float,
    ):
        obs_key, act_key, out_key, transformer_key = jax.random.split(key, 4)
        self.obs_proj = eqx.nn.Linear(num_obs, hidden_size, key=obs_key)
        self.act_proj = eqx.nn.Linear(num_act, hidden_size, key=act_key)
        self.transformer = Transformer(
            hidden_size=hidden_size,
            num_layers=depth,
            num_heads=num_heads,
            context_length=num_frames * 2,  # obs + act
            cls_init_scale=cls_init_scale,
            key=transformer_key,
        )
        self.output_proj = eqx.nn.Linear(hidden_size, 1, key=out_key)

        self.num_obs = num_obs
        self.num_act = num_act
        self.num_frames = num_frames

    def forward(self, obs_n: Array, act_n: Array) -> Array:
        """Forward pass of the critic network."""
        chex.assert_shape(obs_n, (self.num_frames, self.num_obs))
        chex.assert_shape(act_n, (self.num_frames, self.num_act))

        # Create and interleave obs and act embeddings.
        obs_emb = jax.vmap(self.obs_proj)(obs_n)
        act_emb = jax.vmap(self.act_proj)(act_n)
        interleaved_indices = jnp.arange(2 * obs_emb.shape[0]).reshape(2, -1).T.ravel()  # [0,L,1,L+1,2,L+2...]
        x_emb = jnp.concatenate([obs_emb, act_emb])[interleaved_indices]

        # Getting the CLS and predicting output.
        cls_emb = self.transformer.forward_sequence(x_emb)[-1]
        out_n = self.output_proj(cls_emb)
        return out_n


class DefaultHumanoidTransformerModel(eqx.Module):
    """Combined actor-critic model for the humanoid walking task.

    Attributes:
        actor: Policy network that outputs action distributions.
        critic: Value network that estimates state values.
    """

    actor: DefaultHumanoidTransformerActor
    critic: DefaultHumanoidTransformerCritic

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
        num_heads: int,
        num_frames: int,
        cls_init_scale: float,
    ) -> None:
        """Initialize the actor-critic model."""
        self.actor = DefaultHumanoidTransformerActor(
            key,
            num_obs=num_actor_inputs,
            num_act=num_joints,
            min_std=min_std,
            max_std=max_std,
            var_scale=0.5,
            num_mixtures=num_actor_mixtures,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            num_frames=num_frames,
            cls_init_scale=cls_init_scale,
        )
        self.critic = DefaultHumanoidTransformerCritic(
            key,
            num_obs=num_critic_inputs,
            num_act=num_joints,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            num_frames=num_frames,
            cls_init_scale=cls_init_scale,
        )


@dataclass
class HumanoidWalkingTransformerTaskConfig(HumanoidWalkingTaskConfig):
    """Configuration for training a walking policy using an Transformer."""

    hidden_size: int = xax.field(
        value=128,
        help="The hidden size for the Transformer.",
    )
    depth: int = xax.field(
        value=3,
        help="The number of Transformer layers.",
    )
    num_heads: int = xax.field(
        value=1,
        help="The number of attention heads for the Transformer.",
    )
    num_frames: int = xax.field(
        value=3,
        help="The number of obs act frames to use for the Transformer.",
    )
    num_mixtures: int = xax.field(
        value=5,
        help="The number of mixtures for the actor.",
    )
    cls_init_scale: float = xax.field(
        value=0.01,
        help="The initial scale for the CLS token.",
    )


Config = TypeVar("Config", bound=HumanoidWalkingTransformerTaskConfig)


class HumanoidWalkingTransformerTask(HumanoidWalkingTask[Config], Generic[Config]):
    """A task for training a walking policy using an Transformer."""

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

    def get_model(self, key: PRNGKeyArray) -> DefaultHumanoidTransformerModel:
        """Creates the actor-critic model."""
        return DefaultHumanoidTransformerModel(
            key,
            num_actor_inputs=self.actor_num_inputs,
            num_critic_inputs=self.critic_num_inputs,
            num_joints=NUM_JOINTS,
            min_std=0.01,
            max_std=1.0,
            num_actor_mixtures=self.config.num_mixtures,
            hidden_size=self.config.hidden_size,
            depth=self.config.depth,
            num_heads=self.config.num_heads,
            num_frames=self.config.num_frames,
            cls_init_scale=self.config.cls_init_scale,
        )

    def get_initial_model_carry(self, rng: PRNGKeyArray) -> HistoricalCarry:
        """Gets initial carry state for the model.

        Args:
            rng: Random number generator key.

        Returns:
            Initial Transformer hidden states for actor and critic.
        """
        empty_obs = {
            "joint_position_observation": jnp.zeros((self.config.num_frames, NUM_JOINTS)),
            "joint_velocity_observation": jnp.zeros((self.config.num_frames, NUM_JOINTS)),
            "actuator_force_observation": jnp.zeros((self.config.num_frames, NUM_JOINTS)),
            "center_of_mass_inertia_observation": jnp.zeros((self.config.num_frames, 160)),
            "center_of_mass_velocity_observation": jnp.zeros((self.config.num_frames, 96)),
            "base_position_observation": jnp.zeros((self.config.num_frames, 3)),
            "base_orientation_observation": jnp.zeros((self.config.num_frames, 4)),
            "base_linear_velocity_observation": jnp.zeros((self.config.num_frames, 3)),
            "base_angular_velocity_observation": jnp.zeros((self.config.num_frames, 3)),
            "sensor_observation_imu_acc": jnp.zeros((self.config.num_frames, 3)),
            "sensor_observation_imu_gyro": jnp.zeros((self.config.num_frames, 3)),
            "timestep_observation": jnp.zeros((self.config.num_frames, 1)),
        }
        return HistoricalCarry(
            observations=xax.FrozenDict(empty_obs),
            actions=jnp.zeros((self.config.num_frames, NUM_JOINTS)),
        )

    def run_actor(
        self,
        model: DefaultHumanoidTransformerActor,
        observations: xax.FrozenDict[str, Array],
        actions: Array,
        commands: xax.FrozenDict[str, Array],
    ) -> distrax.Distribution:
        """Runs the actor network to get an action distribution.

        Args:
            model: The actor network.
            observations: Previous and current observations [T, obs_dim]
            actions: Action history [T, act_dim]
            commands: Current commands.

        Returns:
            An action distribution.
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

        return model.forward(obs_n, actions)

    def get_obs_act(self, trajectory: ksim.Trajectory, initial_history: HistoricalCarry) -> HistoricalCarry:
        """Gets the observation and action history from a trajectory."""

        def scan_fn(history: HistoricalCarry, transition: ksim.Trajectory) -> tuple[HistoricalCarry, HistoricalCarry]:
            obs_history = history.observations
            obs = jax.tree.map(lambda x, y: jnp.concatenate([x[1:], y[None]], axis=0), obs_history, transition.obs)
            act = history.actions

            # Like sample action, use new obs and old act.
            emit = HistoricalCarry(observations=obs, actions=act)

            # For next carry, we shift actions (or reset).
            next_act = jnp.concatenate([act[1:], transition.action[None]], axis=0)
            next_carry = jax.lax.cond(
                transition.done,
                lambda: self.get_initial_model_carry(jax.random.PRNGKey(0)),
                lambda: HistoricalCarry(observations=obs, actions=next_act),
            )

            return next_carry, emit

        _, obs_act = jax.lax.scan(
            scan_fn,
            initial_history,
            trajectory,
        )
        return obs_act

    def run_critic(
        self,
        model: DefaultHumanoidTransformerCritic,
        observations: xax.FrozenDict[str, Array],
        actions: Array,
        commands: xax.FrozenDict[str, Array],
    ) -> Array:
        """Runs the critic network to get state values.

        Args:
            model: The critic network.
            observations: Previous and current observations [T, obs_dim]
            actions: Action history [T, act_dim]
            commands: Current commands.

        Returns:
            State values.
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

        return model.forward(obs_n, actions)

    def get_ppo_variables(
        self,
        model: DefaultHumanoidTransformerModel,
        trajectory: ksim.Trajectory,
        model_carry: HistoricalCarry,
        rng: PRNGKeyArray,
    ) -> tuple[ksim.PPOVariables, HistoricalCarry]:
        """Computes PPO variables for training.

        Args:
            model: The actor-critic model.
            trajectory: The trajectory to compute variables for.
            model_carry: The starting action history [T, act_dim]
            rng: Random number generator key.

        Returns:
            A tuple of PPO variables and the next model carry state.
        """
        # Get observation and action history from trajectory
        obs_act = self.get_obs_act(trajectory, model_carry)
        actions_history = obs_act.actions
        on_policy_actions = trajectory.action
        observations = obs_act.observations
        commands = trajectory.command

        def get_log_probs_values(
            action_history: Array,
            on_policy_action: Array,
            observations: xax.FrozenDict[str, Array],
            commands: xax.FrozenDict[str, Array],
        ) -> tuple[Array, Array]:
            action_dist = self.run_actor(model.actor, observations, action_history, commands)
            log_probs = action_dist.log_prob(on_policy_action)
            assert isinstance(log_probs, Array)
            values = self.run_critic(model.critic, observations, action_history, commands)
            return log_probs, values.squeeze(-1)

        log_probs, values = jax.vmap(get_log_probs_values)(actions_history, on_policy_actions, observations, commands)
        ppo_variables = ksim.PPOVariables(
            log_probs=log_probs,
            values=values,
        )

        # Passing in the next historical carry to the next rollout.
        next_model_carry = jax.tree.map(lambda x: x[-1], obs_act)

        return ppo_variables, next_model_carry

    def sample_action(
        self,
        model: DefaultHumanoidTransformerModel,
        model_carry: HistoricalCarry,
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
            model_carry: The action history [T, act_dim]
            physics_model: The physics model.
            physics_state: The current physics state.
            observations: Current observations.
            commands: Current commands.
            rng: Random number generator key.
            argmax: Whether to take the mode of the distribution instead of sampling.

        Returns:
            A sampled action with its carry state.
        """
        # Shifting the observations by one (actions are already shifted).
        obs_n = jax.tree.map(
            lambda x, y: jnp.concatenate([x[1:], y[None]], axis=0), model_carry.observations, observations
        )
        act_n = model_carry.actions

        # Runs the actor model to get the action distribution.
        action_dist_j = self.run_actor(
            model=model.actor,
            observations=obs_n,
            actions=act_n,
            commands=commands,
        )

        action_j = action_dist_j.mode() if argmax else action_dist_j.sample(seed=rng)

        # Updates the historical carry and shifts actions.
        next_carry = HistoricalCarry(
            observations=obs_n,
            actions=jnp.concatenate([act_n[1:], action_j[None]], axis=0),
        )

        return ksim.Action(
            action=action_j,
            carry=next_carry,
            aux_outputs=None,
        )


if __name__ == "__main__":
    # To run training, use the following command:
    #   python -m iclrobot.dh_walking.dh_walking_Transformer
    # To visualize the environment, use the following command:
    #   python -m iclrobot.dh_walking.dh_walking_Transformer run_environment=True
    # On MacOS or other devices with less memory, you can change the number
    # of environments and batch size to reduce memory usage. Here's an example
    # from the command line:
    #   python -m iclrobot.dh_walking.dh_walking_Transformer num_envs=8 rollouts_per_batch=4
    HumanoidWalkingTransformerTask.launch(
        HumanoidWalkingTransformerTaskConfig(
            # Training parameters.
            num_envs=2048,
            batch_size=128,
            num_passes=4,
            epochs_per_log_step=1,
            rollout_length_seconds=5.0,
            # Simulation parameters.
            dt=0.005,
            ctrl_dt=0.02,
            max_action_latency=0.0,
            min_action_latency=0.0,
            randomize_physics=True,
        ),
    )

"""Defines simple task for training a walking policy for the default humanoid."""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar

import attrs
import glm
import jax
import jax.numpy as jnp
import numpy as np
import ksim
import mujoco
import optax
from ksim.actuators import NoiseType
from ksim.types import PhysicsModel
from ksim.utils.reference_motion import (
    ReferenceMapping,
    ReferenceMotionData,
    generate_reference_motion,
    get_reference_joint_id,
    visualize_reference_motion,
    visualize_reference_points,
)
import xax
from jaxtyping import Array, PRNGKeyArray, PyTree
from kscale.web.gen.api import JointMetadataOutput
from mujoco import mjx


try:
    import bvhio
    from bvhio.lib.hierarchy import Joint as BvhioJoint
except ImportError as e:
    raise ImportError(
        "In order to use reference motion utilities, please install Bvhio, using 'pip install bvhio'."
    ) from e

from scipy.spatial.transform import Rotation as R

NUM_JOINTS = 21


HUMANOID_REFERENCE_MAPPINGS = (
    ReferenceMapping("CC_Base_L_ThighTwist01", "thigh_left"),  # hip
    ReferenceMapping("CC_Base_L_CalfTwist01", "shin_left"),  # knee
    ReferenceMapping("CC_Base_L_Foot", "foot_left"),  # foot
    ReferenceMapping("CC_Base_L_UpperarmTwist01", "upper_arm_left"),  # shoulder
    ReferenceMapping("CC_Base_L_ForearmTwist01", "lower_arm_left"),  # elbow
    ReferenceMapping("CC_Base_L_Hand", "hand_left"),  # hand
    ReferenceMapping("CC_Base_R_ThighTwist01", "thigh_right"),  # hip
    ReferenceMapping("CC_Base_R_CalfTwist01", "shin_right"),  # knee
    ReferenceMapping("CC_Base_R_Foot", "foot_right"),  # foot
    ReferenceMapping("CC_Base_R_UpperarmTwist01", "upper_arm_right"),  # shoulder
    ReferenceMapping("CC_Base_R_ForearmTwist01", "lower_arm_right"),  # elbow
    ReferenceMapping("CC_Base_R_Hand", "hand_right"),  # hand
)


@dataclass
class HumanoidWalkingTaskConfig(ksim.PPOConfig):
    """Config for the humanoid walking task."""

    # Reward parameters.
    linear_velocity_clip_max: float = xax.field(
        value=2.0,
        help="The maximum value for the linear velocity reward.",
    )

    # Optimizer parameters.
    learning_rate: float = xax.field(
        value=1e-3,
        help="Learning rate for PPO.",
    )
    max_grad_norm: float = xax.field(
        value=2.0,
        help="Maximum gradient norm for clipping.",
    )
    adam_weight_decay: float = xax.field(
        value=0.0,
        help="Weight decay for the Adam optimizer.",
    )

    # Mujoco parameters.
    kp: float = xax.field(
        value=1.0,
        help="The Kp for the actuators",
    )
    kd: float = xax.field(
        value=0.1,
        help="The Kd for the actuators",
    )
    armature: float = xax.field(
        value=1e-2,
        help="A value representing the effective inertia of the actuator armature",
    )
    friction: float = xax.field(
        value=1e-6,
        help="The dynamic friction loss for the actuator",
    )

    # Curriculum parameters.
    num_curriculum_levels: int = xax.field(
        value=10,
        help="The number of curriculum levels to use.",
    )
    increase_threshold: float = xax.field(
        value=3.0,
        help="Increase the curriculum level when the mean trajectory length is above this threshold.",
    )
    decrease_threshold: float = xax.field(
        value=1.0,
        help="Decrease the curriculum level when the mean trajectory length is below this threshold.",
    )
    min_level_steps: int = xax.field(
        value=50,
        help="The minimum number of steps to wait before changing the curriculum level.",
    )

    # Reference motion parameters.
    bvh_path: str = xax.field(
        value=str(Path(__file__).parent / "data" / "walk_normal_dh.bvh"),
        help="The path to the BVH file.",
    )
    rotate_bvh_euler: tuple[float, float, float] = xax.field(
        value=(0, 0, 0),
        help="Optional rotation to ensure the BVH tree matches the Mujoco model.",
    )
    bvh_scaling_factor: float = xax.field(
        value=1.0,
        help="Scaling factor to ensure the BVH tree matches the Mujoco model.",
    )
    bvh_offset: tuple[float, float, float] = xax.field(
        value=(0.0, 0.0, 0.0),
        help="Offset to ensure the BVH tree matches the Mujoco model.",
    )
    mj_base_name: str = xax.field(
        value="pelvis",
        help="The Mujoco body name of the base of the humanoid",
    )
    reference_base_name: str = xax.field(
        value="CC_Base_Pelvis",
        help="The BVH joint name of the base of the humanoid",
    )
    visualize_reference_points: bool = xax.field(
        value=False,
        help="Whether to visualize the reference points.",
    )
    visualize_reference_motion: bool = xax.field(
        value=False,
        help="Whether to visualize the reference motion after running IK.",
    )

    # Engine parameters.
    min_action_latency: float = xax.field(
        value=0.0,
        help="The minimum latency of the action.",
    )
    max_action_latency: float = xax.field(
        value=0.0,
        help="The maximum latency of the action.",
    )
    iterations: int = xax.field(
        value=8,
        help="The number of iterations to use for the solver.",
    )
    ls_iterations: int = xax.field(
        value=8,
        help="The number of line search iterations to use for the solver.",
    )

    # Rendering parameters.
    render_track_body_id: int | None = xax.field(
        value=0,
        help="The body id to track with the render camera.",
    )
    full_size_render: bool = xax.field(
        value=True,
        help="Whether to render the full size video.",
    )

    # Checkpointing parameters.
    export_for_inference: bool = xax.field(
        value=False,
        help="Whether to export the model for inference.",
    )

    # Experimental parameters.
    randomize: bool = xax.field(
        value=False,
        help="Whether to randomize the physics during training.",
    )


Config = TypeVar("Config", bound=HumanoidWalkingTaskConfig)


@attrs.define(frozen=True, kw_only=True)
class NaiveForwardReward(ksim.Reward):
    """A simple reward function that rewards forward velocity up to a maximum value.

    Attributes:
        clip_max: Maximum value to clip the forward velocity reward to.
    """

    clip_max: float = attrs.field(default=5.0)

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        """Compute the reward based on forward velocity.

        Args:
            trajectory: The trajectory to compute the reward for.

        Returns:
            A tuple containing the clipped forward velocity reward.
        """
        return trajectory.qvel[..., 0].clip(max=self.clip_max)


@attrs.define(frozen=True, kw_only=True)
class QposReferenceMotionReward(ksim.Reward):
    reference_qpos: xax.HashableArray
    ctrl_dt: float
    norm: xax.NormType = attrs.field(default="l1")
    sensitivity: float = attrs.field(default=5.0)

    @property
    def num_frames(self) -> int:
        return self.reference_qpos.array.shape[0]

    def __call__(self, trajectory: ksim.Trajectory, _: None) -> tuple[Array, None]:
        """Compute the reward based on the reference motion.

        Args:
            trajectory: The trajectory to compute the reward for.

        Returns:
            A tuple containing the reward and None for the carry state.
        """
        qpos = trajectory.qpos
        step_number = jnp.int32(jnp.round(trajectory.timestep / self.ctrl_dt)) % self.num_frames
        reference_qpos = jnp.take(self.reference_qpos.array, step_number, axis=0)
        error = xax.get_norm(reference_qpos - qpos, self.norm)
        mean_error = error.mean(axis=-1)
        reward = jnp.exp(-mean_error * self.sensitivity)
        return reward, None


class BiasedPositionActuators(ksim.MITPositionActuators, ksim.StatefulActuators):
    """Actuator controller operating on position."""

    def __init__(
        self,
        bias_range: tuple[float, float],
        physics_model: PhysicsModel,
        joint_name_to_metadata: dict[str, JointMetadataOutput],
        action_noise: float = 0.0,
        action_noise_type: NoiseType = "none",
        torque_noise: float = 0.0,
        torque_noise_type: NoiseType = "none",
        ctrl_clip: list[float] | None = None,
        freejoint_first: bool = True,
    ) -> None:
        """Creates easily vector multipliable kps and kds."""
        super().__init__(
            physics_model=physics_model,
            joint_name_to_metadata=joint_name_to_metadata,
            action_noise=action_noise,
            action_noise_type=action_noise_type,
            torque_noise=torque_noise,
            torque_noise_type=torque_noise_type,
            ctrl_clip=ctrl_clip,
            freejoint_first=freejoint_first,
        )
        self.bias_range = bias_range

    def get_stateful_ctrl(
        self,
        action: Array,
        physics_data: ksim.PhysicsData,
        actuator_state: PyTree,
        rng: PRNGKeyArray,
    ) -> tuple[Array, PyTree]:
        """Get the control signal from the action vector."""
        bias = actuator_state
        action = action + bias
        return super().get_ctrl(action, physics_data, rng), actuator_state

    def get_initial_state(self, physics_data: ksim.PhysicsData, rng: PRNGKeyArray) -> PyTree:
        """Get the default state for the actuator."""
        return jax.random.uniform(rng, (NUM_JOINTS,), minval=self.bias_range[0], maxval=self.bias_range[1])


class HumanoidWalkingTask(ksim.PPOTask[Config], Generic[Config]):
    """Task for training a humanoid robot to walk using PPO.

    This class implements a PPO-based training task for teaching a humanoid robot to walk.
    It includes reward shaping, curriculum learning, and various safety checks.
    """

    reference_motion: ReferenceMotionData

    def get_optimizer(self) -> optax.GradientTransformation:
        """Builds the optimizer for training."""
        optimizer = optax.chain(
            optax.clip_by_global_norm(self.config.max_grad_norm),
            (
                optax.adam(self.config.learning_rate)
                if self.config.adam_weight_decay == 0.0
                else optax.adamw(self.config.learning_rate, weight_decay=self.config.adam_weight_decay)
            ),
        )
        return optimizer

    def get_mujoco_model(self) -> tuple[mujoco.MjModel, dict[str, JointMetadataOutput]]:
        """Creates and configures the MuJoCo physics model."""
        mjcf_path = (Path(__file__).parent / "data" / "default_humanoid.mjcf").resolve().as_posix()
        mj_model = mujoco.MjModel.from_xml_path(mjcf_path)

        mj_model.opt.timestep = jnp.array(self.config.dt)
        mj_model.opt.iterations = 4
        mj_model.opt.ls_iterations = 8
        mj_model.opt.disableflags = mjx.DisableBit.EULERDAMP
        mj_model.opt.solver = mjx.SolverType.CG
        return mj_model

    def get_mujoco_model_metadata(self, mj_model: mujoco.MjModel) -> dict[str, JointMetadataOutput]:
        """Gets metadata for the MuJoCo model joints."""
        return ksim.get_joint_metadata(
            mj_model,
            kp=self.config.kp,
            kd=self.config.kd,
            armature=self.config.armature,
            friction=self.config.friction,
        )

    def get_actuators(
        self,
        physics_model: ksim.PhysicsModel,
        metadata: dict[str, JointMetadataOutput] | None = None,
    ) -> ksim.Actuators:
        """Creates actuators for controlling the robot's joints."""
        assert metadata is not None, "Metadata is required"
        return BiasedPositionActuators(
            bias_range=(-0.1, 0.1) if self.config.randomize else (0.0, 0.0),
            physics_model=physics_model,
            joint_name_to_metadata=metadata,
        )

    def get_physics_randomizers(self, physics_model: ksim.PhysicsModel) -> list[ksim.PhysicsRandomizer]:
        """Gets randomizers for domain randomization during training."""
        if self.config.randomize:
            return [
                ksim.StaticFrictionRandomizer(scale_lower=0.1, scale_upper=10.0),
                ksim.ArmatureRandomizer(scale_lower=0.75, scale_upper=1.25),
                ksim.MassMultiplicationRandomizer.from_body_name(
                    physics_model, "torso", scale_lower=0.75, scale_upper=1.25
                ),
                ksim.JointDampingRandomizer(scale_lower=0.75, scale_upper=1.25),
                ksim.JointZeroPositionRandomizer(scale_lower=-0.20, scale_upper=0.20),
            ]
        else:
            return []

    def get_events(self, physics_model: ksim.PhysicsModel) -> list[ksim.Event]:
        """Gets events that can occur during training."""
        # TODO: add better events here...
        return []

    def get_resets(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reset]:
        """Gets reset operations for episode initialization."""
        return [
            ksim.RandomJointPositionReset(),
            ksim.RandomJointVelocityReset(),
        ]

    def get_observations(self, physics_model: ksim.PhysicsModel) -> list[ksim.Observation]:
        """Gets observation components for the policy."""
        return [
            ksim.JointPositionObservation(),
            ksim.JointVelocityObservation(),
            ksim.ActuatorForceObservation(),
            ksim.CenterOfMassInertiaObservation(),
            ksim.CenterOfMassVelocityObservation(),
            ksim.BasePositionObservation(),
            ksim.BaseOrientationObservation(),
            ksim.BaseLinearVelocityObservation(),
            ksim.BaseAngularVelocityObservation(),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="imu_acc"),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="imu_gyro"),
            ksim.TimestepObservation(),
        ]

    def get_commands(self, physics_model: ksim.PhysicsModel) -> list[ksim.Command]:
        """Gets command components for controlling the robot."""
        return []

    def get_rewards(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reward]:
        """Gets reward components for the learning objective."""
        return [
            ksim.StayAliveReward(scale=1.0),
            NaiveForwardReward(scale=0.1, clip_max=self.config.linear_velocity_clip_max),
            # QposReferenceMotionReward(
            #     reference_qpos=self.reference_motion.qpos, ctrl_dt=self.config.ctrl_dt, scale=0.5
            # ),
        ]

    def get_terminations(self, physics_model: ksim.PhysicsModel) -> list[ksim.Termination]:
        """Gets termination conditions for episodes."""
        return [
            ksim.BadZTermination(unhealthy_z_lower=0.9, unhealthy_z_upper=1.6),
            ksim.PitchTooGreatTermination(max_pitch=math.pi / 3),
            ksim.RollTooGreatTermination(max_roll=math.pi / 3),
            ksim.FastAccelerationTermination(),
        ]

    def get_curriculum(self, physics_model: ksim.PhysicsModel) -> ksim.Curriculum:
        """Gets the curriculum for progressive learning."""
        return ksim.EpisodeLengthCurriculum(
            num_levels=self.config.num_curriculum_levels,
            increase_threshold=self.config.increase_threshold,
            decrease_threshold=self.config.decrease_threshold,
            min_level_steps=self.config.min_level_steps,
            dt=self.config.ctrl_dt,
        )

    def run(self) -> None:
        mj_model: PhysicsModel = self.get_mujoco_model()
        root: BvhioJoint = bvhio.readAsHierarchy(self.config.bvh_path)
        reference_base_id = get_reference_joint_id(root, self.config.reference_base_name)
        self.mj_base_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, self.config.mj_base_name)

        def rotation_callback(root: BvhioJoint) -> None:
            euler_rotation = np.array(self.config.rotate_bvh_euler)
            quat = R.from_euler("xyz", euler_rotation).as_quat(scalar_first=True)
            root.applyRotation(glm.quat(*quat), bake=True)

        reference_motion = generate_reference_motion(
            model=mj_model,
            mj_base_id=self.mj_base_id,
            bvh_root=root,
            bvh_to_mujoco_names=HUMANOID_REFERENCE_MAPPINGS,
            bvh_base_id=reference_base_id,
            bvh_offset=np.array(self.config.bvh_offset),
            bvh_root_callback=rotation_callback,
            bvh_scaling_factor=self.config.bvh_scaling_factor,
            ctrl_dt=self.config.ctrl_dt,
            neutral_qpos=None,
            neutral_similarity_weight=0.1,
            temporal_consistency_weight=0.1,
            n_restarts=3,
            error_acceptance_threshold=1e-4,
            ftol=1e-8,
            xtol=1e-8,
            max_nfev=2000,
            verbose=False,
        )
        self.reference_motion = reference_motion
        np_cartesian_motion = jax.tree.map(np.asarray, self.reference_motion.cartesian_poses)

        if self.config.visualize_reference_points:
            visualize_reference_points(
                model=mj_model,
                base_id=self.mj_base_id,
                reference_motion=np_cartesian_motion,
            )
        elif self.config.visualize_reference_motion:
            visualize_reference_motion(
                model=mj_model,
                reference_qpos=np.asarray(self.reference_motion.qpos),
                cartesian_motion=np_cartesian_motion,
                mj_base_id=self.mj_base_id,
            )
        else:
            super().run()

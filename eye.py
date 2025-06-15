#!/usr/bin/env python3


# Based on <https://github.com/google/brax/blob/759256a27ec495c8307bcd141e0798c5f090a1df/brax/envs/ant.py>.


from brax import base, envs, math
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from etils import epath
import jax
from jax import numpy as jp, debug
import mujoco


N_LEGS = 3
SERVOS_PER_LEG = 3
ACTION_SHAPE = (SERVOS_PER_LEG * N_LEGS,)
OBSERVATION_SHAPE = (11 + 2 * SERVOS_PER_LEG * N_LEGS,)


IDEAL_VELOCITY = 1  # m/s


class Eye(PipelineEnv):
    """
    ### Description

    This environment is based on the environment introduced by Schulman, Moritz,
    Levine, Jordan and Abbeel in
    ["High-Dimensional Continuous Control Using Generalized Advantage Estimation"](https://arxiv.org/abs/1506.02438).

    The ant is a 3D robot consisting of one torso (free rotational body) with four
    legs attached to it with each leg having two links.

    The goal is to coordinate the four legs to move in the forward (right)
    direction by applying torques on the eight hinges connecting the two links of
    each leg and the torso (nine parts and eight hinges).

    ### Action Space

    The agent takes a vector for actions.

    The action space is a continuous `(action, action, action, action, action, action)`,
    all in `[-1, 1]`, where `action` represents the positions of a hinge joint.

    | Num | Action                                                       | Control Min | Control Max | Name (in corresponding config) | Joint | Unit        |
    |-----|--------------------------------------------------------------|-------------|-------------|--------------------------------|-------|-------------|
    |  0  | Position of the rotor between the torso and front left hip   | -pi (360 d) | pi (360 d)  | roll_1                         | hinge | angle (rad) |
    |  1  | Position of the rotor between the torso and front left hip   | -pi (360 d) | pi (360 d)  | hip_1                          | hinge | angle (rad) |
    |  2  | Position of the rotor between the front left two links       | -pi (360 d) | pi (360 d)  | knee_1                         | hinge | angle (rad) |
    |  3  | Position of the rotor between the torso and front left hip   | -pi (360 d) | pi (360 d)  | roll_2                         | hinge | angle (rad) |
    |  4  | Position of the rotor between the torso and front right hip  | -pi (360 d) | pi (360 d)  | hip_2                          | hinge | angle (rad) |
    |  5  | Position of the rotor between the front right two links      | -pi (360 d) | pi (360 d)  | knee_2                         | hinge | angle (rad) |
    |  6  | Position of the rotor between the torso and front left hip   | -pi (360 d) | pi (360 d)  | roll_3                         | hinge | angle (rad) |
    |  7  | Position of the rotor between the torso and back left hip    | -pi (360 d) | pi (360 d)  | hip_3                          | hinge | angle (rad) |
    |  8  | Position of the rotor between the back left two links        | -pi (360 d) | pi (360 d)  | knee_3                         | hinge | angle (rad) |

    ### Observation Space

    The state space consists of positional values of different body parts of the
    ant, followed by the velocities of those individual parts (their derivatives)
    with all the positions ordered before all the velocities.

    The observation is an array whose elements correspond to the following:

    | Num | Observation                                                  | Name (in corresponding config) | Joint | Unit                     |
    |-----|--------------------------------------------------------------|--------------------------------|-------|--------------------------|
    |  0  | z-coordinate of the torso (centre)                           | torso                          | free  | position (m)             |
    |  1  | w-orientation of the torso (centre)                          | torso                          | free  | angle (rad)              |
    |  2  | x-orientation of the torso (centre)                          | torso                          | free  | angle (rad)              |
    |  3  | y-orientation of the torso (centre)                          | torso                          | free  | angle (rad)              |
    |  4  | z-orientation of the torso (centre)                          | torso                          | free  | angle (rad)              |
    |  5  | joint angle                                                  | roll_1                         | hinge | angle (rad)              |
    |  6  | joint angle                                                  | hip_1                          | hinge | angle (rad)              |
    |  7  | joint angle                                                  | knee_1                         | hinge | angle (rad)              |
    |  8  | joint angle                                                  | roll_2                         | hinge | angle (rad)              |
    |  9  | joint angle                                                  | hip_2                          | hinge | angle (rad)              |
    | 10  | joint angle                                                  | knee_2                         | hinge | angle (rad)              |
    | 11  | joint angle                                                  | roll_3                         | hinge | angle (rad)              |
    | 12  | joint angle                                                  | hip_3                          | hinge | angle (rad)              |
    | 13  | joint angle                                                  | knee_3                         | hinge | angle (rad)              |
    | 14  | x-coordinate velocity of the torso                           | torso                          | free  | velocity (m/s)           |
    | 15  | y-coordinate velocity of the torso                           | torso                          | free  | velocity (m/s)           |
    | 16  | z-coordinate velocity of the torso                           | torso                          | free  | velocity (m/s)           |
    | 17  | x-coordinate angular velocity of the torso                   | torso                          | free  | angular velocity (rad/s) |
    | 18  | y-coordinate angular velocity of the torso                   | torso                          | free  | angular velocity (rad/s) |
    | 19  | z-coordinate angular velocity of the torso                   | torso                          | free  | angular velocity (rad/s) |
    | 20  | joint angular velocity                                       | roll_1                         | hinge | angle (rad)              |
    | 21  | joint angular velocity                                       | hip_1                          | hinge | angle (rad)              |
    | 22  | joint angular velocity                                       | knee_1                         | hinge | angle (rad)              |
    | 23  | joint angular velocity                                       | roll_2                         | hinge | angle (rad)              |
    | 24  | joint angular velocity                                       | hip_2                          | hinge | angle (rad)              |
    | 25  | joint angular velocity                                       | knee_2                         | hinge | angle (rad)              |
    | 26  | joint angular velocity                                       | roll_3                         | hinge | angle (rad)              |
    | 27  | joint angular velocity                                       | hip_3                          | hinge | angle (rad)              |
    | 28  | joint angular velocity                                       | knee_3                         | hinge | angle (rad)              |

    The (x,y,z) coordinates are translational DOFs while the orientations are
    rotational DOFs expressed as quaternions.

    ### Rewards

    The reward consists of three parts:

    - *reward_forward*: A reward of moving forward which is measured as
      *(x-coordinate before action - x-coordinate after action)/dt*. *dt* is the
      time between actions - the default *dt = 0.05*. This reward would be
      positive if the ant moves forward (right) desired.
    - *reward_orientation*: Reward for facing forward, not turning, flipping, etc.
    - *reward_survive*: Every timestep that the ant is alive, it gets a reward of 1.
    - *reward_torque*: A negative reward for total torque used.
    # - *contact_cost*: A negative reward for penalising the ant if the external
    #   contact force is too large. It is calculated *0.5 * 0.001 *
    #   sum(clip(external contact force to [-1,1])<sup>2</sup>)*.

    ### Starting State

    All observations start in state (0.0, 0.0,  0.75, 1.0, 0.0  ... 0.0) with a
    uniform noise in the range of [-0.1, 0.1] added to the positional values and
    standard normal noise with 0 mean and 0.1 standard deviation added to the
    velocity values for stochasticity.

    Note that the initial z coordinate is intentionally selected to be slightly
    high, thereby indicating a standing up ant. The initial orientation is
    designed to make it face forward as well.

    ### Episode Termination

    The episode terminates when any of the following happens:

    1. The episode duration reaches a 1000 timesteps
    2. The y-orientation (index 2) in the state is below `0.2`.
    """

    # pyformat: enable

    def __init__(
        self,
        torque_cost_weight=0.5,
        # contact_cost_weight=5e-4,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.2, None),
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=True,
        backend="mjx",
        **kwargs,
    ):
        path = "./eye.xml"
        sys = mjcf.load(path)

        n_frames = 5

        kwargs["n_frames"] = kwargs.get("n_frames", n_frames)

        super().__init__(sys=sys, backend=backend, **kwargs)

        self._torque_cost_weight = torque_cost_weight
        # self._contact_cost_weight = contact_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2 = jax.random.split(rng, 3)

        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        q = self.sys.init_q
        # q += jax.random.uniform(
        #     rng1, (self.sys.q_size(),), minval=low, maxval=hi
        # )
        qd = hi * jax.random.normal(rng2, (self.sys.qd_size(),))

        pipeline_state = self.pipeline_init(q, qd)
        obs = self._get_obs(pipeline_state)

        reward, done, zero = jp.zeros(3)
        metrics = {
            "reward_forward": zero,
            "reward_orientation": zero,
            "reward_survive": zero,
            "reward_torque": zero,
            # "reward_contact": zero,
            "x_position": zero,
            "y_position": zero,
            "distance_from_origin": zero,
            "x_velocity": zero,
            "y_velocity": zero,
        }
        info = {}
        return State(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=reward,
            done=done,
            metrics=metrics,
            info=info,
        )

    def step(self, state: State, action: jax.Array) -> State:
        """Run one timestep of the environment's dynamics."""
        pipeline_state0 = state.pipeline_state
        assert pipeline_state0 is not None
        pipeline_state = self.pipeline_step(pipeline_state0, action)

        velocity = (pipeline_state.x.pos[0] - pipeline_state0.x.pos[0]) / self.dt
        forward_reward = -jp.abs(velocity[0] - IDEAL_VELOCITY)

        # # Ignore the first element of the body orientation quaternion,
        # # since that's the one that should be one (and the others zero)
        # # whenever the body is fully facing forward:
        # off_axis_quaternion = pipeline_state.q[4:7]
        # orientation_reward = -jp.sum(jp.square(off_axis_quaternion))
        roll, pitch, yaw = math.quat_to_euler(pipeline_state.q[3:7])
        orientation_reward = -jp.abs(yaw)

        min_z, max_z = self._healthy_z_range
        zpos = pipeline_state.x.pos[0, 2]
        is_healthy = jp.ones(zpos.shape)
        if min_z is not None:
            is_healthy = jp.where(zpos < min_z, 0.0, is_healthy)
        if max_z is not None:
            is_healthy = jp.where(zpos > max_z, 0.0, is_healthy)
        if self._terminate_when_unhealthy:
            healthy_reward = self._healthy_reward
        else:
            healthy_reward = self._healthy_reward * is_healthy

        servo_torques = pipeline_state.qfrc_actuator[:-(N_LEGS * SERVOS_PER_LEG)]
        torque_cost = self._torque_cost_weight * jp.sum(
            jp.abs(servo_torques)
        )

        obs = self._get_obs(pipeline_state)
        reward = (
            forward_reward
            + orientation_reward
            + healthy_reward
            - torque_cost
            # - contact_cost
        )
        done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0
        state.metrics.update(
            reward_forward=forward_reward,
            reward_orientation=orientation_reward,
            reward_survive=healthy_reward,
            reward_torque=-torque_cost,
            # reward_contact=-contact_cost,
            x_position=pipeline_state.x.pos[0, 0],
            y_position=pipeline_state.x.pos[0, 1],
            distance_from_origin=math.safe_norm(pipeline_state.x.pos[0]),
            x_velocity=velocity[0],
            y_velocity=velocity[1],
        )
        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )

    def _get_obs(self, pipeline_state: base.State) -> jax.Array:
        """Observe eye body position and velocities."""
        qpos = pipeline_state.q
        qvel = pipeline_state.qd

        if self._exclude_current_positions_from_observation:
            qpos = pipeline_state.q[2:]

        smushed = jp.concatenate((qpos, qvel))
        assert smushed.shape == OBSERVATION_SHAPE, f"{smushed.shape} =/= {OBSERVATION_SHAPE}"
        return smushed


envs.register_environment("eye", Eye)

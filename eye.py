#!/usr/bin/env python3


# Based on <https://github.com/google/brax/blob/759256a27ec495c8307bcd141e0798c5f090a1df/brax/envs/ant.py>.


from brax import base, envs, math
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from etils import epath
import jax
from jax import numpy as jp, debug
import mujoco


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

    The agent takes a 6-element vector for actions.

    The action space is a continuous `(action, action, action, action, action, action)`,
    all in `[-1, 1]`, where `action` represents the positions of a hinge joint.

    | Num | Action                                                       | Control Min | Control Max | Name (in corresponding config)   | Joint | Unit        |
    |-----|--------------------------------------------------------------|-------------|-------------|----------------------------------|-------|-------------|
    |  0  | Position of the rotor between the torso and front left hip   | -pi (360 d) | pi (360 d)  | hip_1 (front_left_leg)           | hinge | angle (rad) |
    |  1  | Position of the rotor between the front left two links       | -pi (360 d) | pi (360 d)  | ankle_1 (front_left_leg)         | hinge | angle (rad) |
    |  2  | Position of the rotor between the torso and front right hip  | -pi (360 d) | pi (360 d)  | hip_2 (front_right_leg)          | hinge | angle (rad) |
    |  3  | Position of the rotor between the front right two links      | -pi (360 d) | pi (360 d)  | ankle_2 (front_right_leg)        | hinge | angle (rad) |
    |  4  | Position of the rotor between the torso and back left hip    | -pi (360 d) | pi (360 d)  | hip_3 (back_leg) (360 d)         | hinge | angle (rad) |
    |  5  | Position of the rotor between the back left two links        | -pi (360 d) | pi (360 d)  | ankle_3 (back_leg)               | hinge | angle (rad) |

    ### Observation Space

    The state space consists of positional values of different body parts of the
    ant, followed by the velocities of those individual parts (their derivatives)
    with all the positions ordered before all the velocities.

    The observation is a `ndarray` with shape `(29,)` where the elements correspond to the following:

    | Num | Observation                                                  | Name (in corresponding config)   | Joint | Unit                     |
    |-----|--------------------------------------------------------------|----------------------------------|-------|--------------------------|
    |  0  | z-coordinate of the torso (centre)                           | torso                            | free  | position (m)             |
    |  1  | w-orientation of the torso (centre)                          | torso                            | free  | angle (rad)              |
    |  2  | x-orientation of the torso (centre)                          | torso                            | free  | angle (rad)              |
    |  3  | y-orientation of the torso (centre)                          | torso                            | free  | angle (rad)              |
    |  4  | z-orientation of the torso (centre)                          | torso                            | free  | angle (rad)              |
    |  5  | angle between torso and first link on front left             | hip_1 (front_left_leg)           | hinge | angle (rad)              |
    |  6  | angle between the two links on the front left                | ankle_1 (front_left_leg)         | hinge | angle (rad)              |
    |  7  | angle between torso and first link on front right            | hip_2 (front_right_leg)          | hinge | angle (rad)              |
    |  8  | angle between the two links on the front right               | ankle_2 (front_right_leg)        | hinge | angle (rad)              |
    |  9  | angle between torso and first link on back left              | hip_3 (back_leg)                 | hinge | angle (rad)              |
    | 10  | angle between the two links on the back left                 | ankle_3 (back_leg)               | hinge | angle (rad)              |
    | 11  | x-coordinate velocity of the torso                           | torso                            | free  | velocity (m/s)           |
    | 12  | y-coordinate velocity of the torso                           | torso                            | free  | velocity (m/s)           |
    | 13  | z-coordinate velocity of the torso                           | torso                            | free  | velocity (m/s)           |
    | 14  | x-coordinate angular velocity of the torso                   | torso                            | free  | angular velocity (rad/s) |
    | 15  | y-coordinate angular velocity of the torso                   | torso                            | free  | angular velocity (rad/s) |
    | 16  | z-coordinate angular velocity of the torso                   | torso                            | free  | angular velocity (rad/s) |
    | 17  | angular velocity of angle between torso and front left link  | hip_1 (front_left_leg)           | hinge | angle (rad)              |
    | 18  | angular velocity of the angle between front left links       | ankle_1 (front_left_leg)         | hinge | angle (rad)              |
    | 19  | angular velocity of angle between torso and front right link | hip_2 (front_right_leg)          | hinge | angle (rad)              |
    | 20  | angular velocity of the angle between front right links      | ankle_2 (front_right_leg)        | hinge | angle (rad)              |
    | 21  | angular velocity of angle between torso and back left link   | hip_3 (back_leg)                 | hinge | angle (rad)              |
    | 22  | angular velocity of the angle between back left links        | ankle_3 (back_leg)               | hinge | angle (rad)              |
    | 23  | Last requested action between the torso and front left hip   | hip_1 (front_left_leg)           | hinge | angle (rad)              |
    | 24  | Last requested action between the front left two links       | ankle_1 (front_left_leg)         | hinge | angle (rad)              |
    | 25  | Last requested action between the torso and front right hip  | hip_2 (front_right_leg)          | hinge | angle (rad)              |
    | 26  | Last requested action between the front right two links      | ankle_2 (front_right_leg)        | hinge | angle (rad)              |
    | 27  | Last requested action between the torso and back left hip    | hip_3 (back_leg) (360 d)         | hinge | angle (rad)              |
    | 28  | Last requested action between the back left two links        | ankle_3 (back_leg)               | hinge | angle (rad)              |

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
    - *reward_ctrl*: A negative reward for penalising the ant if it takes actions
      that are too large. It is measured as *coefficient **x**
      sum(action<sup>2</sup>)* where *coefficient* is a parameter set for the
      control and has a default value of 0.5.
    - *contact_cost*: A negative reward for penalising the ant if the external
      contact force is too large. It is calculated *0.5 * 0.001 *
      sum(clip(external contact force to [-1,1])<sup>2</sup>)*.

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
        ctrl_cost_weight=0.5,
        use_contact_forces=False,
        contact_cost_weight=5e-4,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.2, None),
        contact_force_range=(-1.0, 1.0),
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

        self._ctrl_cost_weight = ctrl_cost_weight
        self._use_contact_forces = use_contact_forces
        self._contact_cost_weight = contact_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._contact_force_range = contact_force_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        if self._use_contact_forces:
            raise NotImplementedError("use_contact_forces not implemented.")

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2 = jax.random.split(rng, 3)

        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        q = self.sys.init_q + jax.random.uniform(
            rng1, (self.sys.q_size(),), minval=low, maxval=hi
        )
        qd = hi * jax.random.normal(rng2, (self.sys.qd_size(),))

        pipeline_state = self.pipeline_init(q, qd)
        last_action = jp.zeros((6,))
        obs = self._get_obs(pipeline_state, last_action)

        reward, done, zero = jp.zeros(3)
        metrics = {
            "reward_forward": zero,
            "reward_orientation": zero,
            "reward_survive": zero,
            "reward_ctrl": zero,
            "reward_contact": zero,
            "x_position": zero,
            "y_position": zero,
            "distance_from_origin": zero,
            "x_velocity": zero,
            "y_velocity": zero,
        }
        info = {
            "last_action": last_action,
        }
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
        forward_reward = velocity[0]

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
        last_action = state.info["last_action"]
        ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.abs(action - last_action))
        contact_cost = 0.0

        obs = self._get_obs(pipeline_state, last_action)
        reward = (
            forward_reward
            + orientation_reward
            + healthy_reward
            - ctrl_cost
            - contact_cost
        )
        done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0
        state.metrics.update(
            reward_forward=forward_reward,
            reward_orientation=orientation_reward,
            reward_survive=healthy_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            x_position=pipeline_state.x.pos[0, 0],
            y_position=pipeline_state.x.pos[0, 1],
            distance_from_origin=math.safe_norm(pipeline_state.x.pos[0]),
            x_velocity=velocity[0],
            y_velocity=velocity[1],
        )
        state.info.update(last_action=action)
        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )

    def _get_obs(self, pipeline_state: base.State, last_action) -> jax.Array:
        """Observe eye body position and velocities."""
        qpos = pipeline_state.q
        qvel = pipeline_state.qd

        if self._exclude_current_positions_from_observation:
            qpos = pipeline_state.q[2:]

        smushed = jp.concatenate((qpos, qvel, last_action))
        assert smushed.shape == (29,)
        return smushed


envs.register_environment("eye", Eye)

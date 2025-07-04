#!/usr/bin/env python3


# Based on <https://github.com/google/brax/blob/759256a27ec495c8307bcd141e0798c5f090a1df/brax/envs/ant.py>.


from beartype import beartype
from brax import base, envs, math
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from etils import epath
import jax
from jax import debug, numpy as jp, random as jr
from jaxtyping import jaxtyped, Array, Float, UInt
from mujoco import mj_name2id, mjtObj


N_LEGS = 3
SERVOS_PER_LEG = 3
MEMORY_SIZE = 32
ACTION_PERIOD = 0.02  # 20ms
ACTION_SIZE = SERVOS_PER_LEG * N_LEGS + MEMORY_SIZE
N_IMU_AXES = 6
N_FSR = N_LEGS
OBSERVATION_SIZE = N_FSR + N_IMU_AXES + MEMORY_SIZE


IDEAL_VELOCITY = 0.5  # m/s


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
    | etc | Recurrent input to the next iteration of this network        |     n/a     |     n/a     | n/a                            |  n/a  |     n/a     |

    ### Observation Space

    The state space consists of positional values of different body parts of the
    ant, followed by the velocities of those individual parts (their derivatives)
    with all the positions ordered before all the velocities.

    The observation is an array whose elements correspond to the following:

    | Num | Observation                                   | Name (in corresponding config) | Joint | Unit                     |
    |-----|-----------------------------------------------|--------------------------------|-------|--------------------------|
    |  0  | z-coordinate of the torso (centre)            | torso                          | free  | position (m)             |
    |  1  | w-orientation of the torso (centre)           | torso                          | free  | angle (rad)              |
    |  2  | x-orientation of the torso (centre)           | torso                          | free  | angle (rad)              |
    |  3  | y-orientation of the torso (centre)           | torso                          | free  | angle (rad)              |
    |  4  | z-orientation of the torso (centre)           | torso                          | free  | angle (rad)              |
    |  5  | x-coordinate velocity of the torso            | torso                          | free  | velocity (m/s)           |
    |  6  | y-coordinate velocity of the torso            | torso                          | free  | velocity (m/s)           |
    |  7  | z-coordinate velocity of the torso            | torso                          | free  | velocity (m/s)           |
    |  8  | x-coordinate angular velocity of the torso    | torso                          | free  | angular velocity (rad/s) |
    |  9  | y-coordinate angular velocity of the torso    | torso                          | free  | angular velocity (rad/s) |
    | 10  | z-coordinate angular velocity of the torso    | torso                          | free  | angular velocity (rad/s) |
    | etc | Recurrent input from our last iteration       | n/a                            |  n/a  | n/a                      |

    The (x,y,z) coordinates are translational DOFs while the orientations are
    rotational DOFs expressed as quaternions.

    ### Rewards

    The reward consists of three parts:

    - *reward_forward*: Reward for moving forward, measured as
      *(x-coordinate before action - x-coordinate after action)/dt*. *dt* is the
      time between actions - the default *dt = 0.05*. This reward would be
      positive if the ant moves forward (right) desired.
    - *reward_orientation*: Reward for facing forward, not turning, flipping, etc.
    - *reward_torque*: Negative reward for total torque used.
    - *reward_slip*: Negative reward for foot slipping (velocity parallel to floor plane if very close).
    - *reward_drift*: Negative reward for moving sideways, based on position, not velocity.
    - *reward_contact*: Reward per timestep for at least half the feet touching the ground.

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

    def __init__(
        self,
        slip_cost_weight=50.0,
        drift_cost_weight=0.01,
        contact_reward_weight=100.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.2, None),
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=True,
        backend="mjx",
        n_frames=None,
        **kwargs,
    ):
        path = "./eye.xml"
        sys = mjcf.load(path)

        if n_frames is None:
            n_frames = ACTION_PERIOD / sys.opt.timestep

        super().__init__(sys=sys, backend=backend, n_frames=n_frames, **kwargs)

        self._slip_cost_weight = slip_cost_weight
        self._drift_cost_weight = drift_cost_weight
        self._contact_reward_weight = contact_reward_weight
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        foot_ids = []
        i = 0
        while True:
            id = mj_name2id(sys.mj_model, mjtObj.mjOBJ_GEOM.value, f"foot{i}")
            if id == -1:
                assert i > 0, "No feet found!"
                break
            foot_ids.append(id)
            i += 1
        self._foot_ids = tuple(foot_ids)
        foot_radius = sys.mj_model.geom_size[foot_ids[0], 0]
        for id in self._foot_ids:
            assert (
                sys.mj_model.geom_size[id, 0] == foot_radius
            ), f"{sys.mj_model.geom_size[id]} == {foot_radius}"
        self._foot_radius = foot_radius

    @jaxtyped(typechecker=beartype)
    def reset(self, rng: UInt[Array, "2"]) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2 = jr.split(rng, 3)

        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        q = self.sys.init_q
        # q += jr.uniform(
        #     rng1, (self.sys.q_size(),), minval=low, maxval=hi
        # )
        qd = hi * jr.normal(rng2, (self.sys.qd_size(),))

        memory = jp.zeros((MEMORY_SIZE,))
        pipeline_state = self.pipeline_init(q, qd)
        obs = self._get_obs(pipeline_state, memory)

        reward, done, zero = jp.zeros(3)
        metrics = {
            "reward_forward": zero,
            "reward_orientation": zero,
            "reward_contact": zero,
            "reward_slip": zero,
            "reward_drift": zero,
            "x_position": zero,
            "y_position": zero,
            "distance_from_origin": zero,
            "x_velocity": zero,
            "y_velocity": zero,
        }
        info = {"memory": memory, "step": jp.zeros((), dtype=jp.uint32)}
        return State(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=reward,
            done=done,
            metrics=metrics,
            info=info,
        )

    @jaxtyped(typechecker=beartype)
    def step(self, state: State, action: Float[Array, "ACTION_SIZE"]) -> State:
        """Run one timestep of the environment's dynamics."""
        pipeline_state0 = state.pipeline_state
        assert pipeline_state0 is not None
        assert action.shape == (ACTION_SIZE,)
        action, memory = action[:-MEMORY_SIZE], action[-MEMORY_SIZE:]
        pipeline_state = self.pipeline_step(pipeline_state0, action)

        velocity = (pipeline_state.x.pos[0] - pipeline_state0.x.pos[0]) / self.dt
        forward_reward = -jp.sum(
            jp.abs(velocity - jp.asarray([IDEAL_VELOCITY, 0.0, 0.0]))
        )

        roll, pitch, yaw = math.quat_to_euler(pipeline_state.q[3:7])
        orientation_reward = -jp.abs(yaw)

        min_z, max_z = self._healthy_z_range
        zpos = pipeline_state.x.pos[0, 2]
        is_healthy = jp.ones(zpos.shape)
        if min_z is not None:
            is_healthy = jp.where(zpos < min_z, 0.0, is_healthy)
        if max_z is not None:
            is_healthy = jp.where(zpos > max_z, 0.0, is_healthy)

        active_contact = pipeline_state.contact.dist <= 0  # : bool[N_CONTACT_PAIRS]
        prev_contact = pipeline_state0.contact.dist <= 0  # : bool[N_CONTACT_PAIRS]
        persistent_contact = jp.logical_and(
            active_contact, prev_contact
        )  # : bool[N_CONTACT_PAIRS]
        foot_in_each_contact_pair = jp.asarray(
            [
                jp.any(pipeline_state.contact.geom == id, axis=-1)
                for id in self._foot_ids
            ]
        )  # : bool[N_FEET, N_CONTACT_PAIRS]
        foot_contacting_anything = jp.any(
            foot_in_each_contact_pair * persistent_contact[jp.newaxis], axis=-1
        )  # : bool[N_FEET]
        foot_pos = pipeline_state.geom_xpos[self._foot_ids,]
        prev_foot_pos = pipeline_state0.geom_xpos[self._foot_ids,]
        foot_speed_along_floor = (
            jp.sqrt(
                jp.sum(jp.square(foot_pos[..., :2] - prev_foot_pos[..., :2]), axis=-1)
            )
            / self.dt
        )
        slip_cost = jp.sum(
            self._slip_cost_weight * foot_contacting_anything * foot_speed_along_floor
        )

        drift_cost = self._drift_cost_weight * jp.abs(pipeline_state.x.pos[0, ..., 1])

        contact_reward = self._contact_reward_weight * (
            jp.sum(1.0 * foot_contacting_anything) >= (0.5 * N_LEGS)
        )

        obs = self._get_obs(pipeline_state, memory)
        reward = (
            forward_reward
            + orientation_reward
            + contact_reward
            - slip_cost
            - drift_cost
        )
        done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0
        state.metrics.update(
            reward_forward=forward_reward,
            reward_orientation=orientation_reward,
            reward_contact=contact_reward,
            reward_slip=-slip_cost,
            reward_drift=-drift_cost,
            x_position=pipeline_state.x.pos[0, 0],
            y_position=pipeline_state.x.pos[0, 1],
            distance_from_origin=math.safe_norm(pipeline_state.x.pos[0]),
            x_velocity=velocity[0],
            y_velocity=velocity[1],
        )
        state.info.update(
            memory=memory,
            step=(state.info["step"] + 1),
        )
        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )

    @jaxtyped(typechecker=beartype)
    def _get_obs(
        self, pipeline_state: base.State, memory: Float[Array, "MEMORY_SIZE"]
    ) -> Float[Array, "OBSERVATION_SIZE"]:
        obs = jp.concatenate((pipeline_state.sensordata, memory))
        assert obs.shape == (
            OBSERVATION_SIZE,
        ), f"{obs.shape} =/= {(OBSERVATION_SIZE,)}"
        return obs

    @property
    @jaxtyped(typechecker=beartype)
    def action_size(self) -> int:
        return ACTION_SIZE


envs.register_environment("eye", Eye)

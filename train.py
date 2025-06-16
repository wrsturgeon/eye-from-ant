#!/usr/bin/env python3


import eye
from datetime import datetime
import brax
from brax import envs
from brax.io import image, json, model
from brax.training.agents.ppo import networks as ppo_networks, train as ppo
from brax.training.agents.sac import train as sac
from cv2 import VideoWriter, VideoWriter_fourcc as fourcc
import flax
import functools
import jax
from jax import jit, numpy as jp, random as jr
import matplotlib.pyplot as plt
from os import makedirs
from typing import Any, Callable, Dict, Tuple


env_name = "eye"
backend = "mjx"

env = envs.get_environment(env_name=env_name, backend=backend)
state = jax.jit(env.reset)(rng=jax.random.PRNGKey(seed=0))


TRAINING_SECONDS_TOTAL = 1_000_000
EPISODE_SECONDS = 5


N_ACTIONS_TOTAL = TRAINING_SECONDS_TOTAL / eye.ACTION_PERIOD
N_ACTIONS_PER_EPISODE = 5 / eye.ACTION_PERIOD


# with open("init.png", "wb") as f:
#     f.write(image.render(env.sys, [state.pipeline_state]))

# # Training
#
# Brax provides out of the box the following training algorithms:
#
# * [Proximal policy optimization](https://github.com/google/brax/blob/main/brax/training/agents/ppo/train.py)
# * [Soft actor-critic](https://github.com/google/brax/blob/main/brax/training/agents/sac/train.py)
# * [Evolutionary strategy](https://github.com/google/brax/blob/main/brax/training/agents/es/train.py)
# * [Analytic policy gradients](https://github.com/google/brax/blob/main/brax/training/agents/apg/train.py)
# * [Augmented random search](https://github.com/google/brax/blob/main/brax/training/agents/ars/train.py)
#
# Trainers take as input an environment function and some hyperparameters, and return an inference function to operate the environment.
#
# # Training
#
# Let's train the eye policy using the `mjx` backend with PPO.

# @title Training

# We determined some reasonable hyperparameters offline and share them here.
train_fn = functools.partial(
    ppo.train,
    num_timesteps=N_ACTIONS_TOTAL,
    num_evals=10,
    reward_scaling=10,
    episode_length=N_ACTIONS_PER_EPISODE,
    normalize_observations=True,
    action_repeat=1,
    unroll_length=5,
    num_minibatches=32,
    num_updates_per_batch=4,
    discounting=0.97,
    learning_rate=3e-3,  # 3e-4,
    entropy_cost=1e-2,
    num_envs=4096,
    batch_size=2048,
    seed=1,
)


max_y = 8_000
min_y = 0

xdata, ydata = [], []
times = [datetime.now()]


def progress(num_steps, metrics):
    times.append(datetime.now())
    xdata.append(num_steps)
    ydata.append(metrics["eval/episode_reward"])

    plt.xlim([0, train_fn.keywords["num_timesteps"]])
    plt.ylim([min_y, max_y])
    plt.xlabel("# environment steps")
    plt.ylabel("reward per episode")
    plt.plot(xdata, ydata)
    plt.savefig(f"reward_after_{num_steps}_steps.png")

    print()
    print("    Metrics:")
    for k, v in metrics.items():
        print(f"        {k}: {v}")


EVAL_ENV = envs.get_environment(env_name=env_name, backend=backend)
JIT_RESET = jit(EVAL_ENV.reset)
JIT_STEP = jit(EVAL_ENV.step)
RENDER = EVAL_ENV.render


N_SNAPSHOTS = 5
N_STEPS_PER_SNAPSHOT = int(5 / EVAL_ENV.dt)


def snapshot(
    current_step: int,
    make_policy: Callable,
    params,
) -> None:
    policy = make_policy(params)
    snapshot_folder = f"snapshot_step_{current_step}"
    makedirs(snapshot_folder, exist_ok=True)
    keys = jr.split(jr.PRNGKey(42), N_SNAPSHOTS)
    for sample_number, key in enumerate(keys):
        key, *keys = jr.split(key, 1 + N_STEPS_PER_SNAPSHOT)
        state = JIT_RESET(key)
        rollout = []
        for i, key in enumerate(keys):
            result = policy(state.obs, key)
            ctrl, _ = result
            state = JIT_STEP(state, ctrl)
            rollout.append(state.pipeline_state)
            if state.done:
                break
        if len(rollout) == 0:
            makedirs(f"{snapshot_folder}/sample_{sample_number}_was_empty")
        else:
            video = RENDER(rollout)
            assert len(video) > 0, "Empty video!"
            writer = VideoWriter(
                f"{snapshot_folder}/sample_{sample_number}.mp4",
                fourcc(*"mp4v"),
                int(1 / EVAL_ENV.dt),
                video[0].shape[:-1][::-1],
            )
            for frame in video:
                writer.write(frame[..., ::-1])
            writer.release()


make_inference_fn, params, _ = train_fn(
    environment=env, progress_fn=progress, policy_params_fn=snapshot
)

print(f"time to jit: {times[1] - times[0]}")
print(f"time to train: {times[-1] - times[1]}")

# The trainers return an inference function, parameters, and the final set of metrics gathered during evaluation.
#
# # Saving and Loading Policies
#
# Brax can save and load trained policies:

model.save_params("/tmp/params", params)
params = model.load_params("/tmp/params")
inference_fn = make_inference_fn(params)

# The trainers return an inference function, parameters, and the final set of metrics gathered during evaluation.
#
# # Saving and Loading Policies
#
# Brax can save and load trained policies:

# @title Visualizing a trajectory of the learned inference function

# create an env with auto-reset
env = envs.create(env_name=env_name, backend=backend)

jit_env_reset = jax.jit(env.reset)
jit_env_step = jax.jit(env.step)
jit_inference_fn = jax.jit(inference_fn)

rollout = []
rng = jax.random.PRNGKey(seed=1)
state = jit_env_reset(rng=rng)
for _ in range(1000):
    rollout.append(state.pipeline_state)
    act_rng, rng = jax.random.split(rng)
    act, _ = jit_inference_fn(state.obs, act_rng)
    state = jit_env_step(state, act)

with open("trained.mp4", "wb") as f:
    frames = image.render_array(env.sys.tree_replace({"opt.timestep": env.dt}), rollout)
    writer = VideoWriter(
        f"trained.mp4",
        fourcc(*"mp4v"),
        int(1 / env.dt),
        frames[0].shape[:-1][::-1],
    )
    for frame in frames:
        writer.write(frame[..., ::-1])
    writer.release()

# 🙌 See you soon!


from evolve_car.simulator.core.carla_env import CarlaEnv
from evolve_car.ray.naive_cnn import NaiveCNN

from ray.tune.registry import get_trainable_cls
from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment,
)
from ray.rllib.core.rl_module.rl_module import RLModuleSpec


def main():
    parser = add_rllib_example_script_args()
    parser.add_argument(
        "--port",
        type=int,
        default=3005,
        help="carla env port",
    )

    args = parser.parse_args()
    config = (
        get_trainable_cls('PPO')
        .get_default_config()
        .environment(
            CarlaEnv,
            env_config={"port": 4019},
        )
        # Only allow to use on runner right now, but could be run in parallel in the near future.
        .env_runners(num_env_runners=1)
        .learners(
            # How many Learner workers do we need? If you have more than 1 GPU,
            # set this parameter to the number of GPUs available.
            num_learners=1,

            # How many GPUs does each Learner need? If you have more than 1 GPU or only
            # one Learner, you should set this to 1, otherwise, set this to some
            # fraction.
            num_gpus_per_learner=0.4,
        )

        .training(
            minibatch_size=2,
            train_batch_size=1000,
            num_epochs=6,
            lr=0.01,
        )
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=NaiveCNN,
                model_config={
                    "conv_filters": [
                        # num filters, kernel wxh, stride wxh, padding type
                        [4, 3, 2, "same"],
                        [8, 3, 2, "same"],
                        [16, 3, 2, "same"],
                        [32, 3, 2, "same"],
                        [16, 16, 1, "valid"],
                    ],
                },
            ),
        )
        .resources(
            num_cpus_per_worker=4,

        )
    )

    from ray import train, tune
    tuner = tune.Tuner(
        "PPO",
        param_space=config,
        run_config=train.RunConfig(
            stop={"env_runners/episode_return_mean": 1000.0},
        ),
    )

    results = tuner.fit()
    best_result = results.get_best_result(
        metric="env_runners/episode_return_mean", mode="max"
    )

    # Get the best checkpoint corresponding to the best result.
    best_checkpoint = best_result.checkpoint
    print("best_checkpoint:", best_checkpoint.path)


def inference():
    from ray.rllib.algorithms.algorithm import Algorithm

    path_to_checkpoint = '.log/PPO_2025-01-21_09-51-55/driver_artifacts/PPO_CarlaEnv_4b181_000000_2'
    restore_ppo = Algorithm.from_checkpoint(path_to_checkpoint)
    print(restore_ppo)


if __name__ == "__main__":
    if False:
        inference()
    else:
        main()

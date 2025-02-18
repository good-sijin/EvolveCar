
from evolve_car.simulator.core.carla_env import CarlaEnv
from evolve_car.ray.naive_cnn import NaiveCNN

from ray.tune.registry import get_trainable_cls
from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment,
)
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from ray import train, tune
import os


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
        .env_runners(num_env_runners=1,
                     sample_timeout_s=100,
                     )
        .learners(
            # How many Learner workers do we need? If you have more than 1 GPU,
            # set this parameter to the number of GPUs available.
            num_learners=1,

            # How many GPUs does each Learner need? If you have more than 1 GPU or only
            # one Learner, you should set this to 1, otherwise, set this to some
            # fraction.
            num_gpus_per_learner=1,
        )

        .training(
            minibatch_size=1,
            train_batch_size=64,
            num_epochs=32,
            lr=0.01,
        )
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=NaiveCNN,
                model_config={
                    "conv_filters": [
                        # num filters, kernel wxh, stride wxh, padding
                        [4, 7, 2, (3, 3)],  # 128
                        [4, 3, 2, (1, 1)],  # 64
                        [8, 3, 2, (1, 1)],  # 32
                        [16, 3, 2, (1, 1)],  # 16
                        [16, 3, 2, (1, 1)],  # 8
                        [32, 3, 2, (1, 1)],  # 4
                        [32, 4, 1, 0],  # 4
                    ],
                },
            ),
        )
        .resources(
            num_cpus_per_worker=4,
        )
    )

    def train_with_tune():
        cur_path = os.getcwd()
        tuner = tune.Tuner(
            "PPO",
            param_space=config,
            run_config=train.RunConfig(
                stop={"training_iteration": 40},
                storage_path=f"{cur_path}/ray_results"
                checkpoint_config=train.CheckpointConfig(
                    checkpoint_frequency=10, checkpoint_at_end=True),
            ),
            tune_config=tune.TuneConfig(
                reuse_actors=True,
            ),
        )
        results = tuner.fit()
        print(results)

    def restore_with_tune():
        experiment_path = '/root/autodl-fs/code/EvolveCar/ray_results/PPO_2025-02-13_07-33-39/'
        restored_tuner = tune.Tuner.restore(
            experiment_path, trainable=get_trainable_cls('PPO'))
        result_grid = restored_tuner.get_results()
        best_result = result_grid.get_best_result(
            metric="env_runners/episode_return_mean", mode="max"
        )
        best_checkpoint = best_result.checkpoint
        print(best_checkpoint)

    train_with_tune()

    # restore_with_tune()


if __name__ == "__main__":
    main()


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
    base_config = (
        get_trainable_cls('PPO')
        .get_default_config()
        .environment(
            CarlaEnv,
            env_config={"port": 4014},
        )
        # Only allow to use on runner right now, but could be run in parallel in the near future.
        .env_runners(num_env_runners=1)
        .training(
            minibatch_size=2,
            train_batch_size=1000,
            num_epochs=6,
            lr=0.0009,
            vf_loss_coeff=0.001,
            entropy_coeff=0.0,
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
    )

    run_rllib_example_script_experiment(base_config, args)


if __name__ == "__main__":
    main()

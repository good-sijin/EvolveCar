from typing import Any, Dict, Optional

from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.models.torch.misc import (
    normc_initializer,
    same_padding,
    valid_padding,
)
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import TensorType

torch, nn = try_import_torch()


class NaiveCNN(TorchRLModule, ValueFunctionAPI):

    @override(TorchRLModule)
    def setup(self):
        # Get the CNN stack config from our RLModuleConfig's (self.config)
        # `model_config` property:
        conv_filters = self.model_config.get("conv_filters")
        # Default CNN stack with 3 layers:
        if conv_filters is None:
            conv_filters = [
                # num filters, kernel wxh, stride wxh, padding type
                [16, 4, 2, "same"],
                [32, 4, 2, "same"],
            ]

        # Build the CNN layers.
        layers = []

        # Add user-specified hidden convolutional layers first
        width, height, in_depth = self.observation_space['birdeye'].shape
        in_size = [width, height]
        for filter_specs in conv_filters:
            if len(filter_specs) == 4:
                out_depth, kernel_size, strides, padding = filter_specs
            else:
                out_depth, kernel_size, strides = filter_specs
                padding = "same"

            # Pad like in tensorflow's SAME mode.
            if padding == "same":
                padding_size, out_size = same_padding(
                    in_size, kernel_size, strides)
                layers.append(nn.ZeroPad2d(padding_size))
            # No actual padding is performed for "valid" mode, but we will still
            # compute the output size (input for the next layer).
            else:
                out_size = valid_padding(in_size, kernel_size, strides)

            layer = nn.Conv2d(in_depth, out_depth,
                              kernel_size, strides, bias=True)
            # Initialize CNN layer kernel and bias.
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
            layers.append(layer)
            # Activation.
            layers.append(nn.ReLU())

            in_size = out_size
            in_depth = out_depth

        self._base_cnn_stack = nn.Sequential(*layers)

        # Add the final CNN 1x1 layer with num_filters == num_actions to be reshaped to
        # yield the logits (no flattening, no additional linear layers required).
        action_n = self.action_space.shape[0]
        _final_conv = nn.Conv2d(in_depth, action_n, 1, 1, bias=True)
        nn.init.xavier_uniform_(_final_conv.weight)
        nn.init.zeros_(_final_conv.bias)
        self._logits = nn.Sequential(
            nn.ZeroPad2d(same_padding(in_size, 1, 1)[0]), _final_conv
        )

        self._values = nn.Linear(in_depth, 1)
        # Mimick old API stack behavior of initializing the value function with `normc`
        # std=0.01.
        normc_initializer(0.01)(self._values.weight)

    @override(TorchRLModule)
    def _forward(self, batch, **kwargs):
        # Compute the basic 1D feature tensor (inputs to policy- and value-heads).
        _, logits = self._compute_embeddings_and_logits(batch)
        # Return features and logits as ACTION_DIST_INPUTS (categorical distribution).
        return {
            Columns.ACTION_DIST_INPUTS: logits,
        }

    @override(TorchRLModule)
    def _forward_train(self, batch, **kwargs):
        # Compute the basic 1D feature tensor (inputs to policy- and value-heads).
        embeddings, logits = self._compute_embeddings_and_logits(batch)
        # Return features and logits as ACTION_DIST_INPUTS (categorical distribution).
        return {
            Columns.ACTION_DIST_INPUTS: logits,
            Columns.EMBEDDINGS: embeddings,
        }

    # We implement this RLModule as a ValueFunctionAPI RLModule, so it can be used
    # by value-based methods like PPO or IMPALA.
    @override(ValueFunctionAPI)
    def compute_values(
        self,
        batch: Dict[str, Any],
        embeddings: Optional[Any] = None,
    ) -> TensorType:
        # Features not provided -> We need to compute them first.
        if embeddings is None:
            obs = batch[Columns.OBS]['birdeye']
            obs = obs/255.
            embeddings = self._base_cnn_stack(obs.permute(0, 3, 1, 2))
            embeddings = torch.squeeze(embeddings, dim=[-1, -2])
        return self._values(embeddings).squeeze(-1)

    def _compute_embeddings_and_logits(self, batch):
        obs = batch[Columns.OBS]['birdeye']
        obs = obs.permute(0, 3, 1, 2)
        # Normalize
        obs = obs/255.
        embeddings = self._base_cnn_stack(obs)
        logits = self._logits(embeddings)
        return (
            torch.squeeze(embeddings, dim=[-1, -2]),
            torch.squeeze(logits, dim=[-1, -2]),
        )


def main():
    from gymnasium.spaces import Box, Dict
    import numpy as np
    obs_size = 256
    # Obsevation space for the car are camera, lidar, bireye, car-state, etc.
    observation_space_dict = {
        'birdeye': Box(low=0, high=255, shape=(obs_size, obs_size, 3), dtype=np.uint8),
    }

    observation_space = Dict(observation_space_dict)

    my_net = NaiveCNN(observation_space=observation_space,
                      action_space=Box(-1, 1, (2,), dtype=np.float32),
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
                      )

    B, w, h, c = (1, 256, 256, 3)
    data = torch.from_numpy(
        np.random.random_sample(size=(B, w, h, c)).astype(np.float32)
    )
    my_net.forward_train({"obs": {"birdeye": data}})

    num_all_params = sum(int(np.prod(p.size())) for p in my_net.parameters())
    print(f"num params = {num_all_params}")


if __name__ == "__main__":
    main()

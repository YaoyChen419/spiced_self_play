"""SPiCED observation encoder for the unchanged FastTD3 MLP heads."""

from pufferlib.ocean.torch import Drive


class DriveEncoder(Drive):
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        # Keep the original observation modules; FastTD3 supplies its own heads.
        del self.actor
        del self.value_fn

    def forward(self, observations):
        return self.encode_observations(observations)

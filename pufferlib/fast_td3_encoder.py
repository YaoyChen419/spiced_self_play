"""Thin FastTD3 adapter for SPiCED's native Drive encoder."""

from pufferlib.ocean.torch import Drive


class DriveEncoder(Drive):
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        del self.actor
        del self.value_fn

    def forward(self, observations):
        return self.encode_observations(observations)

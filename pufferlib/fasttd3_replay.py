"""Device-resident episodic replay; CPU metadata follows the native collector."""
import numpy as np
import torch


class DeviceEpisodeReplay:
    def __init__(self, agents, episode_length, obs_dim, act_dim, capacity, seed, device='cpu'):
        self.device = torch.device(device)
        self.horizon = episode_length
        self.slots = capacity // episode_length
        if self.slots < agents:
            raise ValueError('replay_capacity must hold one completed episode per agent')
        self.capacity = self.slots * episode_length
        self.size = self.cursor = 0
        self.lengths = np.zeros(agents, np.int64)
        self.saved_lengths = np.zeros(self.slots, np.int64)
        self.device_lengths = torch.zeros(self.slots, dtype=torch.int64, device=self.device)
        self.ends = torch.zeros_like(self.device_lengths)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        shapes = ((episode_length + 1, obs_dim), (episode_length, act_dim),
                  (episode_length,), (episode_length,))
        dtypes = (torch.float32, torch.float32, torch.float32, torch.bool)
        needed = sum((agents + self.slots) * int(np.prod(s)) *
                     torch.empty((), dtype=d).element_size() for s, d in zip(shapes, dtypes))
        if self.device.type == 'cuda':
            free, _ = torch.cuda.mem_get_info(self.device)
            if needed > free * 0.6:
                raise ValueError(f'Replay needs {needed / 2**30:.2f} GiB, free {free / 2**30:.2f} GiB; '
                                 'reduce fasttd3.replay_capacity (environment concurrency is unchanged)')
        self.live = [torch.zeros((agents, *s), dtype=d, device=self.device)
                     for s, d in zip(shapes, dtypes)]
        self.saved = [torch.zeros((self.slots, *s), dtype=d, device=self.device)
                      for s, d in zip(shapes, dtypes)]
        self.bytes = needed

    def add(self, ids, obs, actions, rewards, next_obs, terminals, boundaries):
        pos = self.lengths[ids]
        if np.any(pos >= self.horizon):
            raise RuntimeError('Episode exceeded configured length')
        ix = torch.as_tensor(ids, device=self.device)
        at = torch.as_tensor(pos, device=self.device)
        for storage, value in zip(self.live, (obs, actions, rewards, terminals)):
            storage[ix, at] = torch.as_tensor(value, dtype=storage.dtype, device=self.device)
        self.live[0][ix, at + 1] = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        self.lengths[ids] += 1
        finished = ids[boundaries]
        n = len(finished)
        if n:
            slots = (self.cursor + np.arange(n)) % self.slots
            dest = torch.as_tensor(slots, device=self.device)
            source = torch.as_tensor(finished, device=self.device)
            for saved, live in zip(self.saved, self.live):
                saved[dest] = live[source]
            self.saved_lengths[slots] = self.lengths[finished]
            self.device_lengths[dest] = torch.as_tensor(self.lengths[finished], device=self.device)
            self.ends = self.device_lengths.cumsum(0)
            self.lengths[finished] = 0
            self.cursor = (self.cursor + n) % self.slots
            self.size = int(self.saved_lengths.sum())

    def sample(self, sequences, horizon, device):
        if torch.device(device) != self.device:
            raise ValueError('Sample directly on the replay device')
        draws = torch.randint(self.size, (sequences,), device=self.device, generator=self.generator)
        rows = torch.searchsorted(self.ends, draws, right=True)
        lengths = self.device_lengths[rows]
        starts = draws - (self.ends[rows] - lengths)
        t = starts[:, None] + torch.arange(horizon, device=self.device)
        mask = t < lengths[:, None]
        at = t.clamp_max(self.horizon - 1)
        ot = (starts[:, None] + torch.arange(horizon + 1, device=self.device)).clamp_max(self.horizon)
        r = rows[:, None]
        return dict(obs=self.saved[0][r, ot], prefix=self.saved[0][rows, :self.horizon],
                    lengths=starts, actions=self.saved[1][r, at], rewards=self.saved[2][r, at],
                    terminals=self.saved[3][r, at], mask=mask,
                    weights=mask.float() / (t + 1).clamp_max(horizon))

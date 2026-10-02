# Visualizer

PufferDrive uses [Raylib](https://www.raylib.com/) for rendering the environment. Rendering is driven from Python using the torch policy directly. No separate binary or weight export is required.

## Dependencies

Headless rendering still requires an OpenGL context. On Ubuntu 22.04, install
Mesa's DRI/GLX libraries as well as ffmpeg, xvfb and xauth. Installing only ffmpeg
and xvfb is insufficient when the container cannot load `swrast_dri.so`.

Use the Tsinghua mirror for this installation without replacing system sources:

```bash
cat > /tmp/tsinghua-render.list <<'EOF'
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-updates main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-security main restricted universe multiverse
EOF
APT_ARGS=(-o Dir::Etc::sourcelist=/tmp/tsinghua-render.list -o Dir::Etc::sourceparts=-)
sudo apt-get "${APT_ARGS[@]}" update
sudo apt-get "${APT_ARGS[@]}" install -y ffmpeg xvfb xauth libgl1-mesa-dri libglx-mesa0 libgl1
```

## Render Modes

Configure `render_mode` in `pufferlib/config/ocean/drive.ini`:

```ini
; 0 = pop-up window (requires display)
; 1 = headless (pipes frames to ffmpeg, recommended for servers/training)
render_mode = 1
```

## Rendering once

```bash
LIBGL_ALWAYS_SOFTWARE=1 LIBGL_DRIVERS_PATH=/usr/lib/x86_64-linux-gnu/dri \
xvfb-run -a -s "-screen 0 1920x1080x24 +extension GLX" \
puffer eval puffer_drive --load-model-path /path/to/checkpoint.pt --env.render-mode 1
```

This runs a short rollout, calls `env.render()` each step, and finalizes the video on `vecenv.close()`. Use `render_mode` to determine whether the video shows up as a pop-up window, or whether it is stored as an mp4.

## View modes

Control what is rendered via the `view_mode` argument to `env.render()`:

```python
class RenderView(IntEnum):
    FULL_SIM_STATE = 0  # Top-down, fully observable
    BEV_AGENT_OBS  = 1  # Top-down, selected agent's observations only
    AGENT_PERSP    = 2  # Third-person perspective following selected agent

env.render(view_mode=RenderView.FULL_SIM_STATE, draw_traces=True, env_idx=0)
```

## Training-time evaluation

FastTD3 saves each due checkpoint before running periodic evaluation. With the
default 4096-agent received batch, `checkpoint_interval = 8000` saves every
32,768,000 agent steps (about five minutes at 110k SPS, excluding startup and
save/evaluation overhead).

Human-replay and self-play metric evaluations remain enabled, but their videos
are disabled by default so training does not require OpenGL. To enable videos
after configuring rendering dependencies, set both options in `drive.ini`:

```ini
[eval]
render_human_replay_eval = True
render_self_play_eval = True
```

These settings affect video output, not the policy loss or update schedule.
A native graphics assertion can terminate Python without raising a catchable
exception; `render_mode = 1` alone does not prevent this.

## Sharp edges

- **Raylib is not thread-safe.** If you create two separate render envs, always call `env1.close()` before calling `env2.render()`.
- Headless mode records at 1920 x 1080 in the current simulator.

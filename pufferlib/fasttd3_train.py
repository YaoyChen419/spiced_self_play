"""Driving collector and existing evaluation hooks for recurrent FastTD3."""
from copy import deepcopy
from collections import defaultdict
from pathlib import Path
import os
import random
import time
import shutil
import signal
from types import SimpleNamespace

import numpy as np
import torch

from pufferlib.fasttd3 import Learner
from pufferlib.fasttd3_replay import DeviceEpisodeReplay
from pufferlib import unroll_nested_dict


class VectorClock:
    """Count full-vector-equivalent steps without synchronizing async workers."""
    def __init__(self, agents):
        self.agents = agents
        self.slots = 0
        self.steps = 0

    def advance(self, slots):
        self.slots += slots
        current = self.slots // self.agents
        due = range(self.steps, current)
        self.steps = current
        return due


def print_dashboard(args, learner, utilization, profile, logs):
    """Use the platform dashboard itself; only the learner's loss fields differ."""
    from pufferlib.pufferl import PuffeRL
    view = SimpleNamespace(config={**args['train'], 'env': args['env_name']},
        utilization=utilization, profile=profile, sps=logs['SPS'],
        global_step=logs['agent_steps'], epoch=logs['epoch'], uptime=logs['uptime'],
        model_size=sum(p.numel() for model in (learner.actor, learner.critic)
                       for p in model.parameters()),
        losses={k.removeprefix('losses/'): v for k, v in logs.items() if k.startswith('losses/')},
        stats={k.removeprefix('environment/'): v for k, v in logs.items() if k.startswith('environment/')},
        last_stats={})
    PuffeRL.print_dashboard(view)


class TrainingStatistics:
    """Preserve the original environment/* report means and epoch log cadence."""
    def __init__(self):
        self.environment = defaultdict(list)
        self.losses = defaultdict(list)
        self.raw_steps = self.valid_steps = 0

    def collect(self, infos, raw_steps, valid_steps):
        self.raw_steps += raw_steps
        self.valid_steps += valid_steps
        for info in infos:
            # Pre-reset observations are replay data, not environment metrics.
            public = {k: v for k, v in info.items() if k != '_fasttd3_transition'}
            for key, value in unroll_nested_dict(public):
                values = np.asarray(value)
                if values.size and values.dtype.kind in 'biuf':
                    self.environment[key].extend(values.reshape(-1).tolist())

    def learn(self, result):
        self.losses['critic_loss'].append(result['critic_loss'])
        if result['actor_updated']:
            # A delayed actor step is absent, not an observed zero actor loss.
            self.losses['actor_loss'].append(result['actor_loss'])

    def metrics(self):
        return {
            **{f'environment/{k}': float(np.mean(v)) for k, v in self.environment.items()},
            **{f'losses/{k}': float(np.mean(v)) for k, v in self.losses.items()},
            # Replay reuses transitions; report admission separately instead of
            # mislabelling it as PPO's on-policy perc_transitions_used.
            'replay/valid_transition_fraction': self.valid_steps / max(1, self.raw_steps),
        }


def save_checkpoint(learner, args, logger, steps, epoch):
    """Keep numbered models, trainer state and W&B checkpoint artifacts."""
    from pufferlib.pufferl import WandbLogger
    env_name = args['env_name']
    directory = Path(args['train']['data_dir']) / f'{env_name}_{logger.run_id}'
    directory.mkdir(parents=True, exist_ok=True)
    model_path = directory / f'model_{env_name}_{epoch:06d}.pt'
    payload = dict(algorithm='fasttd3', model_state_dict=learner.actor.state_dict(),
        critic_state_dict=learner.critic.state_dict(), target_state_dict=learner.target.state_dict(),
        actor_optimizer=learner.actor_opt.state_dict(), critic_optimizer=learner.critic_opt.state_dict(),
        grad_scaler=learner.scaler.state_dict(), full_args=args, global_step=steps,
        updates=learner.updates)
    temporary = model_path.with_suffix('.pt.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, model_path)
    state_path = directory / 'trainer_state.pt'
    torch.save(dict(algorithm='fasttd3', global_step=steps, agent_step=steps, update=epoch,
        learner_updates=learner.updates, model_name=model_path.name, run_id=logger.run_id,
        actor_optimizer=learner.actor_opt.state_dict(), critic_optimizer=learner.critic_opt.state_dict(),
        grad_scaler=learner.scaler.state_dict()), str(state_path) + '.tmp')
    os.replace(str(state_path) + '.tmp', state_path)
    # Retain the adapter's existing latest-model path for evaluation commands.
    latest = directory.parent / f'{env_name}_fasttd3_{logger.run_id}.pt'
    shutil.copyfile(model_path, str(latest) + '.tmp')
    os.replace(str(latest) + '.tmp', latest)
    if isinstance(logger, WandbLogger):
        artifact = logger.wandb.Artifact(f'checkpoint-{logger.run_id}-epoch{epoch:06d}',
            type='checkpoint', metadata={'epoch': epoch, 'global_step': steps})
        artifact.add_file(str(model_path))
        artifact.add_file(str(state_path))
        logger.wandb.run.log_artifact(artifact)
    return str(latest)


def validate(args):
    env, cfg, train = args['env'], args['fasttd3'], args['train']
    if args['env_name'] != 'puffer_drive' or env['action_type'] != 'continuous':
        raise ValueError('FastTD3 requires puffer_drive --env.action-type continuous')
    if args['rnn_name'] != 'Recurrent' or args['rnn']['input_size'] != args['policy']['hidden_size']:
        raise ValueError('FastTD3 requires the existing compatible Recurrent/LSTM encoder')
    if env['lambda_value'] != 0 or not env['fix_lambdas']:
        raise ValueError('This stage supports lambda=0 only; anchor regularization is not implemented')
    if not env['async_resets'] or env['episode_length'] <= 0:
        raise ValueError('FastTD3 training requires async_resets and a finite episode_length')
    if int(os.environ.get('WORLD_SIZE', 1)) > 1:
        raise ValueError('FastTD3 supports one learner process')
    if cfg['amp_dtype'] not in ('bf16', 'fp16'):
        raise ValueError('fasttd3.amp_dtype must be bf16 or fp16')
    if args.get('load_id'):
        raise ValueError('Use load-model-path to continue a FastTD3 checkpoint')
    for key in ('replay_capacity', 'microbatch_sequences', 'policy_frequency', 'num_updates', 'learning_starts'):
        if cfg[key] <= 0:
            raise ValueError(f'fasttd3.{key} must be positive')
    if cfg['batch_size'] < train['rollout_horizon'] or cfg['batch_size'] % train['rollout_horizon']:
        raise ValueError('fasttd3.batch_size must be a positive multiple of rollout_horizon')
    if not 0 < cfg['tau'] <= 1 or cfg['num_atoms'] < 2 or cfg['v_min'] >= cfg['v_max']:
        raise ValueError('Invalid target update or distributional support parameters')


def transition_observations(observations, terminals, infos, required):
    next_obs, true_terminals = observations.copy(), terminals.copy()
    offset = 0
    for info in infos:
        meta = info.get('_fasttd3_transition')
        if meta is None:
            continue
        n = meta['count']
        next_obs[offset + meta['indices']] = meta['observations']
        true_terminals[offset:offset + n] = meta['terminals']
        offset += n
    if required and offset != len(observations):
        raise RuntimeError('Missing pre-reset transition metadata; rebuild the Drive extension')
    return next_obs, true_terminals


def evaluate(actor, args, logger, epoch):
    from pufferlib.pufferl import load_env
    from pufferlib.ocean.benchmark.evaluator import Evaluator
    evaluation_args = deepcopy(args)
    evaluation_args['env']['capture_final_observations'] = False
    evaluator = Evaluator(evaluation_args, logger)
    actor.eval()
    try:
        for mode, prefix in (('human_replay', 'hr'), ('self_play', 'sp')):
            if not args['eval'][mode + '_eval']:
                continue
            env = load_env('puffer_drive', getattr(evaluator, prefix + '_eval_config'))
            setattr(evaluator, prefix + '_env', env)
            try:
                evaluator.rollout(actor, mode=mode)
            finally:
                env.driver_env.stop_recorder(0)
                env.close()
            evaluator.log_videos(eval_mode=mode, epoch=epoch)
        return evaluator.collect_stats() if (args['eval']['human_replay_eval'] or
                                             args['eval']['self_play_eval']) else {}
    finally:
        actor.train()


def train(env_name, args, vecenv=None, policy=None, logger=None):
    from pufferlib.pufferl import load_env, NoLogger, WandbLogger, NeptuneLogger, Profile, Utilization
    validate(args)
    if policy is not None or vecenv is not None:
        raise ValueError('FastTD3 creates its own policy and an environment with final-observation capture')
    args = deepcopy(args)
    cfg, train_cfg = args['fasttd3'], args['train']
    seed = train_cfg['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.deterministic = train_cfg['torch_deterministic']
    torch.backends.cudnn.benchmark = True
    args['env'].update(capture_final_observations=True, uses_memory=True,
                       memory_size=train_cfg['rollout_horizon'])
    logger = logger or (WandbLogger(args) if args['wandb'] else
                        NeptuneLogger(args) if args['neptune'] else NoLogger(args))
    env = None
    checkpoint = None
    utilization = None
    stop_requested = False
    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
    previous_sigint = signal.signal(signal.SIGINT, request_stop)
    try:
        env = load_env(env_name, args)
        learner = Learner(env.driver_env, args)
        restored_steps = 0
        if args.get('load_model_path'):
            restored = torch.load(args['load_model_path'], map_location=train_cfg['device'], weights_only=False)
            if restored.get('algorithm') != 'fasttd3':
                raise ValueError('Expected a FastTD3 training checkpoint')
            learner.actor.load_state_dict(restored['model_state_dict'])
            learner.critic.load_state_dict(restored['critic_state_dict'])
            learner.target.load_state_dict(restored['target_state_dict'])
            learner.actor_opt.load_state_dict(restored['actor_optimizer'])
            learner.critic_opt.load_state_dict(restored['critic_optimizer'])
            learner.scaler.load_state_dict(restored['grad_scaler'])
            learner.updates = restored['updates']
            restored_steps = restored['global_step']
            if restored_steps >= train_cfg['total_timesteps']:
                raise ValueError('total_timesteps must exceed the checkpoint step')
            print('Restored learner and optimizer state; environment and replay restart with fresh warmup.', flush=True)
        actor, device = learner.actor, train_cfg['device']
        agents = env.num_agents
        obs_dim = env.single_observation_space.shape[0]
        act_dim = env.single_action_space.shape[0]
        replay = DeviceEpisodeReplay(agents, args['env']['episode_length'], obs_dim, act_dim,
                                     cfg['replay_capacity'], seed, device)
        previous_obs = torch.empty((agents, obs_dim), device=device)
        previous_actions = torch.empty((agents, act_dim), device=device)
        pending = np.zeros(agents, bool)
        dead = np.zeros(agents, bool)
        h = torch.zeros(agents, actor.hidden_size, device=device)
        c = torch.zeros_like(h)
        noise_scales = torch.empty(agents, 1, device=device).uniform_(cfg['std_min'], cfg['std_max'])
        steps, valid_steps = restored_steps, 0
        epoch = steps // train_cfg['batch_size']
        vector_clock = VectorClock(agents)
        utilization = Utilization()
        start_time = time.monotonic()
        speed_start = None
        speed_steps = 0
        conditioning = None
        logs = {}
        evaluation_stats = {}
        statistics = TrainingStatistics()
        profile = Profile()  # Original frequency=5 and CUDA synchronization semantics.
        last_log_time, last_log_step = start_time, steps
        env.async_reset(seed)
        while steps < train_cfg['total_timesteps']:
            profile('eval', epoch)
            profile('env', epoch, nest=True)
            obs, rewards, terminals, truncations, infos, ids, masks = env.recv()
            profile('eval_misc', epoch)
            ids = np.asarray(ids).reshape(-1)
            device_ids = torch.as_tensor(ids, dtype=torch.long, device=device)
            terminals, truncations = np.asarray(terminals, bool), np.asarray(truncations, bool)
            had_pending = pending[ids].copy()
            next_obs, true_terminals = transition_observations(obs, terminals, infos, had_pending.any())
            valid = had_pending & ~dead[ids] & np.asarray(masks, bool)
            statistics.collect(infos, int(had_pending.sum()), int(valid.sum()))
            live = ~dead[ids] & np.asarray(masks, bool)
            if live.any():
                conditioning = obs[live][:, [env.driver_env.lambda_obs_idx,
                                              env.driver_env.reward_veh_obs_idx]].copy()
            boundaries = terminals | truncations
            if valid.any():
                valid_ids = torch.as_tensor(ids[valid], dtype=torch.long, device=device)
                replay.add(ids[valid], previous_obs[valid_ids], previous_actions[valid_ids],
                           np.clip(rewards[valid], -1, 1), next_obs[valid], true_terminals[valid], boundaries[valid])
            steps += int(np.asarray(masks, bool).sum())
            valid_steps += int(valid.sum())
            dead[ids] |= true_terminals
            dead[ids[truncations]] = False
            reset_ids = torch.as_tensor(ids[boundaries], dtype=torch.long, device=device)
            h[reset_ids] = 0
            c[reset_ids] = 0
            noise_scales[reset_ids] = torch.empty(len(reset_ids), 1, device=device).uniform_(cfg['std_min'], cfg['std_max'])
            profile('eval_forward', epoch)
            with torch.no_grad(), learner.autocast():
                device_obs = torch.as_tensor(obs, device=device)
                if actor.obs_normalizer is not None:
                    actor.obs_normalizer.update(device_obs)
                state = dict(lstm_h=h[device_ids], lstm_c=c[device_ids])
                action, _ = actor.forward_eval(device_obs, state)
                h[device_ids], c[device_ids] = state['lstm_h'].float(), state['lstm_c'].float()
                device_action = (action + torch.randn_like(action) * noise_scales[device_ids]).clamp(-1, 1).float()
                previous_obs[device_ids], previous_actions[device_ids] = device_obs, device_action
                action = device_action.cpu().numpy()
            pending[ids] = True
            profile('env', epoch)
            env.send(action)
            profile.end()
            for vector_step in vector_clock.advance(int(had_pending.sum())):
                if vector_step <= cfg['learning_starts'] or not replay.size:
                    continue
                profile('train', epoch)
                profile('learn', epoch, nest=True)
                learner.schedule(steps / train_cfg['total_timesteps'])
                for _ in range(cfg['num_updates']):
                    logs = learner.update(replay)
                    statistics.learn(logs)
                profile.end()
                if speed_start is None and vector_step >= cfg['learning_starts'] + cfg['measure_burnin']:
                    if torch.device(device).type == 'cuda':
                        torch.cuda.synchronize(device)
                    speed_start, speed_steps = time.monotonic(), steps
            next_epoch = steps // train_cfg['batch_size']
            finished = steps >= train_cfg['total_timesteps']
            if next_epoch > epoch or finished or stop_requested:
                epoch = next_epoch
                logs = {k: v for k, v in logs.items() if k in
                        ('critic_loss', 'actor_loss', 'actor_updated', 'valid_samples')}
                logs.update(global_step=steps, valid_transitions=valid_steps,
                            replay_size=replay.size, updates=learner.updates,
                            replay_bytes=replay.bytes,
                            vector_steps=vector_clock.steps,
                            agent_steps=steps, epoch=epoch,
                            uptime=time.monotonic() - start_time,
                            learning_rate=float(learner.actor_opt.param_groups[0]['lr']),
                            critic_learning_rate=float(learner.critic_opt.param_groups[0]['lr']),
                            SPS=(steps - last_log_step) / max(1e-9, time.monotonic() - last_log_time),
                            **statistics.metrics(),
                            **{f'performance/{k}': v['elapsed'] for k, v in profile})
                if conditioning is not None:
                    logs['data/lambda_mean'] = float(conditioning[:, 0].mean())
                    logs['data/lambda_std'] = float(conditioning[:, 0].std())
                    if isinstance(logger, WandbLogger):
                        logs['data/lambda_distrib'] = logger.wandb.Histogram(conditioning[:, 0])
                        logs['data/collision_reward_distrib'] = logger.wandb.Histogram(conditioning[:, 1])
                if speed_start is not None and steps > speed_steps:
                    logs['SPS_train'] = (steps - speed_steps) / (time.monotonic() - speed_start)
                overhead_start = time.monotonic()
                evaluation_due = not stop_requested and (epoch % args['eval']['eval_interval'] == 0 or finished)
                if epoch % train_cfg['checkpoint_interval'] == 0 or finished or evaluation_due or stop_requested:
                    checkpoint = save_checkpoint(learner, args, logger, steps, epoch)
                    if args['eval'].get('wosac_realism_eval') and not stop_requested:
                        from pufferlib.utils import run_wosac_eval_in_subprocess
                        run_wosac_eval_in_subprocess(
                            {**train_cfg, 'env': env_name, 'eval': args['eval']},
                            logger, steps, full_args=args)
                if evaluation_due:
                    # Never retain old scores when the current evaluation fails.
                    evaluation_stats = {}
                    try:
                        scores = evaluate(actor, args, logger, epoch)
                        expected = []
                        if args['eval']['human_replay_eval']:
                            expected.append('eval/hr_score')
                        if args['eval']['self_play_eval']:
                            expected.append('eval/sp_score')
                        if any(key not in scores for key in expected):
                            raise RuntimeError('Evaluation returned no statistics for an enabled mode')
                        evaluation_stats.update(scores)
                        evaluation_stats['eval/failed'] = 0
                    except Exception as error:
                        evaluation_stats['eval/failed'] = 1
                        print(f'Evaluation failed: {error}. Checkpoint saved at {checkpoint}', flush=True)
                # Match PPO: check its 0.25 s throttle at an epoch boundary.
                logs.update(evaluation_stats)
                if finished or stop_requested or time.monotonic() > last_log_time + 0.25:
                    now = time.monotonic()
                    logs['SPS'] = (steps - last_log_step) / max(1e-9, now - last_log_time)
                    logs['uptime'] = now - start_time
                    logger.log(logs, steps)
                    evaluation_stats = {}
                    statistics = TrainingStatistics()
                    profile.clear()
                    print_dashboard(args, learner, utilization, profile, logs)
                    last_log_time, last_log_step = time.monotonic(), steps
                if speed_start is not None:
                    speed_start += time.monotonic() - overhead_start
            if stop_requested:
                break
        return [logs]
    finally:
        if utilization is not None:
            utilization.stop()
        try:
            if env is not None:
                env.close()
        finally:
            try:
                if checkpoint is not None:
                    logger.close(checkpoint)
                elif isinstance(logger, WandbLogger):
                    logger.wandb.finish()
            finally:
                signal.signal(signal.SIGINT, previous_sigint)

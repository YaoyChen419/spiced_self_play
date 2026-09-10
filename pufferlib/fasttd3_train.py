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
        self.sampled_valid = self.sampled_slots = 0
        self.reward_sum = self.reward_count = 0

    def collect(self, infos, raw_steps, valid_steps, rewards=None):
        self.raw_steps += raw_steps
        self.valid_steps += valid_steps
        if rewards is not None:
            self.reward_sum += float(np.sum(rewards))
            self.reward_count += len(rewards)
        for info in infos:
            # Pre-reset observations are replay data, not environment metrics.
            public = {k: v for k, v in info.items() if k != '_fasttd3_transition'}
            for key, value in unroll_nested_dict(public):
                values = np.asarray(value)
                if values.size and values.dtype.kind in 'biuf':
                    self.environment[key].extend(values.reshape(-1).tolist())

    def learn(self, result):
        self.sampled_valid += result.get('valid_samples', 0)
        self.sampled_slots += result.get('sampled_slots', 0)
        self.losses['critic_loss'].append(result['critic_loss'])
        for key in ('qf_loss', 'qf_min', 'qf_max', 'critic_grad_norm', 'buffer_rewards'):
            if key in result:
                self.losses[key].append(result[key])
        if result['actor_updated']:
            # A delayed actor step is absent, not an observed zero actor loss.
            self.losses['actor_loss'].append(result['actor_loss'])
            if 'actor_grad_norm' in result:
                self.losses['actor_grad_norm'].append(result['actor_grad_norm'])

    def metrics(self):
        metrics = {
            **{f'environment/{k}': float(np.mean(v)) for k, v in self.environment.items()},
            **{f'losses/{k}': float(np.mean(v)) for k, v in self.losses.items()},
            # Replay reuses transitions; report admission separately instead of
            # mislabelling it as PPO's on-policy perc_transitions_used.
            'replay/valid_transition_fraction': self.valid_steps / max(1, self.raw_steps),
        }
        if self.sampled_slots:
            metrics['environment/perc_transitions_used'] = self.sampled_valid / self.sampled_slots
        if self.reward_count:
            metrics['env_rewards'] = self.reward_sum / self.reward_count
        return metrics


def save_checkpoint(learner, args, logger, steps, epoch, publish=True):
    """Keep numbered models, trainer state and W&B checkpoint artifacts."""
    env_name = args['env_name']
    directory = Path(args['train']['data_dir']) / f'{env_name}_{logger.run_id}'
    directory.mkdir(parents=True, exist_ok=True)
    model_path = directory / f'model_{env_name}_{epoch:06d}.pt'
    payload = dict(algorithm='fasttd3', model_state_dict=learner.actor.state_dict(),
        critic_state_dict=learner.critic.state_dict(), target_state_dict=learner.target.state_dict(),
        actor_optimizer=learner.actor_opt.state_dict(), critic_optimizer=learner.critic_opt.state_dict(),
        grad_scaler=learner.scaler.state_dict(), full_args=args, global_step=steps,
        updates=learner.updates, epoch=epoch)
    temporary = model_path.with_suffix('.pt.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, model_path)
    state_path = directory / 'trainer_state.pt'
    torch.save(dict(algorithm='fasttd3', global_step=steps, agent_step=steps, update=epoch,
        learner_updates=learner.updates, model_name=model_path.name, run_id=logger.run_id,
        actor_optimizer=learner.actor_opt.state_dict(), critic_optimizer=learner.critic_opt.state_dict(),
        grad_scaler=learner.scaler.state_dict()), str(state_path) + '.tmp')
    os.replace(str(state_path) + '.tmp', state_path)
    # Match the platform's final-model path as well as its numbered directory.
    latest = directory.parent / f'{env_name}_{logger.run_id}.pt'
    shutil.copyfile(model_path, str(latest) + '.tmp')
    os.replace(str(latest) + '.tmp', latest)
    print(f'Checkpoint saved: step={steps} updates={learner.updates} path={latest}', flush=True)
    if publish:
        publish_checkpoint(args, logger, steps, epoch)
    return str(latest)


def publish_checkpoint(args, logger, steps, epoch):
    from pufferlib.pufferl import WandbLogger
    if isinstance(logger, WandbLogger):
        directory = Path(args['train']['data_dir']) / f"{args['env_name']}_{logger.run_id}"
        model_path = directory / f"model_{args['env_name']}_{epoch:06d}.pt"
        state_path = directory / 'trainer_state.pt'
        artifact = logger.wandb.Artifact(f'checkpoint-{logger.run_id}-epoch{epoch:06d}',
            type='checkpoint', metadata={'epoch': epoch, 'global_step': steps})
        artifact.add_file(str(model_path))
        artifact.add_file(str(state_path))
        logger.wandb.run.log_artifact(artifact)


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
    if args.get('load_id') and args.get('load_model_path'):
        raise ValueError('Select either load-id or load-model-path')
    if train['final_rollouts'] < 0:
        raise ValueError('final_rollouts must be non-negative')
    for key in ('replay_capacity', 'microbatch_sequences', 'policy_frequency', 'num_updates'):
        if cfg[key] <= 0:
            raise ValueError(f'fasttd3.{key} must be positive')
    if cfg['learning_starts'] < 0 or cfg['measure_burnin'] < 0:
        raise ValueError('learning_starts and measure_burnin must be non-negative')
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
                try:
                    env.driver_env.stop_recorder(0)
                finally:
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
    logger = logger or (WandbLogger(args, load_id=args.get('load_id')) if args['wandb'] else
                        NeptuneLogger(args, load_id=args.get('load_id')) if args['neptune'] else NoLogger(args))
    env = None
    checkpoint = None
    utilization = None
    stop_requested = False
    learner_pid = os.getpid()
    def request_stop(signum, frame):
        nonlocal stop_requested
        # Forked environment workers must exit when env.close() terminates
        # them; only the learner can perform checkpoint/log finalization.
        if os.getpid() != learner_pid:
            raise SystemExit(128 + signum)
        stop_requested = True
    previous_handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        if args.get('load_id'):
            if not hasattr(logger, 'download'):
                raise ValueError('load-id requires the original W&B or Neptune logger')
            args['load_model_path'] = logger.download()
        env = load_env(env_name, args)
        learner = Learner(env.driver_env, args)
        restored_steps = restored_epoch = 0
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
            restored_epoch = restored.get('epoch', restored_steps // train_cfg['batch_size'])
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
        epoch = restored_epoch
        initial_epoch = epoch
        rollout_slots = 0
        drain_until = None
        pending_checkpoint = None
        all_logs = []
        vector_clock = VectorClock(agents)
        utilization = Utilization()
        start_time = time.monotonic()
        speed_start = None
        speed_steps = 0
        logs = {}
        evaluation_stats = {}
        statistics = TrainingStatistics()
        profile = Profile()  # Original frequency=5 and CUDA synchronization semantics.
        last_log_time, last_log_step = start_time, steps
        env.async_reset(seed)
        while True:
            if (speed_start is None and replay.size and
                    vector_clock.steps >= cfg['learning_starts'] + cfg['measure_burnin']):
                if torch.device(device).type == 'cuda':
                    torch.cuda.synchronize(device)
                speed_start, speed_steps = time.monotonic(), steps
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
            statistics.collect(infos, int(had_pending.sum()), int(valid.sum()), np.clip(rewards[valid], -1, 1))
            boundaries = terminals | truncations
            if valid.any():
                valid_ids = torch.as_tensor(ids[valid], dtype=torch.long, device=device)
                replay.add(ids[valid], previous_obs[valid_ids], previous_actions[valid_ids],
                           np.clip(rewards[valid], -1, 1), next_obs[valid], true_terminals[valid], boundaries[valid])
            steps += int(np.asarray(masks, bool).sum())
            rollout_slots += len(ids)
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
                if actor.obs_normalizer is not None and drain_until is None:
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
                if drain_until is not None or vector_step <= cfg['learning_starts'] or not replay.size:
                    continue
                profile('train', epoch)
                profile('learn', epoch, nest=True)
                learner.schedule(steps / train_cfg['total_timesteps'])
                for update_index in range(cfg['num_updates']):
                    # Match the official schedule, including num_updates=1.
                    actor_due = (update_index % cfg['policy_frequency'] == 1
                                 if cfg['num_updates'] > 1 else vector_step % cfg['policy_frequency'] == 0)
                    logs = learner.update(replay, actor_step=actor_due)
                    statistics.learn(logs)
                profile.end()
            next_epoch = initial_epoch + rollout_slots // train_cfg['batch_size']
            finished = drain_until is None and steps >= train_cfg['total_timesteps']
            drain_finished = (drain_until is not None and rollout_slots >= drain_until
                              and bool(statistics.environment))
            if (drain_until is None and next_epoch > epoch) or finished or drain_finished or stop_requested:
                if drain_until is None:
                    epoch = next_epoch
                logs = {k: v for k, v in logs.items() if k in
                        ('critic_loss', 'actor_loss', 'actor_updated', 'valid_samples',
                         'qf_loss', 'qf_min', 'qf_max', 'actor_grad_norm', 'critic_grad_norm', 'buffer_rewards')}
                logs.update(global_step=steps, valid_transitions=valid_steps,
                            replay_size=replay.size, updates=learner.updates,
                            replay_bytes=replay.bytes,
                            vector_steps=vector_clock.steps,
                            agent_steps=steps, epoch=epoch,
                            phase='final_collection' if drain_until is not None else 'train',
                            uptime=time.monotonic() - start_time,
                            learning_rate=float(learner.actor_opt.param_groups[0]['lr']),
                            critic_learning_rate=float(learner.critic_opt.param_groups[0]['lr']),
                            SPS=(steps - last_log_step) / max(1e-9, time.monotonic() - last_log_time),
                            **statistics.metrics(),
                            **{f'performance/{k}': v['elapsed'] for k, v in profile})
                if learner.conditioning is not None:
                    conditioning = learner.conditioning.cpu().numpy()
                    logs['data/lambda_mean'] = float(conditioning[:, 0].mean())
                    logs['data/lambda_std'] = float(conditioning[:, 0].std())
                    if isinstance(logger, WandbLogger):
                        logs['data/lambda_distrib'] = logger.wandb.Histogram(conditioning[:, 0])
                        logs['data/collision_reward_distrib'] = logger.wandb.Histogram(conditioning[:, 1])
                if speed_start is not None and steps > speed_steps:
                    logs['SPS_train'] = (steps - speed_steps) / (time.monotonic() - speed_start)
                overhead_start = time.monotonic()
                evaluation_due = (drain_until is None and not stop_requested
                                  and (epoch % args['eval']['eval_interval'] == 0 or finished))
                if epoch % train_cfg['checkpoint_interval'] == 0 or finished or evaluation_due or stop_requested or drain_finished:
                    checkpoint = save_checkpoint(learner, args, logger, steps, epoch, publish=False)
                    pending_checkpoint = (steps, epoch)
                    if args['eval'].get('wosac_realism_eval') and not stop_requested and drain_until is None:
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
                if finished or drain_finished or stop_requested or time.monotonic() > last_log_time + 0.25:
                    now = time.monotonic()
                    logs['SPS'] = (steps - last_log_step) / max(1e-9, now - last_log_time)
                    logs['uptime'] = now - start_time
                    profile.clear()
                    print_dashboard(args, learner, utilization, profile, logs)
                    print(f'FastTD3 step={steps} updates={learner.updates} SPS={logs["SPS"]:.1f}', flush=True)
                    logger.log(logs, steps)
                    if steps > .20 * train_cfg['total_timesteps']:
                        if drain_finished:
                            all_logs.append(logs.copy())
                        else:
                            all_logs = [logs.copy()]
                    evaluation_stats = {}
                    statistics = TrainingStatistics()
                    last_log_time, last_log_step = time.monotonic(), steps
                if pending_checkpoint is not None:
                    publish_checkpoint(args, logger, *pending_checkpoint)
                    pending_checkpoint = None
                if speed_start is not None:
                    speed_start += time.monotonic() - overhead_start
            if stop_requested or drain_finished:
                break
            if finished:
                final_rollouts = train_cfg['final_rollouts']
                if final_rollouts == 0:
                    break
                drain_until = rollout_slots + final_rollouts * train_cfg['batch_size']
                print(f'Final collection: {final_rollouts} rollouts, no optimizer updates.', flush=True)
        print(f'Training stopped: reason={"signal" if stop_requested else "budget"} step={steps} checkpoint={checkpoint}', flush=True)
        return all_logs or [logs]
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
                elif isinstance(logger, NeptuneLogger):
                    logger.neptune.stop()
            finally:
                for sig, handler in previous_handlers.items():
                    signal.signal(sig, handler)

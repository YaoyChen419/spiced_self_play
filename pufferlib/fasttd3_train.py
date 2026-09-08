"""Driving collector and existing evaluation hooks for recurrent FastTD3."""
from copy import deepcopy
from pathlib import Path
import os
import random
import time

import numpy as np
import torch

from pufferlib.fasttd3 import Learner
from pufferlib.fasttd3_replay import DeviceEpisodeReplay


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
    if args.get('load_model_path') or args.get('load_id'):
        raise ValueError('Training resume is not implemented; use load-model-path for evaluation only')
    for key in ('replay_capacity', 'microbatch_sequences', 'policy_frequency', 'num_updates', 'learning_starts'):
        if cfg[key] <= 0:
            raise ValueError(f'fasttd3.{key} must be positive')
    if train['minibatch_size'] < train['rollout_horizon'] or train['minibatch_size'] % train['rollout_horizon']:
        raise ValueError('minibatch_size must be a positive multiple of rollout_horizon')
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
    from pufferlib.pufferl import load_env, NoLogger, WandbLogger, NeptuneLogger
    validate(args)
    if policy is not None or vecenv is not None:
        raise ValueError('FastTD3 creates its own policy and an environment with final-observation capture')
    if args['eval'].get('wosac_realism_eval'):
        raise ValueError('WOSAC during FastTD3 training is not connected yet; use standalone evaluation')
    args = deepcopy(args)
    cfg, train_cfg = args['fasttd3'], args['train']
    seed = train_cfg['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.deterministic = train_cfg['torch_deterministic']
    args['env'].update(capture_final_observations=True, uses_memory=True,
                       memory_size=train_cfg['rollout_horizon'])
    logger = logger or (WandbLogger(args) if args['wandb'] else
                        NeptuneLogger(args) if args['neptune'] else NoLogger(args))
    env = load_env(env_name, args)
    checkpoint = None
    try:
        learner = Learner(env.driver_env, args)
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
        steps = valid_steps = epoch = receives = 0
        start_time = time.monotonic()
        speed_start = None
        speed_steps = 0
        logs = {}
        env.async_reset(seed)
        while steps < train_cfg['total_timesteps']:
            obs, rewards, terminals, truncations, infos, ids, masks = env.recv()
            ids = np.asarray(ids).reshape(-1)
            terminals, truncations = np.asarray(terminals, bool), np.asarray(truncations, bool)
            had_pending = pending[ids].copy()
            next_obs, true_terminals = transition_observations(obs, terminals, infos, had_pending.any())
            valid = had_pending & ~dead[ids] & np.asarray(masks, bool)
            boundaries = terminals | truncations
            if valid.any():
                replay.add(ids[valid], previous_obs[ids[valid]], previous_actions[ids[valid]],
                           rewards[valid], next_obs[valid], true_terminals[valid], boundaries[valid])
            steps += int(had_pending.sum())
            valid_steps += int(valid.sum())
            dead[ids] |= true_terminals
            dead[ids[truncations]] = False
            reset_ids = ids[boundaries]
            h[reset_ids] = 0
            c[reset_ids] = 0
            noise_scales[reset_ids] = torch.empty(len(reset_ids), 1, device=device).uniform_(cfg['std_min'], cfg['std_max'])
            with torch.no_grad(), learner.autocast():
                device_obs = torch.as_tensor(obs, device=device)
                if actor.obs_normalizer is not None:
                    actor.obs_normalizer.update(device_obs)
                state = dict(lstm_h=h[ids], lstm_c=c[ids])
                action, _ = actor.forward_eval(device_obs, state)
                h[ids], c[ids] = state['lstm_h'].float(), state['lstm_c'].float()
                device_action = (action + torch.randn_like(action) * noise_scales[ids]).clamp(-1, 1).float()
                previous_obs[ids], previous_actions[ids] = device_obs, device_action
                action = device_action.cpu().numpy()
            pending[ids] = True
            env.send(action)
            receives += 1
            if receives >= cfg['learning_starts'] and replay.size:
                learner.schedule(steps / train_cfg['total_timesteps'])
                for _ in range(cfg['num_updates']):
                    logs = learner.update(replay)
                if speed_start is None and learner.updates >= cfg['measure_burnin']:
                    if torch.device(device).type == 'cuda':
                        torch.cuda.synchronize(device)
                    speed_start, speed_steps = time.monotonic(), steps
            next_epoch = steps // train_cfg['batch_size']
            finished = steps >= train_cfg['total_timesteps']
            if next_epoch > epoch or finished:
                epoch = next_epoch
                logs.update(global_step=steps, valid_transitions=valid_steps,
                            replay_size=replay.size, updates=learner.updates,
                            replay_bytes=replay.bytes,
                            SPS=steps / (time.monotonic() - start_time))
                if speed_start is not None and steps > speed_steps:
                    logs['SPS_train'] = (steps - speed_steps) / (time.monotonic() - speed_start)
                overhead_start = time.monotonic()
                evaluation_due = epoch % args['eval']['eval_interval'] == 0 or finished
                if epoch % train_cfg['checkpoint_interval'] == 0 or finished or evaluation_due:
                    path = Path(train_cfg['data_dir']) / f'{env_name}_fasttd3_{logger.run_id}.pt'
                    path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(dict(algorithm='fasttd3', model_state_dict=actor.state_dict(),
                        critic_state_dict=learner.critic.state_dict(), target_state_dict=learner.target.state_dict(),
                        actor_optimizer=learner.actor_opt.state_dict(), critic_optimizer=learner.critic_opt.state_dict(),
                        grad_scaler=learner.scaler.state_dict(),
                        full_args=args, global_step=steps, updates=learner.updates), path)
                    checkpoint = str(path)
                if evaluation_due:
                    # Never retain old scores when the current evaluation fails.
                    logs = {key: value for key, value in logs.items() if not key.startswith('eval/')}
                    try:
                        scores = evaluate(actor, args, logger, epoch)
                        expected = []
                        if args['eval']['human_replay_eval']:
                            expected.append('eval/hr_score')
                        if args['eval']['self_play_eval']:
                            expected.append('eval/sp_score')
                        if any(key not in scores for key in expected):
                            raise RuntimeError('Evaluation returned no statistics for an enabled mode')
                        logs.update(scores)
                        logs['eval/failed'] = 0
                    except Exception as error:
                        logs['eval/failed'] = 1
                        print(f'Evaluation failed: {error}. Checkpoint saved at {checkpoint}', flush=True)
                logger.log(logs, steps)
                if speed_start is not None:
                    speed_start += time.monotonic() - overhead_start
                print(f'FastTD3-LSTM step={steps} updates={learner.updates} SPS={logs["SPS"]:.0f}', flush=True)
        return [logs]
    finally:
        env.close()
        if checkpoint is not None:
            logger.close(checkpoint)

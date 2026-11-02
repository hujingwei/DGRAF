import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from dgraf_core import DGRAF
from environment import Environment


def run(model, weights, seed, regime, limit):
    episode_seed = seed * 100003
    np.random.seed(episode_seed)
    torch.manual_seed(episode_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(episode_seed)
    env = Environment(weights['groups'].numpy())
    observation, mask = env.reset(episode_seed, regime)
    model.reset(observation, mask, episode_seed)
    done = False
    while not done and (limit is None or env.steps < limit):
        index, action, _, _ = model.act(env.n)
        observation, mask, done = env.step(index)
        model.advance(action, observation, mask)
    return env, done


def save_plot(env, path, limit):
    soc = np.asarray(env.trace)
    time = np.arange(len(soc)) * env.interval
    fig, axis = plt.subplots(figsize=(9, 5))
    for index in range(8):
        axis.plot(time, soc[:, index], linewidth=1.0)
    lower = max(0.0, float(soc.min()) - 1.0) if limit == 150 else 15.0
    axis.set(xlabel='Time (s)', ylabel='SOC (%)', ylim=(lower, 100))
    axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=4100)
    parser.add_argument('--regime', choices=('mild', 'severe'), default='mild')
    parser.add_argument('--window', choices=('150', 'full', 'both'), default='both')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    weights = torch.load(root / 'weights.pt', map_location='cpu', weights_only=True)
    model = DGRAF(weights, 'cuda' if torch.cuda.is_available() else 'cpu')
    outputs = ((150, 'soc_150_steps.png'), (None, 'soc_full_discharge.png'))
    if args.window == '150':
        outputs = outputs[:1]
    elif args.window == 'full':
        outputs = outputs[1:]
    for limit, name in outputs:
        env, done = run(model, weights, args.seed, args.regime, limit)
        save_plot(env, root / name, limit)
        print(f'{name}: steps={env.steps}, energy_wh={env.energy:.6f}, imr_percent={100*env.improvements/env.steps:.3f}, voltage_valid_percent={100*env.voltage_valid/env.steps:.3f}, final_soc_diff_percent={np.ptp(env.soc):.6f}, final_soc_min={env.soc.min():.6f}, stopped_below_20={done}', flush=True)


if __name__ == '__main__':
    main()

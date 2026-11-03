from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from dgraf_core import DGRAF
from environment import Environment

ROOT = Path(__file__).resolve().parent
SEED = 31131
RUNS = 5

def run(model, weights, seed, regime, limit):
    episode_seed = seed * 100003
    np.random.seed(episode_seed)
    torch.manual_seed(episode_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(episode_seed)
    env = Environment(weights['groups'].numpy())
    model.environment = env
    observation, mask = env.reset(episode_seed, regime)
    model.reset(observation, mask, episode_seed)
    done = False
    while not done and (limit is None or env.steps < limit):
        index, action, _, _ = model.act(env.n)
        observation, mask, done = env.step(index)
        model.advance(action, observation, mask)
    return env

def plot(environments, filename, limit, seeds):
    fig, axes = plt.subplots(len(environments), 1, figsize=(9, 16), squeeze=False)
    for i, env in enumerate(environments):
        axis = axes[i, 0]
        soc = np.asarray(env.trace)
        axis.plot(np.arange(len(soc)) * env.interval, soc, linewidth=1.0)
        lower = max(0.0, float(soc.min()) - 1.0) if limit else 15.0
        axis.set(xlabel='Time (s)', ylabel='SOC (%)', ylim=(lower, 100), title=f'Run {i + 1}, seed={seeds[i]}')
        axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(ROOT / filename, dpi=200)
    plt.close(fig)

def main():
    weights = torch.load(ROOT / 'weights.pt', map_location='cpu', weights_only=True)
    model = DGRAF(weights, 'cuda' if torch.cuda.is_available() else 'cpu')
    seeds = [SEED + i for i in range(RUNS)]
    for regime in ('mild', 'severe'):
        for limit, filename in ((150, 'soc_150_steps.png'), (None, 'soc_full_discharge.png')):
            environments = [run(model, weights, seed, regime, limit) for seed in seeds]
            name = f'{regime}_{filename}' if regime == 'severe' else filename
            plot(environments, name, limit, seeds)
            energy = np.array([env.energy for env in environments])
            imr = np.array([100 * env.improvements / env.steps for env in environments])
            print(f'{regime} {"150" if limit else "full"}: energy={energy.mean():.6f}±{energy.std(ddof=1):.6f}, imr={imr.mean():.6f}±{imr.std(ddof=1):.6f}', flush=True)

if __name__ == '__main__':
    main()

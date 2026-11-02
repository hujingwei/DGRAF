import numpy as np


class Environment:
    def __init__(self, groups):
        self.groups = np.asarray(groups, dtype=np.int64)
        self.capacity = 2.0
        self.current = 2.0
        self.interval = 5.0

    def load(self):
        while True:
            n = int(np.rint(self.rng.normal(4.0, 1.5)))
            if 1 <= n <= 8:
                return n

    def observe(self):
        active = self.remaining > 0
        starts = ~active & (self.rng.random(8) < self.probability)
        sources = active | starts
        neighbors = np.zeros(8, dtype=bool)
        neighbors[1:] |= sources[:-1]
        neighbors[:-1] |= sources[1:]
        starts |= ~active & neighbors & (self.rng.random(8) < self.probability)
        for cell in np.flatnonzero(starts):
            self.remaining[cell] = self.rng.integers(*self.duration)
        mask = self.remaining == 0
        self.remaining[~mask] -= 1
        return np.where(mask, self.soc / 100.0, 0.0).astype(np.float32), mask

    def reset(self, seed, regime):
        self.rng = np.random.default_rng(seed)
        self.probability, self.duration = {
            'mild': (0.01, (2, 11)),
            'severe': (0.03, (5, 21)),
        }[regime]
        self.soc = self.rng.uniform(85.0, 100.0, 8)
        self.trace = [self.soc.copy()]
        self.steps = 0
        self.energy = 0.0
        self.improvements = 0
        self.voltage_valid = 0
        self.remaining = np.zeros(8, dtype=np.int64)
        self.n = self.load()
        return self.observe()

    def step(self, action):
        before = float(np.ptp(self.soc))
        groups = self.groups[int(action)]
        ocv = 3.0 + 1.2 * self.soc / 100.0
        currents = np.zeros(8)
        voltage = 0.0
        for group in np.unique(groups):
            if group <= 0:
                continue
            members = groups == group
            conductance = np.full(int(members.sum()), 20.0)
            terminal = (np.sum(ocv[members] * conductance) - self.current) / conductance.sum()
            currents[members] = (ocv[members] - terminal) * 20.0
            voltage += terminal
        self.soc = np.clip(self.soc - currents * self.interval / (self.capacity * 3600.0) * 100.0, 0.0, 100.0)
        self.trace.append(self.soc.copy())
        after = float(np.ptp(self.soc))
        self.steps += 1
        self.energy += max(0.0, voltage * self.current * self.interval / 3600.0)
        self.improvements += int(after <= before + 1e-12)
        self.voltage_valid += int(3.2 * self.n <= voltage <= 4.0 * self.n)
        done = bool(self.soc.min() < 20.0)
        self.n = self.load()
        observation, mask = self.observe()
        return observation, mask, done

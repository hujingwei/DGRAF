from pathlib import Path

import numpy as np
from openpyxl import load_workbook

class Environment:
    def __init__(self, groups):
        self.groups = np.asarray(groups, dtype=np.int64)
        self.capacity, self.current, self.interval = 2.0, 2.0, 5.0
        workbook = load_workbook(Path(__file__).with_name('rint.xlsx'), read_only=True, data_only=True)
        try:
            self.table = np.asarray(list(workbook.active.values), dtype=np.float64)
        finally:
            workbook.close()
        if self.table.shape[1] != 3 or self.table[0, 0] != 0 or self.table[-1, 0] != 1:
            raise ValueError('invalid rint.xlsx')

    def parameters(self, soc):
        fraction = np.asarray(soc, dtype=np.float64) / 100.0
        return tuple(np.interp(fraction, self.table[:, 0], self.table[:, i]) for i in (1, 2))

    def predict_voltage(self, soc, membership):
        ocv, resistance = self.parameters(soc)
        conductance = 1 / resistance
        denominator = np.einsum('mgc,c->mg', membership, conductance)
        numerator = np.einsum('mgc,c->mg', membership, ocv * conductance)
        return np.divide(numerator - self.current, denominator, out=np.zeros_like(denominator), where=denominator > 0).sum(1)

    def load(self):
        while True:
            value = int(np.rint(self.rng.normal(4.0, 1.5)))
            if 1 <= value <= 8:
                return value

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
        return np.where(mask, self.soc / 100, 0).astype(np.float32), mask

    def reset(self, seed, regime):
        self.rng = np.random.default_rng(seed)
        self.probability, self.duration = {'mild': (0.01, (2, 11)), 'severe': (0.03, (5, 21))}[regime]
        self.soc = self.rng.uniform(85, 100, 8)
        self.trace, self.steps, self.energy = [self.soc.copy()], 0, 0.0
        self.improvements = self.voltage_valid = 0
        self.remaining = np.zeros(8, dtype=np.int64)
        self.n = self.load()
        return self.observe()

    def step(self, action):
        before = np.ptp(self.soc)
        groups = self.groups[int(action)]
        ocv, resistance = self.parameters(self.soc)
        currents = np.zeros(8)
        voltage = 0.0
        for group in np.unique(groups):
            if group > 0:
                members = groups == group
                conductance = 1 / resistance[members]
                terminal = (np.sum(ocv[members] * conductance) - self.current) / conductance.sum()
                currents[members] = (ocv[members] - terminal) * conductance
                voltage += terminal
        self.soc = np.clip(self.soc - currents * self.interval / (self.capacity * 3600) * 100, 0, 100)
        self.trace.append(self.soc.copy())
        after = np.ptp(self.soc)
        self.steps += 1
        self.energy += max(0, voltage * self.current * self.interval / 3600)
        self.improvements += int(after <= before + 1e-12)
        self.voltage_valid += int(3.2 * self.n <= voltage <= 4.0 * self.n)
        done = bool(self.soc.min() < 20)
        self.n = self.load()
        observation, mask = self.observe()
        return observation, mask, done

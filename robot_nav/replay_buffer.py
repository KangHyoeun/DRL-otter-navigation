import random
from collections import deque
import itertools

import numpy as np


class ReplayBuffer(object):
    def __init__(self, buffer_size, random_seed=123):
        """
        The right side of the deque contains the most recent experiences
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)

    def add(self, s, a, r, t, s2):
        experience = (s, a, r, t, s2)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch]).reshape(-1, 1)
        t_batch = np.array([_[3] for _ in batch]).reshape(-1, 1)
        s2_batch = np.array([_[4] for _ in batch])

        return s_batch, a_batch, r_batch, t_batch, s2_batch

    def return_buffer(self):
        s = np.array([_[0] for _ in self.buffer])
        a = np.array([_[1] for _ in self.buffer])
        r = np.array([_[2] for _ in self.buffer]).reshape(-1, 1)
        t = np.array([_[3] for _ in self.buffer]).reshape(-1, 1)
        s2 = np.array([_[4] for _ in self.buffer])

        return s, a, r, t, s2

    def clear(self):
        self.buffer.clear()
        self.count = 0


class RolloutReplayBuffer(object):
    def __init__(self, buffer_size, random_seed=123, history_len=10):
        """
        The right side of the deque contains the most recent experiences
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque(maxlen=buffer_size)
        random.seed(random_seed)
        self.buffer.append([])
        self.history_len = history_len

    def add(self, s, a, r, t, s2):
        experience = (s, a, r, t, s2)
        if t:
            self.count += 1
            self.buffer[-1].append(experience)
            self.buffer.append([])
        else:
            self.buffer[-1].append(experience)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        if self.count < batch_size:
            batch = random.sample(
                list(itertools.islice(self.buffer, 0, len(self.buffer) - 1)), self.count
            )
        else:
            batch = random.sample(
                list(itertools.islice(self.buffer, 0, len(self.buffer) - 1)), batch_size
            )

        idx = [random.randint(0, len(b) - 1) for b in batch]

        s_batch = []
        s2_batch = []
        for i in range(len(batch)):
            if idx[i] == len(batch[i]):
                s = batch[i]
                s2 = batch[i]
            else:
                s = batch[i][: idx[i] + 1]
                s2 = batch[i][: idx[i] + 1]
            s = [v[0] for v in s]
            s = s[::-1]

            s2 = [v[4] for v in s2]
            s2 = s2[::-1]

            if len(s) < self.history_len:
                missing = self.history_len - len(s)
                s += [s[-1]] * missing
                s2 += [s2[-1]] * missing
            else:
                s = s[: self.history_len]
                s2 = s2[: self.history_len]
            s = s[::-1]
            s_batch.append(s)
            s2 = s2[::-1]
            s2_batch.append(s2)

        a_batch = np.array([batch[i][idx[i]][1] for i in range(len(batch))])
        r_batch = np.array([batch[i][idx[i]][2] for i in range(len(batch))]).reshape(
            -1, 1
        )
        t_batch = np.array([batch[i][idx[i]][3] for i in range(len(batch))]).reshape(
            -1, 1
        )

        return np.array(s_batch), a_batch, r_batch, t_batch, np.array(s2_batch)

    def return_buffer(self):
        s = np.array([_[0] for _ in self.buffer])
        a = np.array([_[1] for _ in self.buffer])
        r = np.array([_[2] for _ in self.buffer]).reshape(-1, 1)
        t = np.array([_[3] for _ in self.buffer]).reshape(-1, 1)
        s2 = np.array([_[4] for _ in self.buffer])

        return s, a, r, t, s2

    def clear(self):
        self.buffer.clear()
        self.count = 0

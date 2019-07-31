import numpy

class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = numpy.zeros( 2*capacity - 1 )
        self.data = numpy.zeros( capacity, dtype=object )

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])

    def total(self):
        return self.tree[0]

    def p_array(self):
        return self.tree[-self.capacity:]

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])

    def normalise(self):
        self.total = numpy.sum(self.tree[-self.capacity:])
        for i in range(self.capacity - 1, len(self.tree), 1):
            p_norm = self.tree[i]/self.total
            self.update(i, p_norm)

    def unnormalise(self):
        for i in range(self.capacity - 1, len(self.tree), 1):
            p = self.tree[i]*self.total
            self.update(i, p)

    def memory(self):
        return self.tree.nbytes + self.data.nbytes


def perfect_run(iterations, dataset):
    values = []
    for iteration in range(iterations):
        cash = [1000]
        position = [0]
        value =[1000]
        for t in range(len(dataset)- 60):
            # define current environment/state (s) and next environment (s1)
            s = dataset[t:t+60]
            s1 = dataset[t+1:t+61]
            # update cash and position metrics
            def a_is_0():
                if cash[-1] == 0:
                    cash.append(position[-1]*s[-1])
                    position.append(0)
                    value.append(cash[-1])
                else:
                    cash.append(cash[-1])
                    position.append(0)
                    value.append(cash[-1])
            def a_is_1():
                if position[-1] == 0:
                    position.append(cash[-1]/s[-1])
                    cash.append(0)
                    value.append(position[-1]*s[-1])
                else:
                    position.append(position[-1])
                    value.append(position[-1]*s[-1])
            # perfect walk
            if s1[-1] > s[-1]:
                a_is_1()
            else:
                a_is_0()
        values.append(value[-1])
    return values, value

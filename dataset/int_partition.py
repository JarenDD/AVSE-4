import numpy as np
import heapq

# Random partition with given value L and num_parts n
def int_partition(L, n, exist_empty=True):
    if not exist_empty:
        L -= n
        assert L >= 0, 'Total value cannot cover non-empty partitions'
        padding = np.ones(n, dtype=int)
    else:
        padding = np.zeros(n, dtype=int)
    ns = np.random.rand(n)
    ns = L * (ns / np.sum(ns))
    ns_int = ns.astype(int)
    diff = ns - ns_int
    res = L - np.sum(ns_int)
    largest_list = heapq.nlargest(res, diff)
    for i,val in enumerate(diff):
        if val in largest_list:
            ns_int[i] += 1

    assert np.sum(ns_int) == L, 'Partition Error'
    return ns_int + padding

def random_occlude(L, occlude_ratio=0.2, lower=15, upper=25, exist_empty=False):
    occlude_length = max(int(np.ceil(L * occlude_ratio)), lower)
    if occlude_length > np.floor(occlude_length / lower) * upper:
        # When occlude_length is not within any [i*lower, i*upper], assign occlude_length with a closer border
        if occlude_length >= (np.floor(occlude_length / upper) * upper + np.ceil(occlude_length / lower) * lower) / 2:
            occlude_length = int(np.ceil(occlude_length / lower) * lower)
        else:
            occlude_length = int(np.floor(occlude_length / upper) * upper)
    n_range = [int(np.ceil(occlude_length / upper)), int(np.floor(occlude_length / lower)+1)]
    n = np.random.randint(*n_range)
    rand_occlude_length = occlude_length - lower * n
    ns = np.random.randint(0, upper - lower + 1, size=n) # n parts within [0, upper - lower]

    res = np.sum(ns) - rand_occlude_length
    while res > 0:
        ns = np.clip(ns - int_partition(res, n, exist_empty=True), a_min=0, a_max=None)
        res = np.sum(ns) - rand_occlude_length
    while res < 0:
        ns = np.clip(ns + int_partition(-res, n, exist_empty=True), a_min=None, a_max=10)
        res = np.sum(ns) - rand_occlude_length
    vs = int_partition(L - occlude_length, n + 1, exist_empty=False)

    return ns + lower, vs

if __name__ == '__main__':
    # print(int_partition(50, 3, exist_empty=False))
    print(random_occlude(50))
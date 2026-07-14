from typing import Any
from tqdm import tqdm
from time import sleep
import multiprocessing as mp


def waiting_proc(r, p: Any, desc=None):
    remaining = list(range(len(r)))

    with tqdm(desc=desc, total=len(r), disable=not desc) as pbar:
        while len(remaining) > 0:
            all_alive = all([j.is_alive() for j in p._pool])
            if not all_alive:
                raise RuntimeError('Some background worker is break.')
            done = [i for i in remaining if r[i].ready()]
            remaining = [i for i in remaining if i not in done]
            pbar.update(len(done))
            sleep(0.02)


def sample_fn(i, j):
    return i, i * j + j


if __name__ == '__main__':
    rs = []
    with mp.get_context('spawn').Pool(12) as _p:
        for x in range(50):
            rs.append(_p.starmap_async(sample_fn, ((x, x + 12),)))
        waiting_proc(rs, _p, '  State')

    results = [r.get()[0] for r in rs]
    print(results)

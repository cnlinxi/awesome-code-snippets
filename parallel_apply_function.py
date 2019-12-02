from six import moves

'''
modified from https://github.com/bojone/python-snippets
'''


def parallel_apply(func,
                   iterable,
                   workers,
                   max_queue_size,
                   callback=None,
                   dummy=False):
    '''
    Execute func in parallel and disorder.
    :param func: function pointer
    :param iterable: elements processed by func
    :param workers: worker number
    :param max_queue_size: queue size
    :param callback: post-process function
    :param dummy: True if you want to execute IO-intensive else CPU-intensive task
    :return: list. if callback is None, result will be func(x). Or callback(func(x))&result will be empty.
    '''
    if dummy:
        from multiprocessing.dummy import Pool, Queue
    else:
        from multiprocessing import Pool, Queue

    in_queue, out_queue = Queue(max_queue_size), Queue()

    def worker_step(in_queue, out_queue):
        while True:
            d = in_queue.get()
            r = func(d)
            out_queue.put(r)

    # start multi-thread(dummy==True) or multi-progress(dummy==False)
    pool = Pool(workers, worker_step, (in_queue, out_queue))

    results = []

    # post-process wrapper
    def process_out_queue():
        out_count = 0
        for _ in range(out_queue.qsize()):
            d = out_queue.get()
            out_count += 1
            if callback is None:
                results.append(d)
            else:
                callback(d)
        return out_count

    # put into data and get result
    in_count, out_count = 0, 0
    for d in iterable:
        in_count += 1
        while True:
            try:
                in_queue.put(d, block=False)
                break
            except moves.queue.Full:
                out_count += process_out_queue()
        if in_count % max_queue_size == 0:
            out_count += process_out_queue()

    while out_count != in_count:
        out_count += process_out_queue()

    pool.terminate()

    if callback is None:
        return results


if __name__ == '__main__':
    iterable = list(range(10))


    def func(x):
        return x * 10


    def callback(x):
        print('callback: {}'.format(x / 10))

    # Got 'NotImplementedError' on macOS:
    # https://github.com/vterron/lemon/blob/d60576bec2ad5d1d5043bcb3111dff1fcb58a8d6/methods.py#L536-L573
    # https://github.com/keras-team/autokeras/issues/368
    # shit macOS
    print(parallel_apply(func, iterable, workers=2, max_queue_size=5))
    parallel_apply(func, iterable, workers=2, max_queue_size=5)

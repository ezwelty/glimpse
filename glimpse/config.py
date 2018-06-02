from .imports import sharedmem
_MapReduce = sharedmem.MapReduce

def set_sharedmem_backend(backend):
    backends = dict(
        process=sharedmem.MapReduce,
        thread=sharedmem.MapReduceByThread
    )
    global _MapReduce
    _MapReduce = backends[backend]

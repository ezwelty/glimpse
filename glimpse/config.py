from .imports import sharedmem

_MapReduce = sharedmem.MapReduce
_UseMatMul = True


def set_sharedmem_backend(backend):
    backends = {"process": sharedmem.MapReduce, "thread": sharedmem.MapReduceByThread}
    global _MapReduce
    _MapReduce = backends[backend]


def use_numpy_matmul(flag):
    global _UseMatMul
    _UseMatMul = bool(flag)

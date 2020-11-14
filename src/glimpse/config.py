"""Global configuration options."""
import sharedmem

backend = sharedmem.MapReduce
"""
Backend to use for parallel processing.

Either :obj:`sharedmem.MapReduce` or :obj:`sharedmem.MapReduceByThread`.
"""

matmul = True
"""
Whether to use matrix multiplication.

If `False`, disables the use of :obj:`numpy.matmul`,
which can cause problems for parallel processing.
"""

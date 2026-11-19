# Copyright 2026 DeepBeepMeep. Licensed under Apache-2.0 (see LICENSE).
"""Integration points for applications. The defaults work standalone; an application may replace them, for example:

    import sagegp.hooks
    sagegp.hooks.register_constant = my_graph_manager.register_constant
"""


def register_constant(tensor):
    """Called once for each process-wide constant tensor that kernels read at a fixed address (a rotation matrix): a CUDA graph
    manager that recycles memory between captures must keep it alive and unmoved. Returns the tensor the kernels will read."""
    return tensor


def notice(operation):
    """Called each time an optional kernel route is taken (operation is a short readable name), for applications that log the
    kernels they use."""

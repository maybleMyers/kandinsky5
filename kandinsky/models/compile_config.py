import os

# Global flag to control torch.compile() usage
# Set this to False before importing model modules to disable compilation
# Check environment variable first (default to True if not set)
USE_TORCH_COMPILE = os.environ.get("KANDINSKY_NO_COMPILE", "0") != "1"

def maybe_compile(fn=None, **compile_kwargs):
    """
    Decorator that conditionally applies torch.compile() based on USE_TORCH_COMPILE flag.

    Usage:
        @maybe_compile()
        def my_function(x):
            ...

        @maybe_compile(mode="max-autotune-no-cudagraphs", dynamic=True)
        def my_other_function(x):
            ...
    """
    import torch

    def decorator(func):
        if USE_TORCH_COMPILE:
            return torch.compile(func, **compile_kwargs)
        return func

    if fn is not None:
        # Called without parentheses: @maybe_compile
        return decorator(fn)

    # Called with parentheses: @maybe_compile() or @maybe_compile(mode=...)
    return decorator

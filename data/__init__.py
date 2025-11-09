"""Data pipeline modules."""

# Lazy imports to avoid loading TensorFlow unnecessarily
# Only import what's needed when it's needed

def __getattr__(name):
    if name == 'DataLoader':
        from .data_loader import DataLoader
        return DataLoader
    elif name == 'generate_dummy_data':
        # Import from scripts.dummy since we moved dummy_generator there
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from scripts.dummy.dummy_generator import generate_dummy_data
        return generate_dummy_data
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ['DataLoader', 'generate_dummy_data']

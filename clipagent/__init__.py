"""clipagent - turn long videos into viral short-form clips."""

from .config import Settings
from .pipeline import find_clips, run_pipeline

__all__ = ["Settings", "find_clips", "run_pipeline"]
__version__ = "0.1.0"

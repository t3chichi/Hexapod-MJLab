"""Task package for hexapod-mjlab (external mjlab tasks).

Registered as an ``mjlab.tasks`` entry point (see ``pyproject.toml``), so
``import mjlab`` auto-imports this package, which imports the task config
packages that call :func:`mjlab.tasks.registry.register_mjlab_task`. This is
the official external-task discovery mechanism (``mjlab/__init__.py::
_import_registered_packages``).
"""

from . import confined_locomotion  # noqa: F401  (registers El-4090-Confined-Locomotion-Flat)
from . import confined_locomotion_2  # noqa: F401  (registers the Flat_2 and Terrain-Memory_2 tasks)
from .confined_locomotion_2 import tripod2_low  # noqa: F401

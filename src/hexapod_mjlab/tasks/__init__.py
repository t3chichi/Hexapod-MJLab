"""Task package for hexapod-mjlab (external mjlab tasks).

Registered as an ``mjlab.tasks`` entry point (see ``pyproject.toml``), so
``import mjlab`` auto-imports this package, which imports the task config
packages that call :func:`mjlab.tasks.registry.register_mjlab_task`. This is
the official external-task discovery mechanism (``mjlab/__init__.py::
_import_registered_packages``).
"""

from . import el_4090_lidar  # noqa: F401  (registers the EL4090 LiDAR task)

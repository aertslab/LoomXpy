from typing import Tuple

from ._s7 import S7
from ._mode import Modes


class LoomX(S7):
    def __init__(self):
        self.modes: Modes = Modes()
        self._active_mode: str = None
        # Data Matrix
        self._data_matrix = None
        # Features
        self._feature_attrs = None
        # Observations
        self._observation_attrs = None

    @property
    def active(self):
        return self._active_mode

    def _validate_active_value(self, value):
        _mode_keys_str = f"{', '.join(self.modes._keys)}"
        if len(self.modes._keys) == 0:
            raise Exception(
                "Cannot set an active mode. None has been detected. You can add one using `loomx.modes.<mode-name> = <data-matrix>`."
            )
        if isinstance(value, str) and value in self.modes:
            return True
        if isinstance(value, Tuple[str, ...]):
            raise Exception("This is currently not implemented.")
        raise Exception(
            f"The mode {value} does not exist. Choose one of: {_mode_keys_str}."
        )

    @active.setter
    def active(self, value):
        self._validate_active_value(value=value)
        self._active_mode = value
        self._data_matrix = self.modes[value].X
        self._feature_attrs = self.modes[value].f
        self._observation_attrs = self.modes[value].o

    def _check_active_mode_is_set(self):
        if self._active_mode is None:
            raise Exception(
                "No active mode set. Use `loomx.active = <mode-name>` to set one."
            )

    @property
    def X(self):
        self._check_active_mode_is_set()
        return self._data_matrix

    @property
    def f(self):
        self._check_active_mode_is_set()
        return self._feature_attrs

    @property
    def o(self):
        self._check_active_mode_is_set()
        return self._observation_attrs
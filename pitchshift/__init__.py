"""变调不变速与时长拉伸工具包。

核心实现位于 :mod:`pitchshift.stretch`。
"""

from .stretch import (
    StretchError,
    detect_period,
    frame_signal,
    overlap_add,
    pitch_shift,
    preserve_formant,
    process_batch,
    time_stretch,
    validate_params,
)

__all__ = [
    "StretchError",
    "detect_period",
    "frame_signal",
    "overlap_add",
    "pitch_shift",
    "preserve_formant",
    "process_batch",
    "time_stretch",
    "validate_params",
]

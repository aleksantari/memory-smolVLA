"""Compatibility patch for torchvision >= 0.26 which removed VideoReader.

LeRobot 0.4.x routes the ``pyav`` backend through
``torchvision.io.VideoReader``, which no longer exists in torchvision 0.26+.
This module monkey-patches ``lerobot.datasets.video_utils`` so that the
``pyav`` backend uses PyAV directly instead.

Import this module before any dataset access to apply the patch::

    import memory_smolvla.data._video_compat  # noqa: F401
"""

from __future__ import annotations

import logging

import av
import torch
import torchvision

logger = logging.getLogger(__name__)

_HAS_VIDEO_READER = hasattr(torchvision.io, "VideoReader")

if not _HAS_VIDEO_READER:
    from pathlib import Path

    from lerobot.datasets import video_utils as _vu

    _original_decode = _vu.decode_video_frames_torchvision

    def _decode_via_pyav(
        video_path: Path | str,
        timestamps: list[float],
        tolerance_s: float,
        backend: str = "pyav",
        log_loaded_timestamps: bool = False,
    ) -> torch.Tensor:
        """Pure-PyAV replacement for the removed torchvision VideoReader path."""
        video_path = str(video_path)
        first_ts = min(timestamps)
        last_ts = max(timestamps)

        container = av.open(video_path)
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"

        # Seek to nearest keyframe before first_ts
        time_base = float(stream.time_base)
        seek_pts = int(first_ts / time_base) if time_base else 0
        container.seek(seek_pts, stream=stream)

        loaded_frames: list[torch.Tensor] = []
        loaded_ts: list[float] = []

        for frame in container.decode(video=0):
            pts = float(frame.pts * time_base) if frame.pts is not None else 0.0
            if log_loaded_timestamps:
                logger.info("frame loaded at timestamp=%.4f", pts)
            # Convert to CHW uint8 tensor
            arr = frame.to_ndarray(format="rgb24")
            tensor = torch.from_numpy(arr).permute(2, 0, 1)  # HWC -> CHW
            loaded_frames.append(tensor)
            loaded_ts.append(pts)
            if pts >= last_ts:
                break

        container.close()

        if not loaded_frames:
            raise _vu.FrameTimestampError(
                f"No frames decoded from {video_path} for timestamps {timestamps}"
            )

        query_ts = torch.tensor(timestamps)
        loaded_ts_t = torch.tensor(loaded_ts)

        dist = torch.cdist(query_ts[:, None], loaded_ts_t[:, None], p=1)
        min_, argmin_ = dist.min(1)

        is_within_tol = min_ < tolerance_s
        if not is_within_tol.all():
            raise _vu.FrameTimestampError(
                f"One or several query timestamps violate the tolerance "
                f"({min_[~is_within_tol]} > {tolerance_s=})."
                f"\nqueried timestamps: {query_ts}"
                f"\nloaded timestamps: {loaded_ts_t}"
                f"\nvideo: {video_path}"
            )

        closest_frames = torch.stack([loaded_frames[idx] for idx in argmin_])
        closest_frames = closest_frames.float() / 255.0

        if len(timestamps) != len(closest_frames):
            raise _vu.FrameTimestampError(
                f"Retrieved {len(closest_frames)} frames but queried {len(timestamps)} timestamps"
            )

        return closest_frames

    # Apply the patch
    _vu.decode_video_frames_torchvision = _decode_via_pyav
    logger.info(
        "Patched lerobot video_utils: using pure PyAV decoder "
        "(torchvision.io.VideoReader not available)"
    )

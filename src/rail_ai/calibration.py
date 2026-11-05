"""Camera calibration loading and frame undistortion."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class CameraUndistorter:
    """Apply an optional OpenCV camera calibration to decoded frames."""

    def __init__(self, path: Path | None, cv2: Any, np: Any) -> None:
        self.cv2 = cv2
        self.np = np
        self._maps: dict[tuple[int, int], tuple[Any, Any]] = {}
        self._camera_matrix: Any = None
        self._distortion_coeffs: Any = None
        self._model = ""
        self._calibration_resolution: tuple[int, int] | None = None
        if path is not None:
            self._load(path)

    def _load(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"Camera calibration file not found: {path}")
        try:
            import yaml

            with path.open("r", encoding="utf-8") as calibration_file:
                data = yaml.safe_load(calibration_file)
        except (OSError, yaml.YAMLError) as exc:
            raise ValueError(f"Unable to read camera calibration file {path}: {exc}") from exc

        if not isinstance(data, dict):
            raise ValueError(f"Camera calibration file {path} must contain a mapping")
        camera = data
        if "intrinsics" not in camera and len(data) == 1:
            camera = next(iter(data.values()))
        if not isinstance(camera, dict):
            raise ValueError(f"Camera calibration file {path} has no camera mapping")

        try:
            intrinsics = camera["intrinsics"]
            distortion_coeffs = camera["distortion_coeffs"]
            model = str(camera["distortion_model"]).lower()
            if len(intrinsics) != 4 or len(distortion_coeffs) < 4:
                raise ValueError("intrinsics must contain 4 values and distortion_coeffs at least 4 values")
            fx, fy, cx, cy = (float(value) for value in intrinsics)
            self._camera_matrix = self.np.array(
                [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                dtype=self.np.float64,
            )
            self._distortion_coeffs = self.np.asarray(distortion_coeffs, dtype=self.np.float64).reshape(-1, 1)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid camera calibration file {path}: {exc}") from exc

        if model in {"equidistant", "fisheye", "opencv_fisheye"}:
            if len(distortion_coeffs) != 4:
                raise ValueError("Fisheye calibration requires exactly 4 distortion coefficients")
            self._model = "fisheye"
        elif model in {
            "plumb_bob",
            "brown_conrady",
            "brown-conrady",
            "radtan",
            "radial_tangential",
            "radial-tangential",
            "rational_polynomial",
            "opencv",
        }:
            self._model = "tangential"
        else:
            raise ValueError(f"Unsupported camera distortion model: {model}")

        resolution = camera.get("resolution")
        if resolution is not None:
            if not isinstance(resolution, (list, tuple)) or len(resolution) != 2:
                raise ValueError("Camera calibration resolution must contain [width, height]")
            self._calibration_resolution = (int(resolution[0]), int(resolution[1]))
            if min(self._calibration_resolution) <= 0:
                raise ValueError("Camera calibration resolution must be positive")

    def _maps_for(self, width: int, height: int) -> tuple[Any, Any]:
        key = (width, height)
        if key in self._maps:
            return self._maps[key]
        matrix = self._camera_matrix.copy()
        if self._calibration_resolution is not None:
            calibration_width, calibration_height = self._calibration_resolution
            matrix[0] *= width / calibration_width
            matrix[1] *= height / calibration_height
        size = (width, height)
        if self._model == "fisheye":
            maps = self.cv2.fisheye.initUndistortRectifyMap(
                matrix,
                self._distortion_coeffs,
                self.np.eye(3),
                matrix,
                size,
                self.cv2.CV_32FC1,
            )
        else:
            maps = self.cv2.initUndistortRectifyMap(
                matrix,
                self._distortion_coeffs,
                None,
                matrix,
                size,
                self.cv2.CV_32FC1,
            )
        self._maps[key] = maps
        return maps

    def apply(self, frame: Any) -> Any:
        if self._camera_matrix is None:
            return frame
        height, width = frame.shape[:2]
        map_x, map_y = self._maps_for(width, height)
        return self.cv2.remap(frame, map_x, map_y, self.cv2.INTER_LINEAR)

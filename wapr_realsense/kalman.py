"""Constant-velocity pose prediction for WAPR initial guesses.

Inspired by FoundationPose++'s pose/velocity Kalman scheme:
https://github.com/teal024/FoundationPose-plus-plus/blob/master/src/utils/kalman_filter_6d.py
This implementation uses sensor time and a local SO(3) rotation error instead
of filtering absolute Euler angles. All translations refer to the mesh origin.
"""

from dataclasses import dataclass, fields

import numpy as np


@dataclass(frozen=True)
class KalmanSettings:
    activate: bool = False
    translation_measurement_std_m: float = 0.01
    rotation_measurement_std_rad: float = 0.05
    translation_acceleration_std_m_s2: float = 1.0
    rotation_acceleration_std_rad_s2: float = 3.0
    max_dt_s: float = 0.5

    def __post_init__(self):
        if type(self.activate) is not bool:
            raise ValueError("Kalman activate must be a boolean")
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name != "activate" and (isinstance(value, bool) or not np.isfinite(value) or value <= 0):
                raise ValueError("Kalman %s must be finite and positive" % field.name)


def settings_from_args(args):
    return KalmanSettings(**{field.name: getattr(args, "kalman_" + field.name)
                             for field in fields(KalmanSettings)})


def _left_jacobian(vector):
    """SO(3) left Jacobian, including its small-angle limit."""
    x, y, z = vector
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    angle = np.linalg.norm(vector)
    if angle < 1e-6:
        return np.eye(3) + 0.5 * skew + (skew @ skew) / 6.0
    return (np.eye(3) + (1 - np.cos(angle)) / angle**2 * skew
            + (angle - np.sin(angle)) / angle**3 * (skew @ skew))


class PoseKalman:
    """12-state error filter: position, rotation error, velocity, angular velocity.

    Rotation increments and angular velocities use camera axes. Predict once
    before refinement, then update only with an accepted WAPR measurement.
    """

    def __init__(self, settings):
        from scipy.spatial.transform import Rotation
        self.rotation = Rotation
        self.settings = settings
        self.reset()

    def reset(self):
        self.pose = None
        self.timestamp = None
        self.covariance = None
        self.velocity = np.zeros(6)

    def initiate(self, pose, timestamp):
        self.pose = np.array(pose, dtype=np.float64, copy=True)
        self.last_measurement = self.pose.copy()
        self.timestamp = float(timestamp)
        self.velocity = np.zeros(6)
        s = self.settings
        self.noise = np.diag([s.translation_measurement_std_m**2] * 3
                             + [s.rotation_measurement_std_rad**2] * 3)
        self.covariance = np.diag(np.r_[np.diag(self.noise), [0.5**2] * 3, [1.0**2] * 3])

    def predict(self, timestamp):
        dt = float(timestamp) - self.timestamp
        if not np.isfinite(dt) or dt <= 0 or dt > self.settings.max_dt_s:
            # Repeated/backward clocks and long gaps must not extrapolate stale speed.
            self.initiate(self.last_measurement, timestamp)
            return self.pose.astype(np.float32)
        delta = self.velocity[3:] * dt
        rotation_step = self.rotation.from_rotvec(delta).as_matrix()
        self.pose[:3, 3] += self.velocity[:3] * dt
        self.pose[:3, :3] = rotation_step @ self.pose[:3, :3]
        transition = np.eye(12)
        transition[:3, 6:9] = np.eye(3) * dt
        transition[3:6, 3:6] = rotation_step
        transition[3:6, 9:12] = _left_jacobian(delta) * dt
        gain = np.vstack((np.eye(6) * dt**2 / 2, np.eye(6) * dt))
        s = self.settings
        acceleration = np.diag([s.translation_acceleration_std_m_s2**2] * 3
                               + [s.rotation_acceleration_std_rad_s2**2] * 3)
        self.covariance = (transition @ self.covariance @ transition.T
                           + gain @ acceleration @ gain.T)
        self.timestamp = float(timestamp)
        return self.pose.astype(np.float32)

    def update(self, measurement):
        measurement = np.asarray(measurement, dtype=np.float64)
        residual = np.r_[measurement[:3, 3] - self.pose[:3, 3],
                         self.rotation.from_matrix(measurement[:3, :3] @ self.pose[:3, :3].T).as_rotvec()]
        innovation = self.covariance[:6, :6] + self.noise
        gain = np.linalg.solve(innovation, self.covariance[:6, :]).T
        correction = gain @ residual
        self.pose[:3, 3] += correction[:3]
        self.pose[:3, :3] = self.rotation.from_rotvec(correction[3:6]).as_matrix() @ self.pose[:3, :3]
        self.velocity += correction[6:]
        # Joseph covariance update, followed by the rotation-error reset.
        remainder = np.eye(12)
        remainder[:, :6] -= gain
        covariance = remainder @ self.covariance @ remainder.T + gain @ self.noise @ gain.T
        reset = np.eye(12)
        reset[3:6, 3:6] = _left_jacobian(correction[3:6])
        covariance = reset @ covariance @ reset.T
        self.covariance = (covariance + covariance.T) / 2
        self.last_measurement = measurement.copy()

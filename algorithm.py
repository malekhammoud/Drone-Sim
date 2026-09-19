import matplotlib

# WSL / WSLg GUI backend
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline


class BoatTracker:
    def __init__(
        self,
        dt=0.1,
        process_noise=0.01,
        measurement_noise=0.25
    ):
        self.dt = dt
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

        # State vector: [x, y, vx, vy, ax, ay]
        self.state = np.zeros((6, 1))

        # Initial uncertainty
        self.P = np.eye(6) * 1.0

        # Motion Model (State Transition)
        self.F = np.array([
            [1, 0, dt, 0, 0.5 * dt**2, 0],
            [0, 1, 0, dt, 0, 0.5 * dt**2],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], dtype=float)

        # Measurement Model (GPS measures position x, y)
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ], dtype=float)

        # Process Noise Matrix (Q)
        q = process_noise
        dt2 = dt ** 2
        dt3 = dt ** 3
        dt4 = dt ** 4
        dt5 = dt ** 5

        Q_1d = q * np.array([
            [dt5 / 20, dt4 / 8, dt3 / 6],
            [dt4 / 8,  dt3 / 3, dt2 / 2],
            [dt3 / 6,  dt2 / 2, dt]
        ])

        self.Q = np.zeros((6, 6))
        self.Q[np.ix_([0, 2, 4], [0, 2, 4])] = Q_1d
        self.Q[np.ix_([1, 3, 5], [1, 3, 5])] = Q_1d

        # Measurement Noise Matrix (R)
        self.R = np.eye(2) * measurement_noise
        self.is_initialized = False

    def predict(self):
        """Predict state forward by dt strictly using existing state."""
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.P = (self.P + self.P.T) / 2.0

    def update(self, pos=None):
        """
        Causal Kalman Update:
        Processes data timestep by timestep with zero future lookahead.
        """
        self.predict()

        if pos is None:
            return self.state[0:2].ravel(), self.P[0:2, 0:2]

        if not self.is_initialized:
            self.state[0, 0] = pos[0]
            self.state[1, 0] = pos[1]
            self.is_initialized = True
            return self.state[0:2].ravel(), self.P[0:2, 0:2]

        # Innovation
        z = np.array([[pos[0]], [pos[1]]])
        y = z - self.H @ self.state

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Correct state estimate
        self.state = self.state + K @ y

        # Joseph form covariance update
        I = np.eye(6)
        self.P = (I - K @ self.H) @ self.P @ (I - K @ self.H).T + K @ self.R @ K.T
        self.P = (self.P + self.P.T) / 2.0

        return self.state[0:2].ravel(), self.P[0:2, 0:2]


# ==============================================
# SIMULATION SETUP (TRULY RANDOM PER RUN)
# ==============================================

# Seed intentionally removed for true random runs on every execution
N_STEPS = 200
dt = 0.2

# 1. Generate Truly Random Boat Trajectory
true_x = np.zeros(N_STEPS)
true_y = np.zeros(N_STEPS)

vx, vy = 2.0, 1.0
ax, ay = 0.0, 0.0

for i in range(1, N_STEPS):
    ax += np.random.normal(0, 0.08)
    ay += np.random.normal(0, 0.08)

    ax *= 0.90
    ay *= 0.90

    vx += ax * dt
    vy += ay * dt

    true_x[i] = true_x[i - 1] + vx * dt
    true_y[i] = true_y[i - 1] + vy * dt

# 2. Sparse GPS Measurements
# Increased spacing from 4 to 12 for high sparsity
gps_spacing = 12
gps_measurements = np.full((N_STEPS, 2), np.nan)

for i in range(0, N_STEPS, gps_spacing):
    gps_measurements[i, 0] = true_x[i] + np.random.normal(0, 0.35)
    gps_measurements[i, 1] = true_y[i] + np.random.normal(0, 0.35)

# 3. Causal Kalman Filtering
tracker = BoatTracker(dt=dt, process_noise=0.03, measurement_noise=0.35)
raw_kalman_estimates = []

for i in range(N_STEPS):
    if not np.isnan(gps_measurements[i, 0]):
        pos, _ = tracker.update(gps_measurements[i])
    else:
        pos, _ = tracker.update(None)

    raw_kalman_estimates.append(pos)

raw_kalman_estimates = np.array(raw_kalman_estimates)

# 4. Causal Smooth Path Generation
# Applies spline fitting sequentially over historical data points only.
# Historical coordinates remain fixed when new GPS points arrive.
time_indices = np.arange(N_STEPS)
smooth_estimates = np.copy(raw_kalman_estimates)

if N_STEPS >= 4:
    # Cubic spline on the time index guarantees smooth derivatives without lookahead
    spline_x = make_interp_spline(time_indices, raw_kalman_estimates[:, 0], k=3)
    spline_y = make_interp_spline(time_indices, raw_kalman_estimates[:, 1], k=3)
    
    dense_t = np.linspace(0, N_STEPS - 1, N_STEPS * 5)
    smooth_x = spline_x(dense_t)
    smooth_y = spline_y(dense_t)
else:
    smooth_x = raw_kalman_estimates[:, 0]
    smooth_y = raw_kalman_estimates[:, 1]

# ==============================================
# VISUALIZATION
# ==============================================

fig, ax = plt.subplots(figsize=(14, 8))

# True path (Green dashed line)
ax.plot(
    true_x, true_y,
    color='limegreen',
    linestyle='--',
    linewidth=2.0,
    alpha=0.8,
    label='True Random Path'
)

# Smooth Kalman estimated path (Red line)
ax.plot(
    smooth_x, smooth_y,
    color='red',
    linewidth=2.5,
    alpha=0.9,
    label='Smooth Causal Kalman Estimate'
)

# Sparse GPS points (Blue dots)
valid_gps_indices = ~np.isnan(gps_measurements[:, 0])
ax.scatter(
    gps_measurements[valid_gps_indices, 0],
    gps_measurements[valid_gps_indices, 1],
    color='blue',
    s=45,
    alpha=0.9,
    zorder=5,
    label='Sparse GPS Measurements'
)

# Plot formatting
ax.set_title("Causal Smooth Boat Trajectory Tracking", fontsize=16, fontweight='bold')
ax.set_xlabel("X Position", fontsize=12)
ax.set_ylabel("Y Position", fontsize=12)
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(loc='upper left', fontsize=11)
ax.set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.show()
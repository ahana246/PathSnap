# Converts lon/lat to local metres and computes point-to-segment distances
import numpy as np

# Local equirectangular projection: trips lie in different regions, so each trip is
# projected around its own mean latitude (error is negligible at city scale)
M_PER_DEG_LAT = 111139.0


def get_meters_conversion(mean_lat):
    m_per_deg_lon = M_PER_DEG_LAT * np.cos(np.radians(mean_lat))
    return m_per_deg_lon, M_PER_DEG_LAT


def to_meters(lon, lat, mean_lat):
    m_per_deg_lon, m_per_deg_lat = get_meters_conversion(mean_lat)
    return np.asarray(lon) * m_per_deg_lon, np.asarray(lat) * m_per_deg_lat


def point_to_segment_vectorized(px, py, x1, y1, x2, y2):
    # returns distance to segment, projection ratio t in [0, 1] and segment length
    dx = x2 - x1
    dy = y2 - y1
    l2 = dx * dx + dy * dy
    l2_safe = np.where(l2 == 0, 1e-9, l2)

    t = np.clip(((px - x1) * dx + (py - y1) * dy) / l2_safe, 0.0, 1.0)
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy

    dist = np.hypot(px - proj_x, py - proj_y)
    return dist, t, np.sqrt(l2)


def gps_kinematics(track, mean_lat):
    # per-point speed (m/s), heading (rad) and time gap (s) from consecutive GPS points
    x, y = to_meters(track['lon'].values, track['lat'].values, mean_lat)
    dx, dy = np.diff(x), np.diff(y)
    dt = np.diff(track['time'].values)

    speed = np.hypot(dx, dy) / np.maximum(dt, 0.1)
    heading = np.arctan2(dy, dx)

    # first point has no previous point, so reuse the first step's values
    speed = np.concatenate([[speed[0]], speed]) if len(speed) else np.zeros(1)
    heading = np.concatenate([[heading[0]], heading]) if len(heading) else np.zeros(1)
    dt = np.concatenate([[1.0], dt]) if len(dt) else np.ones(1)
    return speed, heading, dt

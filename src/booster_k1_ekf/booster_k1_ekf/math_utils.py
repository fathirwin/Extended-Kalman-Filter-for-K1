import numpy as np

def skew_symmetric(v):
    x, y, z = np.asarray(v, float).ravel()
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])

def quat_to_matrix(q):
    x, y, z, w = q
    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,     1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x**2 - 2*y**2]])

def quaternion_multiply(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
         x1*w2 + y1*z2 - z1*y2 + w1*x2,
        -x1*z2 + y1*w2 + z1*x2 + w1*y2,
         x1*y2 - y1*x2 + z1*w2 + w1*z2,
        -x1*x2 - y1*y2 - z1*z2 + w1*w2])

def zeta(v):
    v = np.asarray(v, float)
    a = np.linalg.norm(v)
    if a < 1e-9:
        return np.array([0.5*v[0], 0.5*v[1], 0.5*v[2], 1.0])
    s = np.sin(0.5*a) / a
    return np.array([v[0]*s, v[1]*s, v[2]*s, np.cos(0.5*a)])

def rot_axis_angle(axis, angle):
    axis = np.asarray(axis, float)
    n = np.linalg.norm(axis)
    if n < 1e-12 or abs(angle) < 1e-15:
        return np.eye(3)
    K = skew_symmetric(axis / n)
    return np.eye(3) + np.sin(angle)*K + (1 - np.cos(angle))*(K @ K)

def rotation_angle(R):
    return float(np.arccos(np.clip((np.trace(R) - 1.0)/2.0, -1.0, 1.0)))

def rotvec(R):
    """Vektor rotasi (log map) dari matriks rotasi."""
    a = rotation_angle(R)
    if a < 1e-9:
        return 0.5*np.array([R[2, 1]-R[1, 2], R[0, 2]-R[2, 0], R[1, 0]-R[0, 1]])
    return a/(2*np.sin(a))*np.array([R[2, 1]-R[1, 2], R[0, 2]-R[2, 0], R[1, 0]-R[0, 1]])

def euler_rpy(R):
    """Roll, pitch, yaw (rad) dari matriks rotasi badan->dunia."""
    return np.array([np.arctan2(R[2, 1], R[2, 2]),
                     -np.arcsin(np.clip(R[2, 0], -1, 1)),
                     np.arctan2(R[1, 0], R[0, 0])])

def wrap_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi
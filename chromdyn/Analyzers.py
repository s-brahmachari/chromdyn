import numpy as np
import h5py
from multiprocessing import Pool, cpu_count

# new version: in traj_utils.py
def _segment_solid_angle(p1, p2, q1, q2):
    r13, r14 = q1 - p1, q2 - p1
    r23, r24 = q1 - p2, q2 - p2

    n1 = np.cross(r13, r14)
    n2 = np.cross(r14, r24)
    n3 = np.cross(r24, r23)
    n4 = np.cross(r23, r13)

    n1 /= np.linalg.norm(n1)
    n2 /= np.linalg.norm(n2)
    n3 /= np.linalg.norm(n3)
    n4 /= np.linalg.norm(n4)

    angles = np.arcsin([
        np.clip(np.dot(n1, n2), -1, 1),
        np.clip(np.dot(n2, n3), -1, 1),
        np.clip(np.dot(n3, n4), -1, 1),
        np.clip(np.dot(n4, n1), -1, 1),
    ])

    omega_star = np.sum(angles)
    sign = np.sign(np.dot(np.cross(q2 - q1, p2 - p1), r13))

    return (omega_star / (4 * np.pi)) * sign

def compute_writhe_single_curve(coords, closed=True):
    N = len(coords)
    if closed:
        coords = np.vstack((coords, coords[0]))  # Close the loop
    
    writhe = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            if abs(i - j) > 1 and not (closed and {i, j} == {0, N-1}):
                writhe += _segment_solid_angle(
                    coords[i], coords[i + 1],
                    coords[j], coords[j + 1]
                )
    return writhe

def compute_writhe_between_curves(curve1, curve2):
    """
    Computes the writhe between two closed curves (curve1 and curve2).

    Parameters:
    curve1, curve2: ndarray of shape (N, 3) and (M, 3)
        Points defining the closed curves C1 and C2.

    Returns:
    writhe: float
        The computed writhe between the two curves.
    """
    N, M = len(curve1), len(curve2)

    # Ensure curves are closed by appending the first point at the end
    curve1_closed = np.vstack([curve1, curve1[0]])
    curve2_closed = np.vstack([curve2, curve2[0]])

    writhe = 0.0
    for i in range(N):
        for j in range(M):
            writhe += _segment_solid_angle(
                curve1_closed[i], curve1_closed[i + 1],
                curve2_closed[j], curve2_closed[j + 1]
            )

    return writhe


def compute_writhe_trajectory(trajectory, closed=True, processes=None):
    if processes is None:
        processes = max(cpu_count() - 1, 1)
    
    with Pool(processes) as pool:
        args = [(frame, closed) for frame in trajectory]
        results = pool.starmap(compute_writhe_single_curve, args)

    return results

def load_traj(traj_file, d=1):
    print('Loading trajectory ...')
    pos = []
    with h5py.File(str(traj_file), "r") as f:
        for key in sorted(f.keys()):
            try:
                frame_id = int(key)
                if frame_id%d==0:
                    pos.append(np.array(f[key]))
            except ValueError:
                pass
    pos = np.array(pos)  
    print('Trajectory shape:', pos.shape)
    return pos

def compute_RG(positions):
    positions = np.asarray(positions)

    if positions.ndim == 2:  # shape (N, 3)
        center_of_mass = np.mean(positions, axis=0)
        squared_distances = np.sum((positions - center_of_mass) ** 2, axis=1)
        return float(np.sqrt(np.mean(squared_distances)))

    elif positions.ndim == 3:  # shape (T, N, 3)
        centers_of_mass = np.mean(positions, axis=1)  # shape (T, 3)
        squared_distances = np.sum((positions - centers_of_mass[:, None, :]) ** 2, axis=2)  # (T, N)
        return np.sqrt(np.mean(squared_distances, axis=1))  # shape (T,)

    else:
        raise ValueError(f"positions must have shape (N, 3) or (T, N, 3), got {positions.shape}")

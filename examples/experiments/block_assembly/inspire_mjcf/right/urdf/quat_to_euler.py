import numpy as np

def quat_to_euler(quat):
    """
    Convert quaternion (x, y, z, w) to Euler angles (roll, pitch, yaw) in radians.
    
    Parameters:
        quat (tuple): A tuple or list containing the quaternion (x, y, z, w).
    
    Returns:
        tuple: A tuple containing the Euler angles (roll, pitch, yaw) in radians.
    """
    # Normalize the quaternion
    x, y, z, w = quat
    norm = np.sqrt(x**2 + y**2 + z**2 + w**2)
    
    if norm == 0:
        raise ValueError("Zero quaternion cannot be normalized.")
    
    x /= norm
    y /= norm
    z /= norm
    w /= norm

    # Roll (x-axis rotation)
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))

    # Pitch (y-axis rotation)
    pitch = np.arcsin(2 * (w * y - z * x))

    # Yaw (z-axis rotation)
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))

    return roll, pitch, yaw

# Example usage
quat = (0.331531, 0.624571, -0.331532, 0.624568)  # Example quaternion representing a 90-degree rotation around the z-axis
euler_angles = quat_to_euler(quat)

# Print results
print("Euler angles (roll, pitch, yaw) in radians:", euler_angles)

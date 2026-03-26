"""
Transform functions for CAST navigation dataset.
"""

import tensorflow as tf
from typing import Any, Dict


def gnm_dataset_transform(trajectory: Dict[str, Any], action_horizon: int = 8) -> Dict[str, Any]:
    """
    Transform CAST trajectories to compute relative waypoints in robot frame.
    
    This transform:
    1. Computes future waypoints over the action horizon
    2. Makes them relative to current position
    3. Rotates them to robot frame
    4. Normalizes by trajectory-specific factor
    
    Args:
        trajectory: Input trajectory dict with observations and actions
        action_horizon: Number of waypoints to predict
        
    Returns:
        Modified trajectory with relative waypoint actions
    """
    traj_len = tf.shape(trajectory["action"])[0]

    # Concatenate position and yaw to create state
    trajectory["observation"]["state"] = tf.concat(
        (trajectory["observation"]["position"], trajectory["observation"]["yaw"]), 
        axis=-1
    )
    
    # Pad trajectory states to handle edge cases at the end
    padding = tf.tile(trajectory["observation"]["state"][-1:, :], [action_horizon, 1])
    trajectory["observation"]["state"] = tf.concat(
        (trajectory["observation"]["state"], padding), axis=0
    )
    
    # Get indices for next action_horizon waypoints
    indices = tf.reshape(tf.range(traj_len), [-1, 1]) + tf.range(1, action_horizon + 1)
    global_waypoints = tf.gather(trajectory["observation"]["state"], indices)[:, :, :2]
    
    # Get current position indices
    curr_pos_indices = tf.reshape(tf.range(traj_len), [-1, 1]) + tf.range(0, action_horizon)
    curr_pos = tf.gather(trajectory["observation"]["state"], curr_pos_indices)[:, :, :2]
    
    # Compute relative waypoints (future positions - current positions)
    global_waypoints -= curr_pos
    global_waypoints = tf.expand_dims(global_waypoints, 2)
    
    # Rotate waypoints to robot frame using yaw rotation matrix
    actions = tf.squeeze(
        tf.linalg.matmul(
            global_waypoints,
            tf.expand_dims(trajectory["observation"]["yaw_rotmat"][:, :2, :2], 1),
        ),
        2,
    )
    
    # Normalize by trajectory-specific normalization factor
    normalization_factor = trajectory["traj_metadata"]["episode_metadata"]["normalization_factor"]
    normalization_factor = tf.cast(normalization_factor[0], tf.float64)
    actions = actions / normalization_factor
    
    # Update trajectory with computed actions
    trajectory["action"] = actions
    trajectory["observation"]["proprio"] = actions
    
    return trajectory



def gnm_action_angle_dataset_transform(trajectory: Dict[str, Any], action_horizon=1) -> Dict[str, Any]:
    traj_len = tf.shape(trajectory["action"])[0]

    # Build full state: [x, y, yaw]
    trajectory["observation"]["state"] = tf.concat(
        (trajectory["observation"]["position"], trajectory["observation"]["yaw"]), axis=-1
    )

    # Pad trajectory states for indexing future steps
    padding = tf.tile(trajectory["observation"]["state"][-1:, :], [action_horizon, 1])
    trajectory["observation"]["state"] = tf.concat(
        (trajectory["observation"]["state"], padding), axis=0
    )

    # Future indices: for each timestep, get the next action_horizon steps
    indices = tf.reshape(tf.range(traj_len), [-1, 1]) + tf.range(1, action_horizon + 1)
    future_states = tf.gather(trajectory["observation"]["state"], indices)  # [T, H, 3]

    # Current indices
    curr_pos_indices = tf.reshape(tf.range(traj_len), [-1, 1]) + tf.range(0, action_horizon)
    curr_states = tf.gather(trajectory["observation"]["state"], curr_pos_indices)  # [T, H, 3]

    # --- Position deltas (same as gnm_dataset_transform) ---
    global_waypoints = future_states[:, :, :2] - curr_states[:, :, :2]

    # Rotate into local frame
    global_waypoints = tf.expand_dims(global_waypoints, 2)  # [T, H, 1, 2]
    local_waypoints = tf.squeeze(
        tf.linalg.matmul(
            global_waypoints,
            tf.expand_dims(trajectory["observation"]["yaw_rotmat"][:, :2, :2], 1),
        ),
        2,
    )  # [T, H, 2]

    # --- Yaw deltas ---
    future_yaw = future_states[:, :, 2:]   # [T, H, 1]
    curr_yaw = curr_states[:, :, 2:]       # [T, H, 1]
    delta_yaw = future_yaw - curr_yaw

    # Normalize yaw delta to [-pi, pi]
    # delta_yaw = tf.math.atan2(tf.math.sin(delta_yaw), tf.math.cos(delta_yaw))

    # --- Yaw delta as sin/cos ---
    sin_yaw = tf.math.sin(delta_yaw)  # [T, H, 1]
    cos_yaw = tf.math.cos(delta_yaw)  # [T, H, 1]

    # --- Normalize positions ---
    normalization_factor = trajectory["traj_metadata"]["episode_metadata"]["normalization_factor"]
    normalization_factor = tf.cast(normalization_factor[0], tf.float64)
    local_waypoints = local_waypoints / normalization_factor

    # Combine: [T, H, 4] = [local_x, local_y, sin_yaw, cos_yaw]
    actions = tf.concat([local_waypoints, sin_yaw, cos_yaw], axis=-1)

    

    trajectory["action"] = actions
    trajectory["observation"]["proprio"] = actions
    return trajectory
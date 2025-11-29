import numpy as np

def map_range(value, in_min, in_max, out_min, out_max):
    """Maps a value from one range to another."""
    try:
        return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
    except ZeroDivisionError:
        return out_min

def lerp(start, end, t):
    """Linear interpolation."""
    return start + (end - start) * t

class EMASmoother:
    """Exponential Moving Average Smoother for 3D points."""
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.state = {}

    def update(self, key, target_value):
        """
        Update the smoothed value for a specific key.
        target_value should be a numpy array or tuple (x, y, z).
        """
        target = np.array(target_value, dtype=np.float32)
        
        if key not in self.state:
            self.state[key] = target
            return target
        
        smoothed = self.state[key] * (1 - self.alpha) + target * self.alpha
        self.state[key] = smoothed
        return smoothed

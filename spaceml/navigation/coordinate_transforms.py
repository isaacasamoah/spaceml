import numpy as np
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timezone

class ReferenceFrame(Enum):
    """Standard reference frames for spacecraft navigation"""

    ECI = "Earth-Centred Inertial"
    ECEF = "Earth-Centred Earth-Fixed"
    BODY = "Body-Fixed"

@dataclass(frozen=True)
class Vector3D:
    """Immutable 3D vector with reference frame tracking"""
    x: float
    y: float
    z: float
    frame: ReferenceFrame
    timestamp: datetime = None
    
    @property
    def magnitude(self) -> float:
        """Calculate vector magnitude (distance from origin)."""
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def to_array(self) -> np.ndarray:
        """convert to NumPy array for matix operations."""
        return np.array([self.x, self.y, self.z])


if __name__ == "__main__":
    # Test vector ISS altitude ~408km

    iss = Vector3D(6_778_000, 0, 0,  ReferenceFrame.ECI)
    print(f"ISS position magnitude: {iss.magnitude:,.0f} metres")
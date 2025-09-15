import numpy as np
import time
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timezone
from dataclasses import dataclass, field 

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
    timestamp: datetime  = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def magnitude(self) -> float:
        """Calculate vector magnitude (distance from origin)."""
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def to_array(self) -> np.ndarray:
        """convert to NumPy array for matix operations."""
        return np.array([self.x, self.y, self.z])
    
@dataclass(frozen=True)
class OrbitState:
    position: Vector3D
    velocity: Vector3D

    def __post_init__(self):
        """Validate position and velocity are in the same reference frame. """
        if self.position.frame != self.velocity.frame:
            raise ValueError("Position and velocity must be in the same reference frame")

class CoordinateTransforms: 
    """Main interface for coorindate transformation using composition pattern"""

    def __init__(self, config=None):
        """Initialise with Transformation registry."""
        # Registry pattern - maps (source, target) -> transformation object
        self.transformations = {}

        # start with basic transformation
        self._register_default_transformations()

    def _register_default_transformations(self):
        """Register the standard coordinate transformations."""
        self.transformations[(ReferenceFrame.ECI, ReferenceFrame.ECEF)] = EciToEcefTransform()

    def transform(self, vector: Vector3D, target_frame: ReferenceFrame) -> Vector3D:
        """Transform vecotr to target reference frame."""
        if vector.frame ==target_frame:
            return vector
        
        transoformation_key = (vector.frame, target_frame)

        if transoformation_key in self.transformations:
            # apply the transformation
            transformer = self.transformations[transoformation_key]
            return transformer.transform(vector)
        else:
            raise ValueError(f"No transfomration avaliable from {vector.frame} to {target_frame}")
                                                                 
class EciToEcefTransform:
    """Transform form Earth-Centred Inertial to Earth-Centred Earth-Fixed frame"""

    #Earth rotation parameters (IAU 2000A model)
    EARTH_ROTATION_RATE = 7.2921159e-5 #rad/s (sidereal rate)
    J2000_EPOCH = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    def transform(self, vector: Vector3D) -> Vector3D:
        """Transform ECI vector to ECEF using Earth rotation."""
        if vector.frame != ReferenceFrame.ECI:
            raise ValueError(f"Expected ECI frame, got {vector.frame}")
        
        # Calculate seconds since J2000 epoch
        dt = (vector.timestamp - self.J2000_EPOCH).total_seconds()

        # Earth rotation angle
        theta = self.EARTH_ROTATION_RATE * dt

        # ECI to ECEF rotation matrix
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        rotation_matrix = np.array([
            [cos_theta, sin_theta, 0],
            [-sin_theta, cos_theta, 0],
            [0, 0, 1]
        ])

        # Apply transformations
        transformed_coords = rotation_matrix @ vector.to_array()

        # Return new Vector3D with ECEF frame
        return Vector3D(
            x = transformed_coords[0],
            y = transformed_coords[1],
            z = transformed_coords[2],
            frame=ReferenceFrame.ECEF,
            timestamp = vector.timestamp
        )




if __name__ == "__main__":

     # Test 1: Default timestamp behavior
    print("=== Testing Vector3D Timestamps ===")
    vector1 = Vector3D(1, 2, 3, ReferenceFrame.ECI)
    print(f"Vector1 timestamp: {vector1.timestamp}")
    
    time.sleep(2)  # Wait 2 seconds
    
    vector2 = Vector3D(4, 5, 6, ReferenceFrame.ECI)
    print(f"Vector2 timestamp: {vector2.timestamp}")
    
    # Test 2: Timestamps should be different
    time_diff = (vector2.timestamp - vector1.timestamp).total_seconds()
    print(f"Time difference: {time_diff:.1f} seconds")
    
    # Test 3: Our existing magnitude test
    iss = Vector3D(6_778_000, 0, 0, ReferenceFrame.ECI)
    print(f"ISS position magnitude: {iss.magnitude:,.0f} meters")
    print(f"ISS timestamp: {iss.timestamp}")

     # Test 4:  Test the transformation
    print("=== Testing ECI→ECEF Transformation ===")
    
    # Create ECI position
    eci_pos = Vector3D(6_778_000, 0, 0, ReferenceFrame.ECI)
    print(f"ECI: ({eci_pos.x/1e6:.3f}, {eci_pos.y/1e6:.3f}, {eci_pos.z/1e6:.3f}) Mm")
    
    # Test transforming it usinig EcitoECefTransform directly
    transformer = EciToEcefTransform()
    ecef_pos = transformer.transform(eci_pos)
    print(f"ECEF: ({ecef_pos.x/1e6:.3f}, {ecef_pos.y/1e6:.3f}, {ecef_pos.z/1e6:.3f}) Mm")

    # Tesst the complete system
    transforms = CoordinateTransforms()
    eci_pos = Vector3D(6_778_00,0, 0, ReferenceFrame.ECI)
    ecef_pos = transforms.transform(eci_pos, ReferenceFrame.ECEF)

    print(f"Registry transform worked! ECEF: ({ecef_pos.x/1e6:.3f}, {ecef_pos.y/1e-6:.3f},{ecef_pos.z}) Mm")

    # Test OrbitState validation
    print("=== Testing OrbitState Error Handling ===")
    
    try:
        # This should work fine
        pos = Vector3D(6_778_000, 0, 0, ReferenceFrame.ECI)
        vel = Vector3D(0, 7_500, 0, ReferenceFrame.ECI)
        orbit = OrbitState(pos, vel)
        print("✅ Valid OrbitState created successfully")
        
        # This should raise an error
        bad_vel = Vector3D(0, 7_500, 0, ReferenceFrame.ECEF)  # Different frame!
        bad_orbit = OrbitState(pos, bad_vel)
        print("❌ ERROR: Should have failed!")
        
    except ValueError as e:
        print(f"✅ Caught expected error: {e}")
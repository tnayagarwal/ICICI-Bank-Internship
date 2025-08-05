"""
GPS and Location Utilities
Handles location verification and distance calculations
"""
import math
from typing import Tuple, Optional, Dict, Any
from geopy.distance import geodesic
from loguru import logger
from config import settings

class LocationError(Exception):
    """Custom exception for location-related errors"""
    pass

class GPSUtils:
    def __init__(self):
        self.office_coords = settings.OFFICE_COORDS
        self.attendance_radius = settings.ATTENDANCE_RADIUS_METERS
        
    def calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate distance between two coordinates in meters
        
        Args:
            lat1, lon1: First coordinate pair
            lat2, lon2: Second coordinate pair
            
        Returns:
            Distance in meters
        """
        try:
            distance = geodesic((lat1, lon1), (lat2, lon2)).meters
            return distance
        except Exception as e:
            logger.error(f"Distance calculation error: {str(e)}")
            raise LocationError(f"Failed to calculate distance: {str(e)}")
    
    def is_within_office_radius(self, lat: float, lon: float) -> bool:
        """
        Check if coordinates are within office attendance radius
        
        Args:
            lat, lon: Coordinates to check
            
        Returns:
            True if within radius, False otherwise
        """
        try:
            distance = self.calculate_distance(
                lat, lon, 
                self.office_coords[0], self.office_coords[1]
            )
            return distance <= self.attendance_radius
        except LocationError:
            return False
    
    def get_location_status(self, lat: float, lon: float) -> Dict[str, Any]:
        """
        Get detailed location status for attendance
        
        Args:
            lat, lon: Current coordinates
            
        Returns:
            Dictionary with location details
        """
        try:
            distance = self.calculate_distance(
                lat, lon,
                self.office_coords[0], self.office_coords[1]
            )
            
            within_range = distance <= self.attendance_radius
            
            return {
                "latitude": lat,
                "longitude": lon,
                "office_latitude": self.office_coords[0],
                "office_longitude": self.office_coords[1],
                "distance_meters": round(distance, 2),
                "within_range": within_range,
                "attendance_radius": self.attendance_radius,
                "status": "ALLOWED" if within_range else "OUT_OF_RANGE"
            }
        except Exception as e:
            logger.error(f"Location status error: {str(e)}")
            return {
                "error": str(e),
                "status": "ERROR"
            }
    
    def validate_coordinates(self, lat: float, lon: float) -> bool:
        """
        Validate coordinate values
        
        Args:
            lat, lon: Coordinates to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
            return False
        
        if not (-90 <= lat <= 90):
            return False
            
        if not (-180 <= lon <= 180):
            return False
            
        return True

# Global GPS utility instance
gps_utils = GPSUtils() 
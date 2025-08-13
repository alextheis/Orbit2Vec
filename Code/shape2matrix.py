import shapefile
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import transform # Optional, for reprojecting coordinates if needed
import torch
import kagglehub
import os

class shape2matrix:
    def __init__(self, num):
        self.path = ""
        self.num_point = num if num else 50


    def importShape(self):
        # Download dataset folder
        folder_path = kagglehub.dataset_download(
            self.path
        )

        # Find the first .shp file in that folder
        shp_file = None
        for root, _, files in os.walk(folder_path):
            for f in files:
                if f.lower().endswith(".shp"):
                    shp_file = os.path.join(root, f)
                    break
            if shp_file:
                break

        if not shp_file:
            raise FileNotFoundError("No .shp file found in downloaded dataset")

        # Return shapefile.Reader instance
        return shapefile.Reader(shp_file)


    def extractShape(self, reader):
        sf = reader
        all_exteriors = []

        for shape_record in sf.iterShapeRecords():
            shape = shape_record.shape

            # Only process Polygon shapes
            if shape.shapeType == shapefile.POLYGON and len(shape.parts) > 0:
                start_index = shape.parts[0]
                end_index = shape.parts[1] if len(shape.parts) > 1 else len(shape.points)

                # Collect coordinates for the exterior ring (always as [x, y] lists)
                exterior_ring = [[x, y] for x, y in shape.points[start_index:end_index]]

                all_exteriors.append(exterior_ring)

        return all_exteriors
    
    def equidistant(self, polygon_data):
        """
        For each polygon in polygon_data, generate num_point equidistant points along
        the exterior ring of the polygon and return them as a list of PyTorch tensors.
        
        Args:
            polygon_data: List of polygons, where each polygon is a list of [x, y] pairs
                        representing the exterior ring.
        
        Returns:
            List of torch.Tensor of shape (num_point, 2) for each polygon.
        """
        num_equidistant_points = self.num_point
        list_of_matrices = []

        for exterior_ring_coords in polygon_data:
            # Create a Shapely LineString from the exterior ring's coordinates
            line = LineString(exterior_ring_coords)

            # Calculate total perimeter length
            perimeter_length = line.length

            # Distance between each equidistant point
            segment_length = perimeter_length / (num_equidistant_points - 1)

            # Generate equidistant points along the perimeter
            equidistant_points = [
                [line.interpolate(i * segment_length).x, line.interpolate(i * segment_length).y]
                for i in range(num_equidistant_points)
            ]

            # Convert to PyTorch tensor
            matrix = torch.tensor(equidistant_points, dtype=torch.float)
            list_of_matrices.append(matrix)

        return list_of_matrices
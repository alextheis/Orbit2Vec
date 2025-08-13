from shapely.geometry import Polygon, LineString, Point
from shapely.ops import transform # Optional, for reprojecting coordinates if needed
import torch

# Assuming polygon_data is loaded from your 'polygon_coordinates.py' file
# For demonstration, let's create a sample polygon data if not loaded
polygon_data = [
    [
        [
            (0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0), (0.0, 0.0)
        ]
    ]
]

# Number of equidistant points you want
num_equidistant_points = 50

equidistant_points_for_all_polygons = []

list_of_matrices = []

for single_polygon_coords in polygon_data:
    # A single_polygon_coords list might contain multiple rings (exterior + interior)
    # We are interested in the exterior ring for the boundary
    exterior_ring_coords = single_polygon_coords[0]

    # Create a Shapely LineString from the exterior ring's coordinates
    # The last point is typically duplicated to close the polygon, but LineString doesn't need it for length calculation
    line = LineString(exterior_ring_coords)

    # Calculate the length of the polygon's perimeter
    perimeter_length = line.length

    # Calculate the distance between equidistant points
    # We need num_equidistant_points, so (num_equidistant_points - 1) segments
    segment_length = perimeter_length / (num_equidistant_points - 1)

    # Generate the equidistant points using interpolate
    equidistant_points = []
    for i in range(num_equidistant_points):
        distance = i * segment_length
        point = line.interpolate(distance)
        equidistant_points.append((point.x, point.y)) # Store as (x, y) tuple 

    equidistant_points_for_all_polygons.append(equidistant_points)  # equidistant_points should be a 2x50 matrix 

    matrix = torch.tensor(equidistant_points)
    list_of_matrices.append(matrix)

# # Now, equidistant_points_for_all_polygons contains the desired points
# for i, points_list in enumerate(equidistant_points_for_all_polygons):
#     print(f"Polygon {i+1} - Equidistant Points:")
#     for point in points_list:
#         print(f"  ({point[0]:.4f}, {point[1]:.4f})")
import shapefile

# Replace 'path/to/your/shapefile.shp' with the actual path to your .shp file
try:
    sf = shapefile.Reader("path/to/your/shapefile.shp")
    print("Shapefile opened successfully!")
except shapefile.ShapefileException as e:
    print(f"Error opening shapefile: {e}")
    exit()

all_polygon_coordinates = []

for shape_record in sf.iterShapeRecords():
    # Each shape_record contains a shape object and a record (attributes)
    shape = shape_record.shape

    # We're interested in Polygon shapes (shapeType 5)
    if shape.shapeType == shapefile.POLYGON:
        polygon_coords = []

        # Extract coordinates for the exterior ring
        # A shapefile polygon's 'points' attribute provides the coordinates of the polygon's exterior and interior rings sequentially.
        # The 'parts' attribute indicates the starting index of each ring within 'points'.
        for i, part_index in enumerate(shape.parts):
            ring_coords = []
            # Determine the end index of the current part (ring)
            end_index = shape.parts[i+1] if i + 1 < len(shape.parts) else len(shape.points)

            # Iterate through the points (vertices) of the current ring
            for j in range(part_index, end_index):
                point = shape.points[j]
                ring_coords.append((point[0], point[1])) # Store as (x, y) tuple

            polygon_coords.append(ring_coords)
        all_polygon_coordinates.append(polygon_coords)

# # Now, save these coordinates to a Python file
# output_filename = "polygon_coordinates.py"
# with open(output_filename, 'w') as f:
#     f.write("polygon_data = [\n")
#     for polygon_coords in all_polygon_coordinates:
#         f.write("    [\n")
#         for ring_coords in polygon_coords:
#             f.write("        [\n")
#             for coord_pair in ring_coords:
#                 f.write(f"            {coord_pair},\n")
#             f.write("        ],\n")
#         f.write("    ],\n")
#     f.write("]\n")

# print(f"Polygon coordinates saved to {output_filename}")

# # You can then import this file into another Python script:
# # from polygon_coordinates import polygon_data
# # print(polygon_data[0])

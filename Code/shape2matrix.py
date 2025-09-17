from typing import List, Optional, Tuple, Union
import shapefile
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import transform # Optional, for reprojecting coordinates if needed
import torch
import kagglehub
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

class shape2matrix:
    def __init__(self, num: Optional[int] = None) -> None:
        """_summary_

        Args:
            num (Optional[int], optional): _description_. Defaults to None.
        """
        self.path = ""
        self.num_point = num if num else 50


    def import_shape(self) -> shapefile.Reader:
        """_summary_

        Raises:
            FileNotFoundError: _description_

        Returns:
            shapefile.Reader: _description_
        """
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


    def extract_shape(self, reader: shapefile.Reader) -> List[List[List[float]]]:
        """_summary_

        Args:
            reader (shapefile.Reader): _description_

        Returns:
            List[List[List[float]]]: _description_
        """
    
        sf = reader
        all_exteriors = []

        for shape_record in sf.iterShapeRecords():
            shape = shape_record.shape

            # Only process Polygon shapes
            if shape.shapeType == shapefile.POLYGON and len(shape.parts) > 0:
                start_index = shape.parts[0]

                if len(shape.parts) > 1:
                    end_index = shape.parts[1]
                else:
                    end_index = len(shape.points)


                # Collect coordinates for the exterior ring (always as [x, y] lists)
                exterior_ring = []

                for x, y in shape.points[start_index:end_index]:
                    exterior_ring.append([x,y])

                all_exteriors.append(exterior_ring)

        return all_exteriors
    
    def equidistant(self, polygon_data: List[List[List[float]]]) -> List[torch.Tensor]:
        """_summary_

        Args:
            polygon_data (List[List[List[float]]]): _description_

        Returns:
            List[torch.Tensor]: _description_
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
            equidistant_points = []
            for i in range(num_equidistant_points):

                equidistant_points.append([line.interpolate(i * segment_length).x, line.interpolate(i * segment_length).y]) 

            equidistant_points.pop()
            # Convert to PyTorch tensor
            matrix = torch.tensor(equidistant_points, dtype=torch.float)
            list_of_matrices.append(matrix)

        return list_of_matrices
    
    def pca(self, data: List[Union[torch.Tensor, np.ndarray, float, int]]) -> np.ndarray:
        # Convert data to vector format
        if isinstance(data[0], torch.Tensor):
            vector = np.array([item.item() if item.dim() == 0 else item.numpy().flatten()[0] for item in data])
        else:
            vector = np.array([item if np.isscalar(item) else item.flatten()[0] for item in data])
        
        # Create 2D array for PCA: [value, index] - THIS IS THE KEY FIX
        indices = np.arange(len(vector))
        stacked = np.column_stack([vector, indices])  # Now you have 2 features!
        
        # Standardize features
        scaler = StandardScaler()
        stacked_scaled = scaler.fit_transform(stacked)  # Now (441, 2)
        
        # Apply PCA - now you can get 2 components!
        pca = PCA(n_components=2)
        stacked_pca = pca.fit_transform(stacked_scaled)
        
        # Plot
        plt.figure(figsize=(12, 4))
        
        # Original vector plot
        plt.subplot(1, 2, 1)
        plt.plot(vector, 'bo-', markersize=6)
        plt.title("Filter Bank Outputs")
        plt.xlabel("Shape Index")
        plt.ylabel("Filter Response Value")
        plt.grid(True, alpha=0.3)
        
        # PCA plot - now you have 2 dimensions!
        plt.subplot(1, 2, 2)
        scatter = plt.scatter(stacked_pca[:,0], stacked_pca[:,1], c=indices, cmap='viridis', s=50)
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
        plt.title("PCA Space")
        plt.colorbar(scatter, label='Shape Index')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return stacked_pca
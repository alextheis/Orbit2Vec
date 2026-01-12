import math
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
from sklearn.manifold import TSNE
# from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def _values_to_grayscale(values):
    """
    Map a 1D array of values to grayscale shades in [0, 1],
    where higher values -> darker (closer to 0).
    """
    v = np.array(values, dtype=float)
    v_min, v_max = v.min(), v.max()

    if v_max == v_min:
        # All values are the same: use mid-gray
        norm = np.full_like(v, 0.5, dtype=float)
    else:
        # Normalize to [0, 1]
        norm = (v - v_min) / (v_max - v_min)

    # Invert so larger values -> darker (0 = black, 1 = white in 'gray' cmap)
    shades = 1.0 - norm
    return shades

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
        # FIX: Download once and store locally 
        # Download dataset folder
        # folder_path = kagglehub.dataset_download(
        #     self.path
        # )
        folder_path = r"C:\Users\16147\OneDrive - The Ohio State University\3345\orbit2vec\Code\unit_tests\tl_2019_us_cd116.dbf"

        # Find the first .shp file in that folder
        # shp_file = None
        # for root, _, files in os.walk(folder_path):
        #     for f in files:
        #         if f.lower().endswith(".shp"):
        #             shp_file = os.path.join(root, f)
        #             break
        #     if shp_file:
        #         break

        # if not shp_file:
        #     raise FileNotFoundError("No .shp file found in downloaded dataset")

        # Return shapefile.Reader instance
        return shapefile.Reader(folder_path)


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
        # Convert all shapes into a consistent 2D matrix: (num_shapes, num_features)
        X = []
        for item in data:
            if isinstance(item, torch.Tensor):
                arr = item.cpu().numpy()
            else:
                arr = np.array(item)
            X.append(arr.flatten())
        X = np.array(X)   # shape = (num_shapes, num_features)

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Apply PCA (get top 2 real components)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        # X_pca = X
        print(X_pca[:, 0])
        print(X_pca[:, 1])
        # Plot
        plt.figure(figsize=(6, 6))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1],
                            c=np.arange(len(X)), cmap='viridis', s=50)
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
        plt.title("PCA of Shapes")
        plt.colorbar(scatter, label="Shape Index")
        plt.grid(True, alpha=0.3)
        plt.show()

        return X_pca
    
    def center(self, equidistant_data: List[List[List[float]]]) -> List[torch.Tensor]:
        centered_data = []

        for shape in equidistant_data:
            # Convert list of pairs to numpy array

            if isinstance(shape, torch.Tensor):
                pts = shape.detach().cpu().numpy()
            else:
                pts = np.asarray(shape, dtype=float)  
            centroid = np.mean(pts, axis=0)     
            centered_pts = pts - centroid       # shift to center at origin

            # Normalize by Frobenius norm
            f = np.linalg.norm(centered_pts)
            if f > 0:
                centered_pts /= f

            # Append as torch tensor
            centered_data.append(torch.tensor(centered_pts))

        return centered_data

    def tsne(self,X, random_state, n_components=2, values=None, value_label="Value", outlines=None, outline_scale=3.0):
        """
        Run t-SNE on feature vectors X. If `outlines` is provided, draw each
        district boundary at its t-SNE location.

        Parameters
        ----------
        X : torch.Tensor or array-like, shape (N, D)
            Feature vectors (e.g. shape2matrix result).
        random_state : int
            Seed for t-SNE.
        n_components : int
            t-SNE dimensionality (usually 2).
        values : array-like or None
            Optional scalar per shape for grayscale shading.
        value_label : str
            Label for colorbar.
        outlines : None or array-like
            Optional polygon list for drawing. Each element can be:
            - shape (M, 2): M (x,y) points, or
            - shape (2M,): flattened [x1,y1,...,xM,yM].
            There must be one outline per row of X.
        outline_scale : float
            Multiplier for the outline size relative to t-SNE distances.
        """

        # Convert X to numpy 
        if isinstance(X, torch.Tensor):
            X_np = X.detach().cpu().numpy()
        else:
            X_np = np.asarray(X, dtype=float)

        if X_np.ndim != 2:
            raise ValueError(f"X must be 2D (N,D); got shape {X_np.shape}")

        N = X_np.shape[0]

        # Run t-SNE
        tsne_model = TSNE(n_components=n_components, random_state=random_state)
        X_embedded = tsne_model.fit_transform(X_np)

        # Plot if 2D
        if n_components == 2:
            fig, ax = plt.subplots(figsize=(8, 6))

            # Optional scalar values -> grayscale scatter
            if values is not None:
                values = np.asarray(values, dtype=float)
                if len(values) != N:
                    raise ValueError(
                        f"len(values)={len(values)} does not match number of shapes={N}"
                    )
                shades = _values_to_grayscale(values)
                scatter = ax.scatter(
                    X_embedded[:, 0],
                    X_embedded[:, 1],
                    s=4,
                    c=shades,
                    cmap="gray",
                )
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label(value_label)
            else:
                ax.scatter(X_embedded[:, 0], X_embedded[:, 1], s=4, color="black")

            # If outlines provided -> draw them
            if outlines is not None:
                # convert to numpy array of objects to keep per-shape flexibility
                if isinstance(outlines, torch.Tensor):
                    outlines_list = outlines.detach().cpu().numpy()
                else:
                    outlines_list = list(outlines)

                if len(outlines_list) != N:
                    raise ValueError(
                        f"len(outlines)={len(outlines_list)} does not match N={N}"
                    )
            
                for i in range(N):
                    boundary = np.asarray(outlines_list[i], dtype=float)

                    # If boundary is flat (2M,), reshape to (M,2)
                    if boundary.ndim == 1:
                        if boundary.size % 2 != 0:
                            raise ValueError(
                                f"Outline {i} has odd length {boundary.size}; "
                                "cannot reshape to (M,2)."
                            )
                        boundary = boundary.reshape(-1, 2)

                    if boundary.shape[1] != 2:
                        raise ValueError(
                            f"Outline {i} must have shape (M,2); got {boundary.shape}"
                        )

                    # centroid in original coordinates
                    cx, cy = boundary.mean(axis=0)
                    px, py = X_embedded[i]

                    # shift + scale so it is visible
                    shifted = (boundary - np.array([cx, cy])) * outline_scale \
                            + np.array([px, py])

                    ax.plot(
                        shifted[:, 0],
                        shifted[:, 1],
                        color="0.5",
                        linewidth=0.7,
                    )

            ax.set_title("t-SNE Embedding of Shape Dataset")
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")
            plt.tight_layout()
            plt.show()

        return X_embedded         
                
    def compute_diameter(self, shapes):
        """
        Compute the diameter (maximum pairwise distance) for each shape.

        Args:
            shapes:
                - list or tuple of shapes, each shape array-like [N, 2]
                - torch.Tensor of shape [num_shapes, N, 2] or [N, 2]

        Returns:
            List[float]: diameter for each shape.
        """

        # Normalize into iterable list of shape tensors
        if isinstance(shapes, torch.Tensor):
            if shapes.ndim == 3 and shapes.shape[2] == 2:      # [num_shapes, N, 2]
                shapes = [shapes[i] for i in range(shapes.shape[0])]
            elif shapes.ndim == 2 and shapes.shape[1] == 2:    # [N, 2] — single shape
                shapes = [shapes]
            else:
                raise ValueError(f"Tensor must be [num_shapes, N, 2] or [N, 2], got {shapes.shape}")
        else:
            shapes = list(shapes)

        diameters = []

        for pts in shapes:
            pts = torch.as_tensor(pts, dtype=torch.float32)
            if pts.ndim != 2 or pts.shape[1] != 2:
                raise ValueError(f"Each shape must be [N, 2], got {pts.shape}")

            # Convert to numpy for SciPy
            pts_np = pts.cpu().numpy()

            # Compute full pairwise distance matrix
            D = cdist(pts_np, pts_np, metric="euclidean")

            # Diameter = max distance in matrix
            diameters.append(D.max())

        return diameters

    def compute_perimeter(self, coordinates):
        """
        Calculates the perimeter of a polygon given a list of its coordinates.

        Args:
            coordinates: A list of tuples, where each tuple is an (x, y) coordinate.

        Returns:
            The perimeter of the polygon.
        """
        
        perimeter = 0
        num_points = len(coordinates)
        for i in range(num_points):
            # Get current point and the next point
            x1, y1 = coordinates[i]
            # Use the modulo operator to wrap around to the first point from the last
            x2, y2 = coordinates[(i + 1) % num_points]

            # Calculate the distance between the two points
            distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            perimeter += distance
        return perimeter
    
    # calculates the area of the shape given the coordinates
    def shoelace_formula(self, shape, absoluteValue = True):
        x, y =  zip(*shape)
        result = 0.5 * np.array(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        if absoluteValue:
            return abs(result)
        else:
            return result
        
    # gives the "roundness" of a shape
    def roundness(self, shape):
        area = self.shoelace_formula(shape)
        perimeter = self.compute_perimeter(shape)
         
        return 4 * math.pi * (area / (perimeter ** 2))


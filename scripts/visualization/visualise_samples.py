import os
import numpy as np
import pickle
import SimpleITK as sitk
import vtk
from vtk.util import numpy_support
import argparse
from torch.utils.data import Subset

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize 3D TIFF volumes from test set")
    parser.add_argument('--test_idx', type=int, default=0, 
                        help='Index in test set to visualize')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing data')
    parser.add_argument('--save_only', action='store_true',
                        help='Save images without displaying')
    return parser.parse_args()

def load_tiff(file_path):
    """Load a 3D TIFF file using SimpleITK"""
    img = sitk.ReadImage(file_path)
    return sitk.GetArrayFromImage(img)

class TiffDataset:
    def __init__(self, data):
        self.data = []
        
        for item in data:
            tiff_files = item['images']
            if len(tiff_files) == 9:  # Ensure we have all 9 images
                self.data.append({
                    'images': tiff_files,
                    'path': '/'.join(tiff_files[0].split(os.sep)[-3:-1])  # Extract animal/chunk path
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        
        # Load all 9 volumes
        volumes = [load_tiff(img) for img in item['images']]
        # Convert to numpy array of shape (9, D, H, W)
        volumes = np.array(volumes, dtype=np.float32)
        
        return {
            "volumes": volumes,
            "path": item['path']
        }

def create_volume_actor(volume_data, color_func, iso_value=0.1):
    """Create a VTK actor for volume rendering"""
    # Convert volume to VTK data
    volume_data = volume_data.astype(np.float32)
    
    # Normalize data to 0-1 range if needed
    if volume_data.max() > 1.0 or volume_data.min() < 0.0:
        data_min, data_max = volume_data.min(), volume_data.max()
        if data_min != data_max:
            volume_data = (volume_data - data_min) / (data_max - data_min)
    
    # Reshape for VTK (z, y, x) to (x, y, z)
    volume_data = np.transpose(volume_data, (2, 1, 0))
    
    # Create VTK data object
    data_importer = vtk.vtkImageImport()
    data_string = volume_data.tobytes()
    data_importer.CopyImportVoidPointer(data_string, len(data_string))
    data_importer.SetDataScalarTypeToFloat()
    data_importer.SetNumberOfScalarComponents(1)
    extent = volume_data.shape
    data_importer.SetDataExtent(0, extent[0]-1, 0, extent[1]-1, 0, extent[2]-1)
    data_importer.SetWholeExtent(0, extent[0]-1, 0, extent[1]-1, 0, extent[2]-1)
    
    # Create surface using marching cubes
    surface = vtk.vtkMarchingCubes()
    surface.SetInputConnection(data_importer.GetOutputPort())
    surface.SetValue(0, iso_value)
    
    # Create mapper and actor
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(surface.GetOutputPort())
    mapper.ScalarVisibilityOn()
    
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color_func[0], color_func[1], color_func[2])
    actor.GetProperty().SetOpacity(1.0)
    actor.GetProperty().SetSpecular(0.2)
    actor.GetProperty().SetSpecularPower(10)
    
    return actor

def create_color_functions():
    """Define color functions for different volume types"""
    return {
        'binary': (0.0, 0.0, 1.0),  # Blue 
        'so2': (1.0, 0.0, 0.0),     # Red
        'velocity': (0.0, 1.0, 0.0), # Green
        'vx': (1.0, 0.5, 0.0),      # Orange
        'vy': (0.5, 0.0, 1.0),      # Purple
        'vz': (0.0, 1.0, 1.0)       # Cyan
    }

def save_render(renderer, window, filename, size=(800, 800)):
    """Save the rendered image to a file"""
    window.SetSize(size)
    window.Render()
    
    # Create image filter
    win_to_img = vtk.vtkWindowToImageFilter()
    win_to_img.SetInput(window)
    win_to_img.Update()
    
    # Write to file
    writer = vtk.vtkPNGWriter()
    writer.SetFileName(filename)
    writer.SetInputConnection(win_to_img.GetOutputPort())
    writer.Write()

def visualize_volumes(sample, save_only=False):
    """Visualize volumes using VTK for realistic 3D rendering"""
    volumes = sample["volumes"]
    path = sample["path"]
    
    # Create output directory
    OUTPUT_DIR = 'results/3d_vis'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Volume indices and names
    volume_indices = {
        'binary': 0,
        'so2': 3,
        'velocity': 5,
        'vx': 6,
        'vy': 7,
        'vz': 8
    }
    
    # Different iso values for different volumes
    iso_values = {
        'binary': 0.5,    # Binary is 0/1
        'so2': 0.1,       # Lower threshold for continuous values
        'velocity': 0.1,
        'vx': 0.1,
        'vy': 0.1,
        'vz': 0.1
    }
    
    color_funcs = create_color_functions()
    
    # Process each volume type
    for vol_name, vol_idx in volume_indices.items():
        render_window = vtk.vtkRenderWindow()
        renderer = vtk.vtkRenderer()
        render_window.AddRenderer(renderer)
        
        # Set background to white
        renderer.SetBackground(1, 1, 1)
        
        # Create volume rendering
        volume_data = volumes[vol_idx]
        
        # For velocity components, we have positive and negative values
        # Handle them separately for better visualization
        if vol_name in ['vx', 'vy', 'vz']:
            # Mask for positive values
            pos_mask = volume_data > 0
            if pos_mask.any():
                pos_vol = np.zeros_like(volume_data)
                pos_vol[pos_mask] = volume_data[pos_mask]
                pos_actor = create_volume_actor(pos_vol, (1.0, 0.0, 0.0), iso_values[vol_name])  # Red for positive
                renderer.AddActor(pos_actor)
            
            # Mask for negative values
            neg_mask = volume_data < 0
            if neg_mask.any():
                neg_vol = np.zeros_like(volume_data)
                neg_vol[neg_mask] = -volume_data[neg_mask]  # Make positive for rendering
                neg_actor = create_volume_actor(neg_vol, (0.0, 0.0, 1.0), iso_values[vol_name])  # Blue for negative
                renderer.AddActor(neg_actor)
        else:
            # Standard rendering for other volumes
            if vol_name == 'binary':
                # For binary, use direct threshold
                iso_val = iso_values[vol_name]
            else:
                # For others, mask with binary and use lower threshold
                binary_mask = volumes[0] > 0.5
                volume_data = volume_data * binary_mask
                iso_val = iso_values[vol_name]
            
            actor = create_volume_actor(volume_data, color_funcs[vol_name], iso_val)
            renderer.AddActor(actor)
        
        # Add axes for orientation
        axes = vtk.vtkAxesActor()
        axes.SetTotalLength(10, 10, 10)
        axes.SetShaftTypeToLine()
        axes.SetNormalizedShaftLength(0.85, 0.85, 0.85)
        axes.SetNormalizedTipLength(0.15, 0.15, 0.15)
        
        # Scale axes and position in corner
        axes.SetScale(0.1)
        axes.SetPosition(0, 0, 0)
        renderer.AddActor(axes)
        
        # Reset camera to show entire volume
        renderer.ResetCamera()
        
        # Set optimal viewpoint
        camera = renderer.GetActiveCamera()
        camera.Elevation(30)
        camera.Azimuth(30)
        camera.Zoom(0.9)
        
        # Save image
        output_file = f'{OUTPUT_DIR}/{vol_name}_{path.replace("/", "_")}.png'
        save_render(renderer, render_window, output_file)
        print(f"Saved {vol_name} visualization to {output_file}")
        
        if not save_only:
            # Create interactive window
            render_window_interactor = vtk.vtkRenderWindowInteractor()
            render_window_interactor.SetRenderWindow(render_window)
            
            # Set up trackball camera style
            style = vtk.vtkInteractorStyleTrackballCamera()
            render_window_interactor.SetInteractorStyle(style)
            
            # Initialize and start
            render_window.SetSize(800, 800)
            render_window.SetWindowName(f"{vol_name.upper()} - {path}")
            render_window_interactor.Initialize()
            render_window.Render()
            
            # Start if not in headless mode
            if os.environ.get('DISPLAY') is not None:
                render_window_interactor.Start()

def main():
    args = parse_args()
    
    # Load data_list and test_indices
    with open(os.path.join('splits', 'data_list.pkl'), 'rb') as f:
        data_list = pickle.load(f)
    
    with open(os.path.join('splits', 'test_indices.pkl'), 'rb') as f:
        test_indices = pickle.load(f)
    
    # Create dataset and get the specified test sample
    test_ds = TiffDataset(data_list)
    test_dataset = Subset(test_ds, test_indices)
    
    if args.test_idx >= len(test_dataset):
        print(f"Error: test_idx {args.test_idx} is out of bounds. Max index is {len(test_dataset)-1}")
        return
    
    # Get the sample and visualize
    sample = test_dataset[args.test_idx]
    visualize_volumes(sample, args.save_only)
    
    print(f"Total test samples available: {len(test_dataset)}")

if __name__ == "__main__":
    main()
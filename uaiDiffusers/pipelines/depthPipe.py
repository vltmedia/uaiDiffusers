import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from PIL import Image
import torch
from transformers import GLPNImageProcessor, GLPNForDepthEstimation
import numpy as np
import open3d as o3d



def GenerateDepthFromImage(glpnImageProcessor = "vinvino02/glpn-nyu", glpnForDepthEstimation = "vinvino02/glpn-nyu", image = "image.jpg"):
    """
    Generate a depth image from a given image.
    
    Args:
        glpnImageProcessor (str): The GLPN Image processor to use, usually the same as the deph estimation model.
        glpnForDepthEstimation (str): The GLPN Depth Estimation model to use.
        image (str): The path to the image to use.
        
    Returns:
        numpy.ndarray: The depth image as a numpy array.
        PIL.Image: The depth image as a PIL Image.
    """
    feature_extractor = GLPNImageProcessor.from_pretrained(glpnImageProcessor)
    model = GLPNForDepthEstimation.from_pretrained(glpnForDepthEstimation)

    # load and resize the input image
    if isinstance(image, str):
        image = Image.open(image)
    new_height = 480 if image.height > 480 else image.height
    new_height -= (new_height % 32)
    new_width = int(new_height * image.width / image.height)
    diff = new_width % 32
    new_width = new_width - diff if diff < 16 else new_width + 32 - diff
    new_size = (new_width, new_height)
    image = image.resize(new_size)

    # prepare image for the model
    inputs = feature_extractor(images=image, return_tensors="pt")

    # get the prediction from the model
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # remove borders
    pad = 16
    output = predicted_depth.squeeze().cpu().numpy() * 1000.0
    output = output[pad:-pad, pad:-pad]
    image = image.crop((pad, pad, image.width - pad, image.height - pad))
    return output, image



def GeneratePointsFromDepth(numpyDepthImage):
    """
    Generate Points from a Depth image
    
    Args:
        numpyDepthImage (numpy.ndarray): Depth image as a numpy array
    
    Returns:
        numpy.ndarray: Points as a numpy array
    """

    width, height = image.size

    depth_image = (numpyDepthImage * 255 / np.max(numpyDepthImage)).astype('uint8')
    image = np.array(image)

    # create rgbd image
    depth_o3d = o3d.geometry.Image(depth_image)
    image_o3d = o3d.geometry.Image(image)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(image_o3d, depth_o3d, convert_rgb_to_intensity=False)

    # camera settings
    camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
    camera_intrinsic.set_intrinsics(width, height, 500, 500, width/2, height/2)

    # create point cloud
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)
    return pcd

def GenerateMeshFromPoints(pointCloud, quality = 10, threads = 1):
    """
    Generate a mesh from a point cloud.
    
    Args:
        pointCloud (open3d.geometry.PointCloud): The point cloud to generate a mesh from.
        quality (int): The quality of the mesh.
        threads (int): The number of threads to use.
    
    Returns:
        open3d.geometry.TriangleMesh: The generated mesh.
    
    """
    # outliers removal
    cl, ind = pointCloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=20.0)
    pointCloud = pointCloud.select_by_index(ind)

    # estimate normals
    pointCloud.estimate_normals()
    pointCloud.orient_normals_to_align_with_direction()

    # surface reconstruction
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pointCloud, depth=quality, n_threads=threads)[0]

    # rotate the mesh
    rotation = mesh.get_rotation_matrix_from_xyz((np.pi, 0, 0))
    mesh.rotate(rotation, center=(0, 0, 0))
    return mesh

    
def SaveOutMesh( mesh, outputPath = './mesh.obj'):
    """
    Save out a mesh to a file.
    
    Args:
        mesh (open3d.geometry.TriangleMesh): The mesh to save out.
    """
    o3d.io.write_triangle_mesh(outputPath, mesh)
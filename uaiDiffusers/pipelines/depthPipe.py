import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from PIL import Image
import torch
from transformers import GLPNImageProcessor, GLPNForDepthEstimation
import numpy as np
import open3d as o3d
import pymeshlab
import cv2



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
    del model
    del feature_extractor
    return output, image



def GeneratePointsFromDepth(numpyDepthImage, convertedPILImage):
    """
    Generate Points from a Depth image
    
    Args:
        numpyDepthImage (numpy.ndarray): Depth image as a numpy array
    
    Returns:
        numpy.ndarray: Points as a numpy array
    """

    width, height = convertedPILImage.size

    depth_image = (numpyDepthImage * 255 / np.max(numpyDepthImage)).astype('uint8')
    image = np.array(convertedPILImage)

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

def GenerateMeshFromPoints(pointCloud, quality = 10, textureSize =512, threads = 1, tempPath = "./temp_mesh.obj"):
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
    rotation = mesh.get_rotation_matrix_from_xyz((np.pi, 0, 0))
    mesh.rotate(rotation, center=(0, 0, 0))
    tempMesh = f"{tempPath}/temp_mesh.obj"
    
    SaveOutMesh(mesh, tempMesh)
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(o3d.io.read_triangle_mesh(tempPath))
    
    mesh.compute_uvatlas(size = textureSize)
    # rotate the mesh
    return mesh


def GenerateMeshFromPointsPymeshlab(pointCloud, cv2RGBA, quality = 10, scale = 1.1):
    """
    Generate a mesh from a point cloud.
    
    Args:
        pointCloud (open3d.geometry.PointCloud): The point cloud to generate a mesh from.
        quality (int): The quality of the mesh.
        threads (int): The number of threads to use.
    
    Returns:
        pymeshlab.Mesh: The generated mesh.
    
    """
    # convert cv2RGBA to  numpy.ndarray[numpy.float64[m, 4]] = array([], shape=(0, 4), dtype=float64)
    cv2RGBA = np.array(cv2RGBA)
    cl, ind = pointCloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=20.0)
    pointCloud = pointCloud.select_by_index(ind)

    # estimate normals
    pointCloud.estimate_normals()
    pointCloud.orient_normals_to_align_with_direction()
    
    # outliers removal
    m = pymeshlab.Mesh(pointCloud.points,v_normals_matrix = pointCloud.normals, v_color_matrix=cv2RGBA / 255.0) # color is N x 4 with alpha info
    ms = pymeshlab.MeshSet()
    ms.add_mesh(m, "pc_scan")
    ms.generate_surface_reconstruction_screened_poisson(depth=quality, scale=scale)
    # not familiar with the crop API, but I'm sure it's doable
    # now we generate UV map; there are a couple options here but this is a naive way
    ms.compute_texcoord_parametrization_triangle_trivial_per_wedge()
    # create texture using UV map and vertex colors
    ms.compute_texmap_from_color(textname=f"my_texture_name") # textname will be filename of a png, should not be a full path
    # texture file won't be saved until you save the mesh
    return ms

    
def SaveOutMeshLab( mesh, outputPath = './mesh.obj'):
    """
    Save out a mesh to a file.
    
    Args:
        mesh (open3d.geometry.TriangleMesh): The mesh to save out.
    """
    mesh.save_current_mesh(outputPath)
    
    
def SaveOutMesh( mesh, outputPath = './mesh.obj'):
    """
    Save out a mesh to a file.
    
    Args:
        mesh (open3d.geometry.TriangleMesh): The mesh to save out.
    """
    o3d.io.write_triangle_mesh(outputPath, mesh)
    
def GenerateMeshFromImage(glpnImageProcessor = "vinvino02/glpn-nyu", glpnForDepthEstimation = "vinvino02/glpn-nyu", image = "image.jpg",  quality = 10, threads = 1, tempPath = "./temp_mesh.obj"):
    """
    Generate a mesh from an image.
    
    Args:
        glpnImageProcessor (str): The GLPN Image processor to use, usually the same as the deph estimation model.
        glpnForDepthEstimation (str): The GLPN Depth Estimation model to use.
        image (str): The path to the image to use.
    
    Returns:
        open3d.geometry.TriangleMesh: The generated mesh.
    """
    output, image = GenerateDepthFromImage(glpnImageProcessor, glpnForDepthEstimation, image)
    pointCloud = GeneratePointsFromDepth(output, image)
    mesh = GenerateMeshFromPoints(pointCloud, quality = quality, threads = threads, tempPath = tempPath)
    return mesh
    
def GeneratePyMeshlabMeshFromImage(glpnImageProcessor = "vinvino02/glpn-nyu", glpnForDepthEstimation = "vinvino02/glpn-nyu", image = "image.jpg", quality = 10, scale = 1.1):
    """
    Generate a Pymeshlab mesh from an image.
    
    Args:
        glpnImageProcessor (str): The GLPN Image processor to use, usually the same as the deph estimation model.
        glpnForDepthEstimation (str): The GLPN Depth Estimation model to use.
        image (str): The path to the image to use.
    
    Returns:
        open3d.geometry.TriangleMesh: The generated mesh.
    """
    output, image = GenerateDepthFromImage(glpnImageProcessor, glpnForDepthEstimation, image)
    pointCloud = GeneratePointsFromDepth(output, image)
    cv2RGBA = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2RGBA)
    mesh = GenerateMeshFromPointsPymeshlab(pointCloud,cv2RGBA, quality = quality, scale = scale )
    return mesh
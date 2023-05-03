import numpy as np
import torch
from PIL import Image, ImageFilter
import PIL
import mediapipe as mp
import numpy as np
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation
from diffusers import ControlNetModel, UniPCMultistepScheduler, DPMSolverMultistepScheduler
from diffusers.utils import load_image
from diffusers import StableDiffusionControlNetPipeline, StableDiffusionPipeline
import cv2
import uaiDiffusers.hair as hair

def GenerateFace(inputFaceImage, inputFaceMask, sdRepo =  "runwayml/stable-diffusion-v1-5", cannyRepo = "lllyasviel/sd-controlnet-canny", loraPath = "justinjaro_lora/pytorch_lora_weights.bin", imagesToGenerate = 1, steps = 20, device="cuda", prompt_ = "A person", negPrompt_ = "bad face", low_threshold = 100, high_threshold = 200, seed=42):
    if isinstance(inputFaceImage, str):
        inputFaceImage = load_image(inputFaceImage)
    if isinstance(inputFaceMask, str):
        inputFaceMask = load_image(inputFaceMask)
    inputFaceMask = np.array(inputFaceMask)
    image = np.array(inputFaceImage)
    generator = torch.Generator(device=device).manual_seed(seed)
    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    # mask the image with inputFaceMask using bitwise_and
    composite = np.zeros_like(image)
    composite[inputFaceMask] = image[inputFaceMask]
    
    image = PIL.Image.fromarray( cv2.cvtColor(composite, cv2.COLOR_BGR2RGB))
    
    
    # image = Image.fromarray(image)
    
    controlnet = ControlNetModel.from_pretrained(
        cannyRepo, torch_dtype=torch.float16
    )
    
    pipe = StableDiffusionPipeline.from_pretrained(sdRepo, torch_dtype=torch.float16, safety_checker=None)
    pipe.load_attn_procs(loraPath)
    
    pipe = StableDiffusionControlNetPipeline(**pipe.components, controlnet=controlnet
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    # Remove if you do not have xformers installed
    # see https://huggingface.co/docs/diffusers/v0.13.0/en/optimization/xformers#installing-xformers
    # for installation instructions
    #pipe.enable_xformers_memory_efficient_attention()

    pipe.enable_model_cpu_offload()

    images = pipe(prompt=prompt_, negative_prompt=negPrompt_, image=image, num_inference_steps=steps, num_images_per_prompt = imagesToGenerate, generator = generator).images
    return images

def MaskOutGeneratedFace (generatedImage, mask_image):
    import numpy
    I = numpy.asarray(generatedImage)
    mask_I = numpy.asarray(mask_image)
    import cv2
    import numpy as np
    # Mask input image with binary mask
    result = cv2.bitwise_and(I, mask_I)
    # Color background white
    result[mask_image==0] = 255 # Optional
    return result

def InvertMask(mask_):
    matrix = 255 - np.asarray(mask_)
    newmask_ = Image.fromarray(matrix)
    return newmask_

def RunInpainting(templateImage, mask_):
    print("do painting")

def GenerateBackground(foregroundImage, mask_, sdRepo = "stabilityai/stable-diffusion-2-inpainting", prompt_ = "A park",negPrompt_="Missing body", imagesToGenerate = 1, seed = 42, device="cuda"):
    from diffusers import StableDiffusionInpaintPipeline
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        sdRepo,
        torch_dtype=torch.float16,
    )
    pipe.to("cuda")
    prompt = prompt_
    generator = torch.Generator(device=device).manual_seed(seed)
    #image and mask_image should be PIL images.
    #The mask structure is white for inpainting and black for keeping as is
    image = pipe(prompt=prompt,negative_prompt=negPrompt_, image=foregroundImage, mask_image=mask_, num_images_per_prompt=imagesToGenerate, generator=generator).images
    return image


def ExtractFace(pilImage):
    import numpy as np
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)
    results = face_mesh.process(cv2.cvtColor(np.array(pilImage), cv2.COLOR_BGR2RGB))
    landmarks = results.multi_face_landmarks[0]
    img = np.array(pilImage)
    face_oval = mp_face_mesh.FACEMESH_FACE_OVAL
    import pandas as pd
    df = pd.DataFrame(list(face_oval), columns = ["p1", "p2"])
    routes_idx = []
    
    p1 = df.iloc[0]["p1"]
    p2 = df.iloc[0]["p2"]
    
    for i in range(0, df.shape[0]):
        
        #print(p1, p2)
        
        obj = df[df["p1"] == p2]
        p1 = obj["p1"].values[0]
        p2 = obj["p2"].values[0]
        
        route_idx = []
        route_idx.append(p1)
        route_idx.append(p2)
        routes_idx.append(route_idx)
    
    # -------------------------------
    
    routes = []
    
    for source_idx, target_idx in routes_idx:
        
        source = landmarks.landmark[source_idx]
        target = landmarks.landmark[target_idx]
            
        relative_source = (int(img.shape[1] * source.x), int(img.shape[0] * source.y))
        relative_target = (int(img.shape[1] * target.x), int(img.shape[0] * target.y))
    
        #cv2.line(img, relative_source, relative_target, (255, 255, 255), thickness = 2)
        
        routes.append(relative_source)
        routes.append(relative_target)
    import numpy as np
    
    mask = np.zeros((img.shape[0], img.shape[1]))
    mask = cv2.fillConvexPoly(mask, np.array(routes), 1)
    mask = mask.astype(bool)
    
    out = np.zeros_like(img)
    out[mask] = img[mask]
    out = PIL.Image.fromarray(out)
    return PIL.Image.fromarray(mask) , out

def PILToCV2(pilImage, colorSpace = cv2.COLOR_RGB2BGR):
    import numpy as np
    return cv2.cvtColor(np.array(pilImage), colorSpace)

def CV2ToPIL(cv2Image, colorSpace = cv2.COLOR_BGR2RGB):
    return PIL.Image.fromarray(cv2.cvtColor(cv2Image, colorSpace))

def GetSelfieBodyMask(img, threshold = 0.5 , model_selection = 0):
    import numpy as np
    
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    segment = mp_selfie_segmentation.SelfieSegmentation(model_selection = model_selection)
    baseFaceImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    results = segment.process(baseFaceImage)
    image_seg_mask = results.segmentation_mask
    binary_mask = image_seg_mask > threshold
    image_seg_mask = PIL.Image.fromarray(binary_mask)
    
    import numpy as np
    
    # mask the img by the image_seg_mask
    composite = np.zeros_like(baseFaceImage)
    composite[binary_mask] = baseFaceImage[binary_mask]
    
    composite = PIL.Image.fromarray( cv2.cvtColor(composite, cv2.COLOR_BGR2RGB))
    
    return image_seg_mask, composite

def GetCannyEdges(img, mask = None, low_threshold = 50, high_threshold = 150):
    import cv2
    import numpy as np
    import cv2
    from PIL import Image
    import numpy as np

    image = np.array(img)
    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    if mask is not None:
        mask_ = np.array(mask)
        out = np.zeros_like(image)
        out[mask_] = image[mask_]

        canny_image = Image.fromarray(out)
        return canny_image
    else:
        canny_image = Image.fromarray(image)
        return canny_image
    

def GetSelfieBodyCannyEdges(img, threshold = 0.5 , model_selection = 0,  low_threshold = 50, high_threshold = 150):
    selfieMask, person = GetSelfieBodyMask(img, threshold = threshold , model_selection = model_selection)
    canny_image = GetCannyEdges(img, selfieMask, low_threshold, high_threshold)
    return canny_image

def GetSelfieFaceCannyEdges(img,   low_threshold = 50, high_threshold = 150):
    facemask, faceMasked = ExtractFace(img)
    canny_image = GetCannyEdges(img, facemask, low_threshold, high_threshold)
    return canny_image

def GetHairMask(img):
    import cv2
    image = PILToCV2(img, cv2.COLOR_RGB2BGR)

    hair_segmentation = hair.HairSegmentation(image.shape[1], image.shape[0])

    hair_mask = hair_segmentation(image)
    hair_image, hair_mask = hair_segmentation.draw_hair_mask(image, hair_mask)
    # out = np.zeros_like(img)
    # mask = hair_mask.astype(bool)
    # out[mask] = img[mask]
    # out = PIL.Image.fromarray(out)
    # out = CV2ToPIL(hair_image, cv2.COLOR_BGRA2RGBA)
    hair_mask = CV2ToPIL(hair_mask)
    return hair_image, hair_mask


def RunSDLora(sdRepo = "wavymulder/portraitplus", loraPath = "justinjaro_lora/pytorch_lora_weights.bin", imagesToGenerate = 3, prompt = "A photo of ", negative_prompt=" blurry"):
    from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
    pipe = DiffusionPipeline.from_pretrained(sdRepo, torch_dtype=torch.float16)
    # pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")

    pipe.load_attn_procs(loraPath, )

    images = pipe(prompt=prompt,negative_prompt=negative_prompt, num_inference_steps=55, guidance_scale=7, num_images_per_prompt=imagesToGenerate).images
    return images

def imageGrid(imgs, rows, maxWidth = 1080):
    from  PIL import Image
    cols =  int(len(imgs)/rows)
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    
    grid = grid.resize((maxWidth, int(grid_h/grid_w*maxWidth)))
    return grid

def AugmentFaceSDControlnetLoraFace(pilImg, sdRepo="wavymulder/portraitplus",cannyRepo="lllyasviel/sd-controlnet-canny", loraPath="justinjaro_lora\pytorch_lora_weights.bin", prompt_="portrait+ style A photo of ", negPrompt_=" blurry, high contrast, hdr", imagesToGenerate=4, maskFeather = 2, seed =42):
    baseFaceImage = np.array(pilImg)
    selfieMask, person = GetSelfieBodyMask(baseFaceImage)
    facemask, faceMasked = ExtractFace(baseFaceImage)
    faceImages = GenerateFace(inputFaceImage=baseFaceImage, inputFaceMask=selfieMask, sdRepo=sdRepo,cannyRepo=cannyRepo, loraPath=loraPath, prompt_=prompt_, negPrompt_=negPrompt_, imagesToGenerate=imagesToGenerate, seed=seed)

    mask_blur = faceMasked.filter(ImageFilter.GaussianBlur(maskFeather))
    appliedFace = Image.composite(faceImages[0], pilImg, mask_blur)
    return appliedFace, selfieMask,  facemask

def CropFace(cv2image, padding = 200, size = (256, 256)):
    # convert to RGB
    (height, width) = cv2image.shape[:2]
    newImage = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGB)
    newImage = cv2.resize(newImage, (height + padding, width + padding))
    cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGB)
    # initialize the face detection model
    faceDetection = mp.solutions.face_detection.FaceDetection()
    # detect the face
    results = faceDetection.process(cv2image)
    try:
        # get the bounding box of the face
        boundingBox = results.detections[0].location_data.relative_bounding_box
        # get the width and height of the image
        # get the coordinates of the bounding box
        dimensionPad = int(padding / 2)
        (startX, startY, endX, endY) = (boundingBox.xmin, boundingBox.ymin, boundingBox.xmin + boundingBox.width, boundingBox.ymin + boundingBox.height)
        # convert the coordinates to pixels
        (startX, startY, endX, endY) = (int(startX * width), int(startY * height), int(endX * width), int(endY * height))
        (startX, startY, endX, endY) = (startX - dimensionPad, startY - dimensionPad, endX + dimensionPad, endY + dimensionPad)
        if startX < 0:
            startX = 0
        if startY < 0:
            startY = 0
        if endX > width:
            endX = width
        if endY > height:
            endY = height
        faceROI = cv2image[startY:endY, startX:endX]
        faceROI = cv2.resize(faceROI, size)
        # convert the face ROI to RGB
        faceROI = cv2.cvtColor(faceROI, cv2.COLOR_BGR2RGB)
        # return the face ROI
        return faceROI
    except:
        return None
    
def CreateFaceDataset(inputSearchPattern = "johnSmith\johnSmith*.jpg", outputImageDir = "johnSmith\output", prefix = "johnSmith", padding = 200, size = (512, 512)):
    import glob
    import os
    inputImageDir = glob.glob(inputSearchPattern)
    if not os.path.exists(outputImageDir):
        os.makedirs(outputImageDir)
    for indx, img in enumerate(inputImageDir):
        print(f"Processing image {indx+1} / {len(inputImageDir)}")
        n= cv2.imread(img)
        faceROI = CropFace(n, padding, size)
        if isinstance( faceROI, np.ndarray):
            newimage = CV2ToPIL(faceROI)
            newimage.save(f"{outputImageDir}\{prefix}_{str(indx).zfill(4)}.png")

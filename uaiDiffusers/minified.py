import time
import json

# startImport()
import numpy as np
import torch
from PIL import Image, ImageFilter
import PIL
# from transformers import AutoImageProcessor, UperNetForSemanticSegmentation
import uuid

import cv2
import zipfile
import io
import flask
from flask import jsonify
import io
import requests

import base64
import glob
import os
import mimetypes
from uaiDiffusers.media.mediaRequestBase import MediaRequestBase64, MultipleMediaRequest
responseMessage = ""
CLRMaxSteps = 20
def SendCLRProgress(progress = 0.01, message = "Loading...", done = False):
    from System import PythonCLR
    PythonCLR.SendProgress(progress, message, done)

def CLRProgressCallback(step, timeScale, tensor):
        global CLRMaxSteps
        from System import PythonCLR
        percent = step / CLRMaxSteps
        PythonCLR.SendProgress(percent, f"{str(int(percent * 100))}% conjuring image.", False)
        
def SendCLRJSON(message = {}):
    from System import PythonCLR
    PythonCLR.SendJSON(message)

def SetCLRMaxSteps(steps):
        global CLRMaxSteps
        CLRMaxSteps = steps
        


def OptimizePipe(pipe, enableCPUOffload = False,enable_attention_slicing=False,enable_vae_tiling=False ):
    if enableCPUOffload:
        pipe.enable_model_cpu_offload()
    if enable_attention_slicing:
        pipe.enable_attention_slicing()
    if enable_vae_tiling:
        pipe.enable_vae_tiling()
    return pipe


def LoadCustomSDPipeFiles(pipe, loraPath = "", textualInversion = "",customSDBin = "" ):
    if loraPath != "":
        pipe.load_attn_procs(loraPath)
    import os
    
    if customSDBin != "":   
        pipe.unet.load_attn_procs(customSDBin)
    if textualInversion != "":
        pipe.load_textual_inversion(textualInversion)
    return pipe

def SetResponseMessage(message):
    global responseMessage
    responseMessage = message
    
def GenerateFace(inputFaceImage, inputFaceMask, sdRepo =  "runwayml/stable-diffusion-v1-5", cannyRepo = "lllyasviel/sd-controlnet-canny", loraPath = "", textualInversion = "",customSDBin = "", imagesToGenerate = 1, steps = 20, device="cuda", prompt_ = "A person", negPrompt_ = "bad face", low_threshold = 100, high_threshold = 200, seed=42, pipe_ = None, enableCPUOffload = True,enable_attention_slicing=False,enable_vae_tiling=False):
    """
    Generates a face from a face image and a face mask using Stable Diffusion.
    
    Args:
        inputFaceImage (str): Path to the face image.
        inputFaceMask (str): Path to the face mask.
        sdRepo (str): Path to the Stable Diffusion model.
        cannyRepo (str): Path to the Canny model.
        loraPath (str): Path to the LoRA model.
        textualInversion (str): Path to the textual inversion model.
        customSDBin (str): Path to the custom Stable Diffusion model.
        imagesToGenerate (int): Number of images to generate.
        steps (int): Number of steps.
        device (str): Device to use.
        prompt_ (str): Text Prompt to use.
        negPrompt_ (str): Text Negative prompt to remove certain things from an image.
        low_threshold (int): Low threshold for Canny.
        high_threshold (int): High threshold for Canny.
        seed (int): Seed to use.
        pipe_ (StableDiffusionPipeline): Stable Diffusion pipeline to use.
        
    
    """
    
    from diffusers import ControlNetModel, DPMSolverMultistepScheduler, StableDiffusionControlNetPipeline, StableDiffusionPipeline
    from diffusers.utils import load_image
    
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
    if pipe_ is None:
        pipe = StableDiffusionPipeline.from_pretrained(sdRepo, torch_dtype=torch.float16, safety_checker=None)
    else:
        pass
    
    pipe = LoadCustomSDPipeFiles(pipe, loraPath, textualInversion, customSDBin)

    
    pipe = StableDiffusionControlNetPipeline(**pipe.components, controlnet=controlnet
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    torch.manual_seed(seed)
    
    pipe = OptimizePipe(pipe,enableCPUOffload=enableCPUOffload, enable_attention_slicing=enable_attention_slicing, enable_vae_tiling=enable_vae_tiling)
    
    # Remove if you do not have xformers installed
    # see https://huggingface.co/docs/diffusers/v0.13.0/en/optimization/xformers#installing-xformers
    # for installation instructions
    #pipe.enable_xformers_memory_efficient_attention()


    images = pipe(prompt=prompt_, negative_prompt=negPrompt_, image=image, num_inference_steps=steps, num_images_per_prompt = imagesToGenerate, generator = generator).images
    return images

def GenerateImage(sdRepo =  "runwayml/stable-diffusion-v1-5", loraPath = "", textualInversion = "",customSDBin = "", imagesToGenerate = 1, steps = 20, device="cuda", prompt_ = "A person", negPrompt_ = "bad face", seed=42, width=512, height=512, textGuidance=7.5, enableCPUOffload = True,enable_attention_slicing=False,enable_vae_tiling=False,callback = None, pipe_ = None):
    """
    Generate an image from a text prompt using Stable Diffusion.
    
    Args:
        sdRepo (str): The Stable Diffusion model to use.
        loraPath (str): The path to the LoRA model to use.
        textualInversion (str): The path to the textual inversion model to use. 
        customSDBin (str): The path to the custom Stable Diffusion model to use.
        imagesToGenerate (int): The number of images to generate.
        steps (int): The number of steps to use.
        device (str): The device to use.
        prompt_ (str): The prompt to use.
        negPrompt_ (str): The negative prompt to use.
        seed (int): The seed to use.
        width (int): The width of the image to generate.
        height (int): The height of the image to generate.
        textGuidance (float): The scale to guide the image based on the text. The higher the closer it follows, but will look deep-fried.
        pipe_ (StableDiffusionPipeline): The Stable Diffusion pipeline to use. Set this to keep from reinitializing models.
        
    Returns:
        images (list): A list of images generated.
        
    """
    
    # image = Image.fromarray(image)
    from diffusers import StableDiffusionPipeline
  
    if pipe_ is None:
        pipe = StableDiffusionPipeline.from_pretrained(sdRepo, torch_dtype=torch.float16, safety_checker=None)
    else:
        pipe = pipe_
        
    pipe = LoadCustomSDPipeFiles(pipe, loraPath, textualInversion, customSDBin)
    torch.manual_seed(seed)
    pipe = OptimizePipe(pipe,enableCPUOffload=enableCPUOffload, enable_attention_slicing=enable_attention_slicing, enable_vae_tiling=enable_vae_tiling)
    # Remove if you do not have xformers installed
    # see https://huggingface.co/docs/diffusers/v0.13.0/en/optimization/xformers#installing-xformers
    # for installation instructions
    #pipe.enable_xformers_memory_efficient_attention()
    


    images = pipe(prompt=prompt_, negative_prompt=negPrompt_, num_inference_steps=steps, num_images_per_prompt = imagesToGenerate, width = width, height= height, guidance_scale=textGuidance, callback=callback).images
    return images


def GenerateImageXL(sdRepo =  "stabilityai/stable-diffusion-xl-base-1.0", loraPath = "minimaxir/sdxl-wrong-lora", textualInversion = "",customSDBin = "", refinerRepo = "", vaeRepo = "madebyollin/sdxl-vae-fp16-fix", imagesToGenerate = 1, steps = 20, device="cuda", prompt_ = "A person", negPrompt_ = "bad face", seed=42, width=1024, height=1024, textGuidance=7.5,  high_noise_frac=0.8, enableCPUOffload = True,enable_attention_slicing=False,enable_vae_tiling=False,callback = None):
    """
    Generate an XLimage from a text prompt using Stable Diffusion.
    
    Args:
        sdRepo (str): The Stable Diffusion model to use.
        loraPath (str): The path to the LoRA model to use.
        textualInversion (str): The path to the textual inversion model to use. 
        customSDBin (str): The path to the custom Stable Diffusion model to use.
        refinerRepo (str): The path to the custom Stable Diffusion XL Refiner model to use.
        vaeRepo (str): The path to the custom Stable Diffusion VAE model to use.
        imagesToGenerate (int): The number of images to generate.
        steps (int): The number of steps to use.
        device (str): The device to use.
        prompt_ (str): The prompt to use.
        negPrompt_ (str): The negative prompt to use.
        seed (int): The seed to use.
        width (int): The width of the image to generate.
        height (int): The height of the image to generate.
        high_noise_frac (float): When to start denoising. default 0.8 .
        textGuidance (float): The scale to guide the image based on the text. The higher the closer it follows, but will look deep-fried.
        pipe_ (StableDiffusionPipeline): The Stable Diffusion pipeline to use. Set this to keep from reinitializing models.
        
    Returns:
        images (list): A list of images generated.
        
    """
    
    # image = Image.fromarray(image)
    from diffusers import AutoencoderKL, StableDiffusionXLPipeline, DiffusionPipeline
  
    vae = AutoencoderKL.from_pretrained(
        vaeRepo,
        torch_dtype=torch.float16
    )
    pipe = StableDiffusionXLPipeline.from_pretrained(
        sdRepo, torch_dtype=torch.float16, variant="fp16", use_safetensors=True,vae=vae
    )
    
    torch.manual_seed(seed)
    pipe = OptimizePipe(pipe,enableCPUOffload=enableCPUOffload, enable_attention_slicing=enable_attention_slicing, enable_vae_tiling=enable_vae_tiling)
    
   
    pipe.to(device)
    pipe.load_lora_weights(loraPath)
    if refinerRepo != "":
        refiner = DiffusionPipeline.from_pretrained(
            refinerRepo,
            text_encoder_2=pipe.text_encoder_2,
            vae=pipe.vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )
        refiner.to(device)

    n_steps = steps
    # n_steps = 40
    torch.manual_seed(seed)
    prompt = prompt_
    # prompt = "A 3d render death metal album cover with Marilyn Monroe wearing Raybans while shooting a machine gun in a jungle. octane render. high quality. cinematic. aesthetic. award winning. hyper metal. 8k"
    negativePrompt = "wrong . low quality. " + negPrompt_

    image = pipe(prompt=prompt, negative_prompt=negativePrompt, num_inference_steps=n_steps,  guidance_scale=textGuidance, num_images_per_prompt= imagesToGenerate,callback=callback).images
    outimages = []
    for index, img in enumerate(image):
        if refinerRepo != "":
            image_ = refiner(
                prompt=prompt,
                num_inference_steps=n_steps,
                denoising_start=high_noise_frac,
                image=img,
                callback=callback
            ).images[0]
            outimages.append(image_)
        else:
            outimages.append(img)
    return outimages
   
    # Remove if you do not have xformers installed
    # see https://huggingface.co/docs/diffusers/v0.13.0/en/optimization/xformers#installing-xformers
    # for installation instructions
    #pipe.enable_xformers_memory_efficient_attention()
    


    images = pipe(prompt=prompt_, negative_prompt=negPrompt_, num_inference_steps=steps, num_images_per_prompt = imagesToGenerate, width = width, height= height, guidance_scale=textGuidance).images
    return images

def MaskOutGeneratedFace (generatedImage, mask_image):
    """
    Mask out a generated face using a 0 - 1 mask
    
    Args:
        generatedImage (PIL.Image): The generated image to mask.
        mask_image (PIL.Image): The mask to use.
    Returns:
        result (PIL.Image): The masked image.
    """
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
    """
    Invert a mask from 0 - 1 to 1 - 0
    
    Args:
        mask_ (PIL.Image): The mask to invert.
    Returns:
        newmask_ (PIL.Image): The inverted mask.
    """
    matrix = 255 - np.asarray(mask_)
    newmask_ = Image.fromarray(matrix)
    return newmask_

def RunInpainting(templateImage, mask_):
    print("do painting")

def GenerateBackground(foregroundImage, mask_, sdRepo = "stabilityai/stable-diffusion-2-inpainting", prompt_ = "A park",negPrompt_="Missing body", imagesToGenerate = 1, seed = 42, device="cuda"):
    """
    Generate a background for an image using Stable Diffusion.
    
    Args:
        foregroundImage (PIL.Image): The foreground image to use.
        mask_ (PIL.Image): The mask to use.
        sdRepo (str): The Stable Diffusion model to use.
        prompt_ (str): The prompt to use.
        negPrompt_ (str): The negative prompt to use.
        imagesToGenerate (int): The number of images to generate.
        seed (int): The seed to use.
        device (str): The device to use.
        
    Returns:
        image (PIL.Image): The generated image.
        
    """
    
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
    """
    Extract a face from an image
    
    Args:
        pilImage (PIL.Image): The image to extract a face from.
        
    Returns:
        face (PIL.Image): The extracted face.
        
    """
    import mediapipe as mp
    
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
    """
    Convert a PIL image to a CV2 image.
    
    Args:
        pilImage (PIL.Image): The PIL image to convert.
        colorSpace (int): The  cv2 color space to use.
    
    Returns:
        cv2Image (cv2.Image): The converted image.
    """
    import numpy as np
    return cv2.cvtColor(np.array(pilImage), colorSpace)

def CV2ToPIL(cv2Image, colorSpace = cv2.COLOR_BGR2RGB):
    """
    Convert a CV2 image to a PIL image.
    
    Args:
        cv2Image (cv2.Image): The cv2 image to convert.
        colorSpace (int): The  cv2 color space to use.
    
    Returns:
        image (PIL.Image): The converted PIL image.
    """
    return PIL.Image.fromarray(cv2.cvtColor(cv2Image, colorSpace))

def GetSelfieBodyMask(img, threshold = 0.5 , model_selection = 0):
    """
    Get a mask of a person in a selfie.
    
    Args:
        img (PIL.Image): The image to get the mask from.
        threshold (float): The threshold to use.
        model_selection (int): The model to use.
        
    Returns:
        image_seg_mask (PIL.Image): The mask of the person.
        composite (PIL.Image): The masked image.
        image_seg_maskRaw (np.array): The raw mask.
        
    
    """
    import numpy as np
    import mediapipe as mp
    
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    segment = mp_selfie_segmentation.SelfieSegmentation(model_selection = model_selection)
    baseFaceImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    results = segment.process(baseFaceImage)
    image_seg_maskRaw = results.segmentation_mask
    binary_mask = image_seg_maskRaw > threshold
    image_seg_mask = PIL.Image.fromarray(binary_mask)
    
    import numpy as np
    
    # mask the img by the image_seg_mask
    composite = np.zeros_like(baseFaceImage)
    composite[binary_mask] = baseFaceImage[binary_mask]
    
    composite = PIL.Image.fromarray( cv2.cvtColor(composite, cv2.COLOR_BGR2RGB))
    
    return image_seg_mask, composite, image_seg_maskRaw

def GetGroupSelfieBodyMask(img, threshold = 0.5 , depthThreshold = 0.8 , depthMax = 1 ,maskFeather=1, model_selection = 0, maxSize = 0):
    
    """
    Get a mask of a group of people in a selfie.
    
    Args:
        img (PIL.Image): The image to get the mask from.
        threshold (float): The threshold to use.
        depthThreshold (float): The depth threshold to use.
        depthMax (float): The depth max to use.
        maskFeather (int): The mask feather to use.
        model_selection (int): The model to use. 0 - 1. 1 is more accurate.
        maxSize (int): The max size to use.
        
        
    Returns:
        outImage (PIL.Image): The masked image of the people.
        baseimg (PIL.Image): The input image.
        imageMask (np.array): The raw mask.
    """
    
    baseimg = img
    if maxSize != 0:
        baseimg = ResizeImage(img, maxSize)

    imageMask, comp, imageMaskRaw = GetSelfieBodyMask(baseimg, threshold,model_selection)
    clamped, clampDepthImage, normalizedZeroOne = ClampDepth(imageMaskRaw, depthThreshold, depthMax)
    mask_blur = clamped.filter(ImageFilter.GaussianBlur(maskFeather))
    b1 = PILToCV2(baseimg.convert("RGBA"),cv2.COLOR_RGBA2BGRA)
    b2 = PILToCV2(mask_blur.convert("RGBA"),cv2.COLOR_RGBA2BGRA)
    b22 = b2[:,:,3]
    outImage = AddAlphaToCV2Image(b1,b22,cv2.COLOR_RGBA2BGRA)
    
    return outImage, baseimg,imageMask

def GetCannyEdges(img, mask = None, low_threshold = 50, high_threshold = 150):
    """
    Get the canny edges of an image.
    
    Args:
        img (PIL.Image): The image to get the canny edges from.
        mask (PIL.Image): The mask to use.
        low_threshold (int): The low threshold to use.
        high_threshold (int): The high threshold to use.
    
    Returns:
        canny_image (PIL.Image): The canny edges of the image.
    """
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

def AugmentFaceCustomSDControlnetFace(pilImg, sdRepo="wavymulder/portraitplus",cannyRepo="lllyasviel/sd-controlnet-canny", textualInversion="justinjaro_lora/johnsmith.bin", customSDBin="justinjaro_lora/pytorch_lora_weights.bin", prompt_="portrait+ style A photo of ", negPrompt_=" blurry, high contrast, hdr", imagesToGenerate=4, maskFeather = 2, seed =42, steps=25):
    baseFaceImage = np.array(pilImg)
    selfieMask, person = GetSelfieBodyMask(baseFaceImage)
    facemask, faceMasked = ExtractFace(baseFaceImage)
    faceImages = GenerateFace(inputFaceImage=baseFaceImage, inputFaceMask=selfieMask, sdRepo=sdRepo,cannyRepo=cannyRepo, customSDBin=customSDBin,textualInversion=textualInversion, prompt_=prompt_, negPrompt_=negPrompt_, imagesToGenerate=imagesToGenerate, seed=seed, steps=steps)
    outimages =  []
    for img in faceImages:
        mask_blur = faceMasked.filter(ImageFilter.GaussianBlur(maskFeather))
        # make sure PilImage is in RGBA mode
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        if pilImg.mode != "RGBA":
            pilImg = img.convert("RGBA")
        if mask_blur.mode != "RGBA":
            mask_blur = img.convert("RGBA")
        appliedFace = Image.composite(img, pilImg, mask_blur)
        outimages.append(appliedFace)
    return outimages, selfieMask,  facemask

def AugmentFaceSDControlnetFace(pilImg, sdRepo="wavymulder/portraitplus",cannyRepo="lllyasviel/sd-controlnet-canny", loraPath="justinjaro_lora/pytorch_lora_weights.bin", prompt_="portrait+ style A photo of ", negPrompt_=" blurry, high contrast, hdr", imagesToGenerate=4, maskFeather = 2, seed =42):
    baseFaceImage = np.array(pilImg)
    selfieMask, person = GetSelfieBodyMask(baseFaceImage)
    facemask, faceMasked = ExtractFace(baseFaceImage)
    faceImages = GenerateFace(inputFaceImage=baseFaceImage, inputFaceMask=selfieMask, sdRepo=sdRepo,cannyRepo=cannyRepo, loraPath=loraPath, prompt_=prompt_, negPrompt_=negPrompt_, imagesToGenerate=imagesToGenerate, seed=seed)
    
    outimages =  []
    for img in faceImages:
        mask_blur = faceMasked.filter(ImageFilter.GaussianBlur(maskFeather))
        appliedFace = Image.composite(img, pilImg, mask_blur)
        outimages.append(appliedFace)
    return outimages, selfieMask,  facemask

def CropFace(cv2image, padding = 200, size = (256, 256)):
    # convert to RGB
    import mediapipe as mp
    
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
    
def CreateFaceDataset(inputSearchPattern = "johnSmith/johnSmith*.jpg", outputImageDir = "johnSmith/output", prefix = "johnSmith", padding = 200, size = (512, 512)):

    inputImageDir = glob.glob(inputSearchPattern)
    if not os.path.exists(outputImageDir):
        os.makedirs(outputImageDir)
    for indx, img in enumerate(inputImageDir):
        print(f"Processing image {indx+1} / {len(inputImageDir)}")
        n= cv2.imread(img)
        faceROI = CropFace(n, padding, size)
        if isinstance( faceROI, np.ndarray):
            newimage = CV2ToPIL(faceROI)
            newimage.save(f"{outputImageDir}/{prefix}_{str(indx).zfill(4)}.png")


    
def GetFace(filePath = "johnSmith/johnSmith*.jpg", padding = 200, size = (512, 512)):
    n= cv2.imread(filePath)
    faceROI = CropFace(n, padding, size)
    if isinstance( faceROI, np.ndarray):
        newimage = CV2ToPIL(faceROI)
        return newimage
    else:
        return None



def ImagesToZip(pilImages):
    '''
    Takes a list of PIL images and returns a zip binary in memory ready to be saved out or sent
    
    Args:
        pilImages (list): list of PIL images
    
    Returns:
        memory_file (BytesIO): zip binary in memory
    '''
    imgs = []
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w', compression=zipfile.ZIP_DEFLATED) as zipObject:
        for indx, img in enumerate(pilImages):
            img_io = io.BytesIO()
            img.save(img_io, 'PNG', quality=100)
            img_io.seek(0)
            imgs.append(img_io)
            zipObject.writestr("image_"+ str(indx).zfill(4) + ".png" , img_io.read())
    memory_file.seek(0)
    return memory_file

def ImagesToZipFlaskResponse(pilImages):
    '''
    Takes a list of PIL images and creates a flask response with a zip binary in memory ready to be saved out or sent
    Args:
        pilImages (list): list of PIL images
    
    Returns:
        flask.Response: flask response with a zip binary in memory
    
    '''
    return  flask.send_file(ImagesToZip(pilImages), mimetype='application/zip')

def ImageToBytes(pilImage):
    '''
    Takes a PIL image and convert it to a BytesIO object
    
    Args:
        pilImage (PIL.Image): PIL image
    
    Returns:
        img_io (BytesIO): BytesIO object
    '''
    img_io = io.BytesIO()
    pilImage.save(img_io, 'PNG', quality=100)
    img_io.seek(0)
    # return the pilImage as an image
    return img_io
def ImagesToBase64(pilImage):
    '''
    
    Takes a PIL image and convert it to a Base64 string
    
    Args:
        pilImage (PIL.Image): PIL image
    
    Returns:
        base64String (str): Base64 string
    '''
    # return the pilImage as an image
    return  base64.b64encode(ImageToBytes(pilImage).getvalue()).decode()

def ImagesToFlaskResponse(pilImage):
    '''
    Takes a list of PIL images and returns a zip binary ready to be saved out or sent
    '''
    # return the pilImage as an image
    return ImageToBytes(pilImage)


def MultiImagesToFlaskResponse(pilImages, prompts):
    '''
    Takes a list of PIL images and prompts of the same count and returns a MultipleMediaRequest JSON object containing the images and prompts as MediaRequestBase64 objects
    
    Args:
        pilImages (list): A list of PIL images
        prompts (list): A list of prompts
        
    Returns:
        JSONString (str): A MultipleMediaRequest JSON string
    '''
    # return the pilImage as an image
    
    MultipleMediaRequest_ = MultipleMediaRequest([])
    for indx, img in enumerate(pilImages):
        prompt = ""
        try:
            prompt = prompts[indx]
        except:
            prompt = ""
        MultipleMediaRequest_.addMedia(MediaRequestBase64(ImagesToBase64(img), prompt))
        
    return jsonify(MultipleMediaRequest_.toDict())

def MultiImagesToJSONResponse(pilImages, prompts):
    '''
    Takes a list of PIL images and prompts of the same count and returns a MultipleMediaRequest JSON object containing the images and prompts as MediaRequestBase64 objects
    
    Args:
        pilImages (list): A list of PIL images
        prompts (list): A list of prompts
        
    Returns:
        JSONString (str): A MultipleMediaRequest JSON string
    '''
    # return the pilImage as an image
    
    MultipleMediaRequest_ = MultipleMediaRequest([])
    for indx, img in enumerate(pilImages):
        prompt = ""
        try:
            prompt = prompts[indx]
        except:
            prompt = ""
        MultipleMediaRequest_.addMedia(MediaRequestBase64(ImagesToBase64(img), prompt))
        
    return json.dumps(MultipleMediaRequest_.toDict())

def Base64StringToPILImage(base64String):
    '''
    Takes a base64 string and returns a PIL image
    '''
    return PIL.Image.open(io.BytesIO(base64.b64decode(base64String)))

def uaiPromptImageJSONObjectToPILImage(promptImage):
    '''
    Takes a prompt image from the UAI API and returns a PIL image
    
    Args:
        promptImage (dict): A prompt image from the UAI API
    
    Returns:
        PIL.Image: A PIL image
    '''
    return Base64StringToPILImage(promptImage["media"])


def uaiPromptMultiImageJSONObjectToPILImage(promptImages):
    '''
    Takes a prompt image from the UAI API and returns a PIL image
    '''
    imgs = []
    for indx, img in enumerate(promptImages['media']):
        imgs.append(Base64StringToPILImage(img["media"]))
    return imgs

def saveUAIPromptMultiPILImageObjectToFiles(promptImages, filepath):
    for indx , img in enumerate(promptImages):
        img.save(filepath + str(indx).zfill(4) + ".png")

def saveUAIPromptMultiImageJSONObjectToFiles(promptImages, filepath):
    for indx , img in enumerate(promptImages['media']):
        saveUAIPromptImageJSONObjectToFiles(img, filepath  + str(indx).zfill(4) + ".png")

def saveUAIPromptImageJSONObjectToFiles(promptImage, filepath):
    '''
    Save the image to file
    
    Args:
        promptImage (dict): A prompt image from the UAI API
        filepath (str): The path to save the image to
    
    Returns:
        PIL.Image: A PIL image
    '''
    piImage = Base64StringToPILImage(promptImage["media"])
    piImage.save(filepath)
    return piImage

def GenerateDepthImage(pilImage, modelName = "Intel/dpt-large"):
    """
    Generate a depth image from a mono image
    
    Args:
        pilImage (PIL.Image): A PIL image
        modelName (str): The model name to use from Huggingface for the DPT model Depth Estimation.
    
    Returns:
        depth (PIL.Image): The depth image
        output (np.array): The depth image as a numpy array
    """
    from transformers import pipeline, DPTImageProcessor, DPTForDepthEstimation
    processor = DPTImageProcessor.from_pretrained(modelName)
    model = DPTForDepthEstimation.from_pretrained(modelName)

    # prepare image for the model
    inputs = processor(images=pilImage, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # interpolate to original size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=pilImage.size[::-1],
        mode="bicubic",
        align_corners=False,
    )

    # visualize the prediction
    output = prediction.squeeze().cpu().numpy()
    normalized = NormalizeNumpyArray(output)
    conv = ConvertNumpyArrayToImage(normalized)
    depth = CV2ToPIL(conv)
    return depth, output

def NormalizeNumpyArray(numpyArray):
    """
    Normalize a numpy array to 0-1
    
    Args:
        numpyArray (np.array): A numpy array
    
    Returns:
        np.array: The normalized numpy array
    """
    normalizedNumpy = (numpyArray - np.min(numpyArray)) / (np.max(numpyArray) - np.min(numpyArray))
    normalizedZeroOne = (normalizedNumpy-np.min(normalizedNumpy))/(np.max(normalizedNumpy)-np.min(normalizedNumpy))
    return normalizedZeroOne

def ConvertNumpyArrayToImage(numpyArray, processType = "uint8"):
    """
    Convert a numpy array to an image
    
    Args:
        numpyArray (np.array): A numpy array
        processType (str, optional): The type to convert to. Defaults to "uint8".
        
    Returns:
        np.array: The converted numpy array
    """
    return (numpyArray * 255 / np.max(numpyArray)).astype(processType)

def ClampDepth(rawDepth,thresholdAdd = 0.2 , maxDepth = 1):
    """
    Clamp a depth image
    
    Args:
        rawDepth (np.array): A depth image
        thresholdAdd (float, optional): The threshold to add.
        maxDepth (int, optional): The max depth.
        
    Returns:
        pilImage (PIL.Image): The clamped depth image
        normalizedNumpy (np.array): The normalized numpy array
        alpha (np.array): The alpha numpy array
        
        
    
    """
    # Normalize and Threshold the depth mask
    normalizedNumpy = (rawDepth - np.min(rawDepth)) / (np.max(rawDepth) - np.min(rawDepth))
    normalizedZeroOne = (normalizedNumpy-np.min(normalizedNumpy))/(np.max(normalizedNumpy)-np.min(normalizedNumpy))
    
    threshold = np.min(normalizedNumpy[normalizedZeroOne > 0]) + thresholdAdd
    cleaned = np.where(normalizedNumpy > threshold, normalizedNumpy, 0)
    cleaned = np.where(cleaned > maxDepth, maxDepth, cleaned)
    clampedCleaned = np.interp(cleaned, (cleaned.min(), cleaned.max()), (0, maxDepth))

    # Convert to 8-bit 255
    formatted = (clampedCleaned * 255 / np.max(clampedCleaned)).astype("uint8")
    alpha = (clampedCleaned * 255 / np.max(clampedCleaned)).astype("uint8")

    # Apply Alpha
    pilImage = CV2ToPIL(formatted, cv2.COLOR_GRAY2RGBA)
    alphaPil = PILToCV2(pilImage, cv2.COLOR_RGBA2BGRA)
    alphaPil[:, :, 3] = alpha
    pilImage = CV2ToPIL(alphaPil, cv2.COLOR_BGRA2RGBA)
    return pilImage, normalizedZeroOne, alpha



def AddAlphaToCV2Image(baseimgCV2, maskCV2, baseImageColorpsace = cv2.COLOR_RGBA2BGRA):
    alphaPil = baseimgCV2
    alphaPil[:, :, 3] = maskCV2
    pilImage = CV2ToPIL(alphaPil, cv2.COLOR_BGRA2RGBA)
    return pilImage
    
def ClampImageByDepth(imagePath, threshold = 0.2, maxDepth = 1 , maskFeather=0.2):
    """
    Clamp an image by generate depth
    
    Args:
        imagePath (str | PIL.Image): The path OR PIL Image to the image
        threshold (float, optional): The threshold to add
        maxDepth (int, optional): The max depth
        maskFeather (float, optional): The mask feather
        
    Returns:
        pilImage (PIL.Image): The clamped image
        clampDepthImage (PIL.Image): The clamped depth image
        normalizedZeroOne (np.array): The normalized numpy array
    """
    baseimg = imagePath
    if isinstance(imagePath, str):
        baseimg = Image.open(imagePath)
    img, raw = GenerateDepthImage(baseimg)
    clampDepthImage, normalizedZeroOne, alpha = ClampDepth(raw, threshold, maxDepth)
    mask_blur = clampDepthImage.filter(ImageFilter.GaussianBlur(maskFeather))
    maskAlpha = PILToCV2(mask_blur, cv2.COLOR_RGBA2GRAY)
    # Apply Alpha
    alphaPil = PILToCV2(baseimg, cv2.COLOR_RGBA2BGRA)
    alphaPil[:, :, 3] = maskAlpha
    pilImage = CV2ToPIL(alphaPil, cv2.COLOR_BGRA2RGBA)
    return pilImage, clampDepthImage, normalizedZeroOne

def ResizeImage(pilImage, maxWidth = 700):
    """
    Resize an image based on max width
    
    Args:
        pilImage (PIL.Image): A PIL image
        maxWidth (int, optional): The max width
    
    Returs:
        pilImage (PIL.Image): The resized PIL image
    
    """
    width, height = pilImage.size
    if width > maxWidth:
        pilImage = pilImage.resize((maxWidth, int(height * maxWidth / width)))
    else:
        newHeight = int(height * maxWidth / width)
        pilImage = pilImage.resize((maxWidth, newHeight))
    return pilImage



def upload_file_to_space(spaces_client, space_name, file_src, save_as, **kwargs):
    """
    Upload a file to a Digital Ocean Space
    
    Args:
        spaces_client (client): Your DigitalOcean Spaces client from get_spaces_client()
        space_name (str): Unique name of your space. Can be found at your digitalocean panel
        file_src(str): File location on your disk
        save_as(str): Where to save your file in the space
    
    Returns:
        aws response
    
    """

    is_public = kwargs.get("is_public", False)
    content_type = kwargs.get("content_type")
    meta = kwargs.get("meta")
    if not content_type:
        file_type_guess = mimetypes.guess_type(file_src)

        if not file_type_guess[0]:
            raise Exception("We can't identify content type. Please specify directly via content_type arg.")

        content_type = file_type_guess[0]

    extra_args = {
        'ACL': "public-read-write",
        'ContentType': content_type
    }

    if isinstance(meta, dict):
        extra_args["Metadata"] = meta

    return spaces_client.upload_file(
        os.path.abspath(file_src),
        space_name,
        save_as,

        # boto3.s3.transfer.S3Transfer.ALLOWED_UPLOAD_ARGS
        ExtraArgs=extra_args
    )


def upload_pil_to_space(spaces_client, space_name, pilImage, save_as, **kwargs):
    
    """
    Upload a PIL Image to a Digital Ocean Space
    
    Args:
        pilImage (PIL.Image): A PIL image
        spaces_client (client): Your DigitalOcean Spaces client from get_spaces_client()
        space_name (str): Unique name of your space. Can be found at your digitalocean panel
        file_src(str): File location on your disk
        save_as(str): Where to save your file in the space
    
    Returns:
        aws response
    
    """


    # Save the image to an in-memory file
    in_mem_file = io.BytesIO()
    pilImage.save(in_mem_file, format=pilImage.format)
    in_mem_file.seek(0)
    content_type = "image/jpeg"
    meta = kwargs.get("meta")
    extra_args = {
        'ACL': "public-read-write",
        'ContentType': content_type
    }

    if isinstance(meta, dict):
        extra_args["Metadata"] = meta

    return spaces_client.upload_fileobj(
        in_mem_file,
        space_name,
        save_as,

        # boto3.s3.transfer.S3Transfer.ALLOWED_UPLOAD_ARGS
        ExtraArgs=extra_args
    )

def upload_bytes_array_to_space(spaces_client, space_name, file_body, save_as, **kwargs):
    """
    Upload a byte array to a Digital Ocean Space
    
    Args:
        file_body (byte[]): The file as a byte array
        spaces_client (client): Your DigitalOcean Spaces client from get_spaces_client()
        space_name (str): Unique name of your space. Can be found at your digitalocean panel
        file_src(str): File location on your disk
        save_as(str): Where to save your file in the space
    
    Returns:
        aws response
    
    """

    is_public = kwargs.get("is_public", False)
    content_type = kwargs.get("content_type")
    meta = kwargs.get("meta")

    args = {
        "Bucket": space_name,
        "Body": file_body,
        "Key": save_as,
        "ACL": "public-read"
    }

    if content_type:
        args["ContentType"] = content_type

    if isinstance(meta, dict):
        args["Metadata"] = meta

    return spaces_client.put_object(**args)

def CreateNewMediaContent(name = "Untitled", url = "",user = 152, description = "", nsfw = False, remixable = True, metadata = "", tags = "AI", project = "", app = "", settings = "", visibility = "Public", views = 1):
    """
    Create a new media content object
    
    Args:
        name (str, optional): The name of the media content
        url (str, optional): The url of the media content
        user (int, optional): The user id of the media content
        description (str, optional): The description of the media content
        nsfw (bool, optional): Is the media content nsfw
        remixable (bool, optional): Is the media content remixable
        metadata (str, optional): The metadata of the media content
        tags (str, optional): The tags of the media content
        project (str, optional): The project of the media content
        app (str, optional): The app of the media content
        settings (str, optional): The settings of the media content
        visibility (str, optional): The visibility of the media content
        views (int, optional): The views of the media content
        
    Returns:
        dict: The media content object
    """
    
    media = {
		"name": name,
		"url": url,
		"description": description,
		"nsfw": nsfw,
		"remixable": remixable,
		"user": {
			"id": user,
        },
		"metadata": metadata,
		"tags": tags,
		"project": project,
		"app": app,
		"settings": settings,
		"Visibility": visibility,
		"views": views,
		"Comments": [],
		"achievements": [],
		"playlists": [
		],
		"upvotes": [
		],
		"favs": []
	}
    return media
def GenerateJobID():
    """
    Generate a new job id
    
    Returns:
        str: The job id
    """
    url = "https://faas-nyc1-2ef2e6cc.doserverless.co/api/v1/web/fn-33d7e743-bdcf-4b28-8afd-5589d75e6700/uaibackend/uaibackendgeneratejobcode"
    
    return requests.get(url).json()["jobCode"]
def CreateJob(route, stringData,user,  jobID = "",  organization = "0000000000", processingServer = "0000"):
    """
    Create a new UAI Job object
    
    Args:
        route (str): The route of the job
        stringData (str): The data of the job
        user (str): The user id of the job
        organization (str, optional): The organization id of the job
        processingServer (str, optional): The processing server id of the job
    
    Returns:
        dict: The UAI Job object
    
    """
    url = "https://faas-nyc1-2ef2e6cc.doserverless.co/api/v1/web/fn-33d7e743-bdcf-4b28-8afd-5589d75e6700/uaibackend/uaibackendgeneratejobcode"
    
    job = {"job": {
		"addDate": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
		"data": stringData,
		"finishedDate": "1999-02-01T00:00:00.000Z",
		"jobID": jobID if jobID != "" else GenerateJobID(),
		"organization": organization,
		"processingServer": processingServer,
		"route": route,
		"output":[],
		"status": "waiting",
		"user": user,
	"progress":0
	}}
    return job

def AddJob(job):
    """
    Add a new UAI Job to the database
    
    Args:
        job (dict): The UAI Job object
    Returns:
        dict: The UAI Response object
        
    """
    request = requests.post("https://faas-nyc1-2ef2e6cc.doserverless.co/api/v1/web/fn-33d7e743-bdcf-4b28-8afd-5589d75e6700/uaibackend/uaibackendaddjob", json=job)
    return request.json()

def AddCreateJob(route, stringData,user,  organization = "0000000000", processingServer = "0000"):
    """
    Add and Create a new UAI Job object
    
    Args:
        route (str): The route of the job
        stringData (str): The data of the job
        user (str): The user id of the job
        organization (str, optional): The organization id of the job
        processingServer (str, optional): The processing server id of the job
    
    Returns:
        dict: The UAI Job object
    
    """
    randomID = str(uuid.uuid4())
    job = CreateJob(route, stringData,user,randomID,  organization , processingServer )
    AddJob(job)
    return job



def GetAUserStrapiID(user):
    """
    Get a User Strapi ID from an Auth0 ID
    
    Args:
        user (str): The Auth0 ID of the user
    Returns:
        str: The Strapi ID of the user
    
    """
    request = requests.get("https://faas-nyc1-2ef2e6cc.doserverless.co/api/v1/web/fn-33d7e743-bdcf-4b28-8afd-5589d75e6700/uaibackend/uaibackendgetuserstrapiid/",json = {"user":user})
    return request.json()['user']

def AddMediaContent(mediaContent):
    """
    Add a new media content object to the database.
    
    Args:
        mediaContent (dict): The media content object
    Returns:
        dict: The UAI Response object with the media content id included
    
    """
    request = requests.post("https://faas-nyc1-2ef2e6cc.doserverless.co/api/v1/web/fn-33d7e743-bdcf-4b28-8afd-5589d75e6700/uaibackend/uaibackendaddnewmediacontent",json = {"media":mediaContent})
    print(request.reason)
    print(request.status_code)
    return request.json()

def AddNewMediaContent(url,route, stringData,user):
    """
    Add and create a new Media Content object and add it to the database.
    
    Args:
        url (str): Filepath of the file in the space
        route (str): The route of the job
        stringData (str): The data of the job
        user (str): The user id of the job
        
    Returns:
        dict: The UAI Response object with the media content id included
    """
    media =  AddMediaContent(CreateNewMediaContent(name = "Untitled", url = url,user = user, description = "", nsfw = False, remixable = True, metadata = stringData, tags = "AI", project = "", app = "", settings = route, visibility = "Private", views = 0))['media']
    media['user'] = { "id" : user}
    return media


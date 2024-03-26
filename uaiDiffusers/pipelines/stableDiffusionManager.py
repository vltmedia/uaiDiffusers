from uaiDiffusers.pipelines.jsonParse import MultiImagesToJSONResponse
import torch
from PIL import Image
import cv2
import numpy as np
from diffusers import StableDiffusionInstructPix2PixPipeline, StableDiffusionPipeline
from diffusers import EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, DDIMInverseScheduler, DDIMScheduler, DDPMScheduler, DPMSolverSDEScheduler , StableDiffusionControlNetPipeline, ControlNetModel, DPMSolverMultistepScheduler

def tempResponse():
    print("")

def tempProgress(percent, message, isDone):
    print("")

def tempProgressResponse(steps, timescale, latents):
    print("")

def tempImageProgressResponse(diffusers, steps, timescale, latents):
    return latents



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


class StableDiffusionManager:
    """
    Manager for Stable Diffusion usage and displosing.
    
    """
    
    instance = None
    
    def __init__(self, pipe = None, sendProgress = tempProgress, onFinishedCallback = tempResponse, progressCallback = tempProgressResponse, imageProgressCallback = tempImageProgressResponse):
        """
        Attributes:
            pipe  (StableDiffusionPipeline): The Stable Diffusion pipeline.
            modelName (str): The name of the model used.
            device (str): The device used. Defaults to "cuda".
            scheduler (StableDiffusionScheduler): The scheduler used.
            schedulerModel (StableDiffusionScheduler): The scheduler model used. Defaults to EulerAncestralDiscreteScheduler.
            loraWeights (str): The weights used for LoRA.
            textualInversionWeights (str, optional): The weights used for Textual Inversion.
            customSDBinWeights (str, optional): The weights used for Custom SDBin.
        
        """

        self.pipe  = pipe    
        self.modelName = "-"   
        self.device = "cuda" 
        self.scheduler  = None   
        self.steps  = 20   
        self.schedulerModel  = EulerAncestralDiscreteScheduler   
        self.sendProgress  = sendProgress   
        self.onFinishedCallback  = onFinishedCallback   
        self.imageProgressCallback  = imageProgressCallback   
        self.onProgressCallback  = progressCallback   
        self.loraWeights = ""
        self.textualInversionWeights = ""
        self.customSDBinWeights = ""
        self.aiType = "SD:GEN"
        StableDiffusionManager.InitInstance(self)
        
    def __del__(self):
        self.CleanCache()
        
    @staticmethod
    def InitInstance(manager):
        if StableDiffusionManager.instance == None:
            StableDiffusionManager.instance = manager
        
    @staticmethod
    def GetInstance():
        if StableDiffusionManager.instance == None:
            StableDiffusionManager.instance = StableDiffusionManager()
        return StableDiffusionManager.instance
        
        
        
        
    def SayHi(self):
        print("Hi")
        return "Hi"
    
    def InitPipe(self, modelName="" ,device = "cuda",loraWeights = "",textualInversionWeights = "",customSDBinWeights = "",scheduler = ""):
        """
        Initializes the Stable Diffusion pipeline.
        
        Args:
            modelName (str, optional): The name of the model used. Defaults to "".
            device (str, optional): The device used. Defaults to "cuda".
            loraWeights (str, optional): The weights used for LoRA. Defaults to "".
            textualInversionWeights (str, optional): The weights used for Textual Inversion. Defaults to "".
            customSDBinWeights (str, optional): The weights used for Custom SDBin. Defaults to "".  
            scheduler (str, optional): The scheduler used. Defaults to ""
                    
        """
        shouldRefresh = self.modelName != modelName  or self.loraWeights != loraWeights or self.textualInversionWeights != textualInversionWeights or self.customSDBinWeights != customSDBinWeights or self.scheduler != scheduler
        self.device = device
        self.sendProgress(0.08, "Load Model", False)
        
        self.LoadModel(modelName,device, False)
        self.sendProgress(0.09, "Load Lora Weights", False)
        
        self.LoadLoraWeights( loraWeights)
        self.sendProgress(0.1, "Load Textual Inversion Weights", False)
        
        self.LoadTextualInversionWeights( textualInversionWeights)
        self.sendProgress(0.11, "Load Custom SD Bin Weights", False)
        
        self.LoadCustomSDBinWeights( customSDBinWeights)
        self.sendProgress(0.12, "Change Scheduler", False)
        
        self.ChangeScheduler(scheduler, shouldRefresh)
        self.sendProgress(0.13, "Pipeline Setup", False)
        
        
    def GenerateImageXL(self, sdRepo =  "stabilityai/stable-diffusion-xl-base-1.0", loraPath = "minimaxir/sdxl-wrong-lora", textualInversion = "",customSDBin = "", refinerRepo = "", vaeRepo = "madebyollin/sdxl-vae-fp16-fix", imagesToGenerate = 1, steps = 20, device="cuda", prompt_ = "A person", negPrompt_ = "bad face", seed=42, width=1024, height=1024, textGuidance=7.5,  high_noise_frac=0.8, enableCPUOffload = True,enable_attention_slicing=False,enable_vae_tiling=False,callback = None):
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
    
        
        
        
    def isModelLoaded(self):
        """
        Check if the pipe is loaded and not None.
        
        Returns:
            isModelLoaded bool: True if the pipe is loaded and not None, False otherwise.
        """
        return self.pipe != None
        
    def LoadModel(self, modelName = "tiny", device = "cuda", shouldRefresh = True):
        self.device = device
        if self.modelName == modelName:
            return
        else:
            self.modelName = modelName
            if shouldRefresh == True:
                self.RefreshModel()
            
    def RefreshModel(self):
            self.scheduler = self.schedulerModel.from_pretrained(self.modelName, subfolder="scheduler")
            if self.aiType == "SD:GEN":
                self.pipe = StableDiffusionPipeline.from_pretrained(self.modelName, scheduler=self.scheduler, torch_dtype=torch.float16, safety_checker=None)
            elif self.aiType == "pix2pix":
                self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(self.modelName,  scheduler=self.scheduler, torch_dtype=torch.float16, safety_checker=None)
            self.pipe = self.pipe.to(self.device)
                    
            
            if self.customSDBinWeights != "":   
                self.pipe.unet.load_attn_procs(self.customSDBinWeights)
            if self.textualInversionWeights != "":
                self.pipe.load_textual_inversion(self.textualInversionWeights)
            self.pipe.enable_xformers_memory_efficient_attention()
            
    def CleanCache(self):
        del(self.pipe)
        self.pipe = None
        
    def LoadPipe(self, pipe):
        del(self.pipe)
        self.pipe = pipe
        
    def SetAIType(self, aiType = "SD:GEN"):
        self.aiType = aiType
        
        
    def ChangeScheduler(self, schedulerModel = "EulerAncestralDiscreteScheduler", shouldRefresh = True):
        if self.schedulerModel == schedulerModel:
            return
        if self.schedulerModel == "EulerAncestralDiscreteScheduler":
            self.schedulerModel = EulerDiscreteScheduler
        elif self.schedulerModel == "EulerDiscreteScheduler":
            self.schedulerModel = EulerAncestralDiscreteScheduler
        elif self.schedulerModel == "DDIMScheduler":
            self.schedulerModel = DDIMScheduler
        elif self.schedulerModel == "DDIMInverseScheduler":
            self.schedulerModel = DDIMInverseScheduler
        elif self.schedulerModel == "DPMSolverSDEScheduler":
            self.schedulerModel = DPMSolverSDEScheduler
        elif self.schedulerModel == "DDPMScheduler":
            self.schedulerModel = DDPMScheduler
        if shouldRefresh:
            self.RefreshModel()
        
    def ChangeModelImage(self, modelName):
        self.LoadModel(modelName)
        
    def UnloadFinetunedWeights(self):
        self.RefreshModel()
        
    def LoadLoraWeights(self, loraWeights_):
        if self.loraWeights != loraWeights_:
            self.loraWeights = loraWeights_
            if self.loraWeights != "":
                self.pipe.load_attn_procs(self.loraWeights)
        
    def LoadCustomSDBinWeights(self, customSDBinWeights_):
        if self.customSDBinWeights != customSDBinWeights_:
            self.customSDBinWeights = customSDBinWeights_
            if self.customSDBinWeights != "":
                self.pipe.load_attn_procs(self.customSDBinWeights)
        
    def LoadTextualInversionWeights(self, textualInversionWeights_):
        if self.textualInversionWeights != textualInversionWeights_:
            
            self.textualInversionWeights = textualInversionWeights_
            if self.textualInversionWeights != "":
                self.pipe.load_textual_inversion(self.textualInversionWeights)
                
    def progressCallback(self, step, timeScale, tensor):
        percent = step / self.steps
        self.sendProgress(percent, f"{str(int(percent * 100))}% conjuring image.", False)
        
    def GenerateImage(self, sdRepo = "",  prompt = "A person", negPrompt = "bad face", imagesToGenerate = 1, steps = 20, device="cuda",seed=42, width=512, height=512, textGuidance = 7.5, loraPath = "", textualInversion = "",customSDBin = "",  enableCPUOffload = True,enable_attention_slicing=False,enable_vae_tiling=False):
        return GenerateImage( sdRepo =  sdRepo, loraPath = loraPath, textualInversion = textualInversion,customSDBin = customSDBin, imagesToGenerate = imagesToGenerate, steps = steps, device=device, prompt_ = prompt, negPrompt_ = negPrompt, seed=seed,textGuidance = textGuidance,  enableCPUOffload = enableCPUOffload,enable_attention_slicing=enable_attention_slicing,enable_vae_tiling=enable_vae_tiling, pipe_=self.pipe, width=width, height=height, callback=self.imageProgressCallback)
    
    def GenerateImageRESTAPI(self, sdRepo = "",   prompt = "A person", negPrompt = "bad face", imagesToGenerate = 1, steps = 20, device="cuda",seed=42, width=512, height=512, textGuidance = 7.5 , loraPath = "", textualInversion = "",customSDBin = "",schedueler = "EulerAncestralDiscreteScheduler",  enableCPUOffload = True,enable_attention_slicing=False,enable_vae_tiling=False):
    # Get from JSON request ['prompt']
        model = sdRepo
        # if model != SDModel_id:
        #     ChangeModelImage(model)
        self.steps = steps
        self.InitPipe(model, device, loraPath, textualInversion, customSDBin, schedueler )
        if self.aiType != "SD:GEN":
            self.aiType = "SD:GEN"
            self.RefreshModel()
        self.sendProgress(0.14, "Stable Diffusion Starting Image Generation.", False)
        
        
        images = self.GenerateImage( prompt = prompt,negPrompt = negPrompt ,imagesToGenerate = imagesToGenerate, steps = steps,  device = device,seed = seed, width = width, height =height, textGuidance = textGuidance,loraPath = loraPath, textualInversion = textualInversion, customSDBin = customSDBin,enableCPUOffload=enableCPUOffload, enable_attention_slicing=enable_attention_slicing, enable_vae_tiling=enable_vae_tiling)
        
        outResponse = MultiImagesToJSONResponse(images, [prompt for i in range(len(images))])
        self.onFinishedCallback( outResponse)
        return outResponse
        
    def GenerateImageXLRESTAPI(self, sdRepo = "",   prompt = "A person", negPrompt = "bad face", imagesToGenerate = 1, steps = 20, device="cuda",seed=42, width=512, height=512, textGuidance = 7.5 , loraPath = "", textualInversion = "",customSDBin = "", refinerRepo = "", vaeRepo = "madebyollin/sdxl-vae-fp16-fix", schedueler = "EulerAncestralDiscreteScheduler",  high_noise_frac=0.8,  enableCPUOffload = True,enable_attention_slicing=False,enable_vae_tiling=False):
    # Get from JSON request ['prompt']
        # if model != SDModel_id:
        #     ChangeModelImage(model)
        self.steps = steps
        
        images = self.GenerateImageXL( sdRepo =  sdRepo, loraPath = loraPath, textualInversion = textualInversion,customSDBin = customSDBin, imagesToGenerate = imagesToGenerate, steps = steps, device=device, prompt_ = prompt, negPrompt_ = negPrompt, seed=seed,textGuidance = textGuidance,  enableCPUOffload = enableCPUOffload,enable_attention_slicing=enable_attention_slicing,enable_vae_tiling=enable_vae_tiling, width=width, height=height, high_noise_frac=high_noise_frac, refinerRepo=refinerRepo, vaeRepo=vaeRepo, callback=self.onProgressCallback)
        outResponse = MultiImagesToJSONResponse(images, [prompt for i in range(len(images))])
        self.onFinishedCallback( outResponse)
        return outResponse
        

    # def download_image(self, url, channels = "RGB"):
    #     image = Image.open(requests.get(url, stream=True).raw)
    #     image = ImageOps.exif_transpose(image)
    #     image = image.convert(channels)
    #     return image

    def Pix2PixDiffusers(self, pilImage, model_id = "timbrooks/instruct-pix2pix",schedueler = "EulerAncestralDiscreteScheduler", num_inference_steps = 10, image_guidance_scale = 1,prompt = "turn him into cyborg", negative_prompt = "blurry, low quality", device = "cuda", seed = 42, num_images_per_prompt=2, callback = None, callbackSteps = 2):
        """
        Generate an image using the Pix2Pix model. The image is generated using the prompt and negative prompt with an input image as the driver.

        Args:
            pilImage (PIL.Image): The input image to use as the driver.
            model_id (str, optional): The model to use. 
            schedueler (str, optional): The scheduler to use.
            num_inference_steps (int, optional): The number of inference steps to use. .
            image_guidance_scale (int, optional): The image guidance scale to use.
            prompt (str, optional): The prompt to use.
            negative_prompt (str, optional): The negative prompt to use.
            device (str, optional): The device to use. 
            seed (int, optional): The seed to use. 
            callback (function, optional): The callback function to use. 
            callbackSteps (int, optional): The number of steps between each callback.
            

        """
        print(schedueler)
        if self.aiType != "pix2pix":
            self.SetAIType("pix2pix")
            self.CleanCache()
            # self.ChangeScheduler(schedueler, False)
            
            self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
            self.pipe = self.pipe.to(device)
        generator = torch.Generator().manual_seed(seed)
        images = self.pipe(prompt, image=pilImage.convert('RGB'), num_inference_steps=num_inference_steps, image_guidance_scale=image_guidance_scale, negative_prompt=negative_prompt, generator=generator, num_images_per_prompt=num_images_per_prompt).images
        # Return the last num_images_per_prompt images. slice from the images list
        outarray = []
        for indx in range(num_images_per_prompt):
            outarray.append(images[(len(images) - 1) - indx])
        print(len(images))
        print(len(outarray))
        return outarray
        # return "Hey"
        

    def AugmentImageCanny(self, inputPILImages = [] , seed =424,device = "cuda", controlNetRepo = "lllyasviel/sd-controlnet-canny",sdrepo ="runwayml/stable-diffusion-v1-5" ,imageSize = 512, low_threshold = 100, high_threshold = 200,
                        prompt="", negPrompt="", num_inference_steps=80, controlnet_conditioning_scale=0.5,num_images_per_prompt=1,  nodeid = "0000000000"):
        """
        Augment an image into another image with a given prompt, negative prompt, and input image.
        
        Args:
            inputPILImages (list, optional): The input PIL images to use as the driver.
            seed (int, optional): The seed to use.
            device (str, optional): The device to use.
            controlNetRepo (str, optional): The control net repo to use.
            sdrepo (str, optional): The sd repo to use.
            imageSize (int, optional): The image size to use.
            low_threshold (int, optional): The low threshold to use.
            high_threshold (int, optional): The high threshold to use.
            prompt (str, optional): The prompt to use.
            negPrompt (str, optional): The negative prompt to use.
            num_inference_steps (int, optional): The number of inference steps to use.
            controlnet_conditioning_scale (float, optional): The control net conditioning scale to use  
            num_images_per_prompt (int, optional): The number of images per prompt to use.
            
        Returns:
            list (list): The list of PIL images.
        
        """
        
        generator = torch.Generator(device=device).manual_seed(seed)
        if self.aiType != "SD:CANNY":
            self.SetAIType("SD:CANNY")
            self.CleanCache()
                
            if "krea/aesthetic-controlnet" in controlNetRepo == False:
                controlnet = ControlNetModel.from_pretrained(
                    controlNetRepo,
                    torch_dtype=torch.float16,
                )
                
                self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
                    sdrepo,
                    controlnet=controlnet,
                    torch_dtype=torch.float16
                )
                # change the scheduler
                self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
            else:
                    
                self.pipe = StableDiffusionControlNetPipeline.from_pretrained("krea/aesthetic-controlnet").to("cuda")
                self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)
                # enable xformers (optional), requires xformers installation
                self.pipe.enable_xformers_memory_efficient_attention()
                # cpu offload for memory saving, requires accelerate>=0.17.0
                self.pipe.enable_model_cpu_offload()
        outImages = []                
        for indx, image in enumerate(inputPILImages):
            image = PILToCV2(image)
            print("Loading next file")
            image = np.array(image)
            # scale image to 512x512 with padding if not square ratio
            # image = cv2.resize(image, (imageSize, imageSize), interpolation=cv2.INTER_NEAREST)
            image = cv2.Canny(image, low_threshold, high_threshold)
            image = image[:, :, None]
            image = np.concatenate([image, image, image], axis=2)
            canny_image = Image.fromarray(image)

            # cpu offload for memory saving, requires accelerate>=0.17.0
            # pipe.enable_model_cpu_offload()
            images = self.pipe(
                prompt,
                negative_prompt=negPrompt,
                num_inference_steps=num_inference_steps,
                num_images_per_prompt=num_images_per_prompt,
                generator=generator,
                image=canny_image,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                callback=self.progressCallback
            ).images
            for image in images:
                outImages.append(image)
            uaiTCPclient.sendProgress(progress=0.9,total= 1, message = f"Generating files", status = "Cleanup", nodeid=nodeid)
            
        return outImages
    
    
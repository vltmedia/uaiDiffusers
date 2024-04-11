import io
from PIL import Image
import json
import glob
import os
import base64
from diffusers import I2VGenXLPipeline
from diffusers.utils import export_to_gif, load_image, export_to_video
import torch
from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler, AutoencoderKL
import base64
from PIL import Image
from diffusers import AnimateDiffPipeline, LCMScheduler, MotionAdapter
from diffusers.utils import export_to_gif
from huggingface_hub import hf_hub_download
from diffusers import StableVideoDiffusionPipeline
from uaiDiffusers.common.imageRequest import ImageRequest
from safetensors.torch import load_file
from uaiDiffusers.pipelines.stableDiffusionManager import StableDiffusionManager
from uaiDiffusers.common.utils import UAICallBacks
from uaiDiffusers import uaiDiffusers
pipe = None
ip_model = None
modelName = ""

h264string = " libx264 -pix_fmt yuv420p -profile:v baseline -level 3.0 -preset slow -crf 12"

def RegisterAI(packageName = "StableDiffusion_1_5",pipeType="IG:SD" ):
    needsRegister = False
    instance = StableDiffusionManager.GetInstance(packageName)
    if instance is None:
        needsRegister = True
        # instance = StableDiffusionManager(None, packageName, "IG:SD", uaiDiffusers.SendCLRProgress, uaiDiffusers.SetResponseMessage, uaiDiffusers.CLRProgressXLLightningCallback)
        instance = StableDiffusionManager( pipe = None, pipeName=packageName,pipeType=pipeType, sendProgress = uaiDiffusers.SendCLRProgress, onFinishedCallback = uaiDiffusers.SetResponseMessage, progressCallback = uaiDiffusers.CLRProgressCallback, imageProgressCallback = uaiDiffusers.CLRProgressXLLightningCallback)
    return instance, needsRegister


def SanitizeRequest(request):
 
    ratio = 16/9
    maxSize = 512
    seed = 465
    steps = 50
    topOffset = 0
    guidance_scale = 1.0
    fps = 16
    loops = 1
    num_images_per_prompt  = 16
    encodedFps  = 16
    encodeSize = [1024, 576]
    exportGif = False
    size = [1920, 1080]
    if "ratio" not in request:
        request['ratio'] = ratio 
    if "maxSize" not in request:
        request['maxSize'] = maxSize
    if "seed" not in request:
        request['seed'] = seed
    if "steps" not in request:
        request['steps'] = steps
    if "topOffset" not in request: 
        request['topOffset'] = topOffset
    if "exportGif" not in request:
        request['exportGif'] = exportGif
    if "fps" not in request:
        request['fps'] = fps
    if "loops" not in request:
        request['loops'] = loops
    if "num_images_per_prompt" not in request:
        request['num_images_per_prompt'] = num_images_per_prompt
    if "size" not in request: 
        request['size'] = size
    if "encodedFps" not in request:
        request['encodedFps'] = encodedFps
    if "encodeSize" not in request:
        request['encodeSize'] = encodeSize
    if "guidance_scale" not in request:
        request['guidance_scale'] = guidance_scale
    outputPath = request['output'].replace(".png", ".mp4")
    outputPath = outputPath +  ".mp4"
    root, ext = os.path.splitext(outputPath)
    if not ext:
        ext = '.mp4'
    request['output'] = root + ext
    if not os.path.exists(os.path.dirname(request['output'])):
        os.makedirs(os.path.dirname(request['output']))
    
    request = ImageRequest.FromDict(request)
    request.DetectSizeFromExportSize()
    request.DetectEncodeSize()
    return request


def ResizeImage(image, maxSize = 1024):

    width = maxSize
    ratio = image.size[0] / image.size[1]
    height = int(ratio * width)
    if image.size[1] > image.size[0]:
        height = 1024
        width = int(ratio * height)
    return image.resize((width, height))


def loadImage(image, aspect_ratio=16/9, max_size=1024, top_offset=0):
    image = image.convert("RGB")
    image = ResizeImage(image, max_size)
    width, height = image.size
    new_width, new_height = image.size
    sixteenNineSize = [1024, 576]
    # Resize the image with the longest side being max_size
    initial_ratio = width / height
    if width > height:
        new_width = max_size
        new_height = int(max_size / initial_ratio)
    else:
        new_height = max_size
        new_width = int(max_size * initial_ratio)
    image = image.resize((new_width, new_height))

    if width / height != aspect_ratio:
        # Crop the image to the correct aspect ratio and the max_size depending on the longest side
        if new_width / new_height > aspect_ratio:
            new_width = new_height * aspect_ratio
        else:
            new_height = new_width / aspect_ratio

        left = (width - new_width) / 2
        top = ((height - new_height) / 2) + top_offset
        right = (width + new_width) / 2
        bottom = (height + new_height) / 2

        image = image.crop((left, top, right, bottom))
    else:
        if image.size != (sixteenNineSize[0], sixteenNineSize[1] ) and aspect_ratio == 16/9:
            image = image.resize((sixteenNineSize[0], sixteenNineSize[1]))
    return image
    




def ImageToBase64(image):
    import base64
    import io
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')



def Text2VideoProcess( prompt, image, steps, negative_prompt, generator, num_frames=16, targetFPS=16 ):
    instance, needsRegister = RegisterAI("StableDiffusion_Video", "VG:SDI2VGenXL")
    
    frames = instance.pipe(
    prompt=prompt,
    image=image,
    num_inference_steps=steps,
    negative_prompt=negative_prompt,
    generator=generator,
    num_frames=num_frames,
    target_fps=targetFPS
).frames
    return frames


def StableVideoText2VideoProcess(  image,  seed, num_frames=16, targetFPS=16, motion_bucket_id=170, noise_aug_strength=0.3, debug=False, outputPath = "output.mp4" ):
    global pipe
    global modelName
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)) )
    # from simple_video_sample import sample

    instance, needsRegister = RegisterAI("StableDiffusion_Video", "VG:SDXT")
    
    if needsRegister:
        instance.pipe = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
        )
    #pipe.enable_model_cpu_offload()
    instance.pipe.to(0) # Force to GPU
    if debug == True:
        return "Debug"
    # Load the conditioning image
    ratio = 16/9
    image = image
    if image.size != (1024, 576):
        imageRatio = image.size[0] / image.size[1]
        if imageRatio != ratio:
            # crop the image to the correct aspect ratio
            if imageRatio > ratio:
                width = image.size[1] * ratio
                left = (image.size[0] - width) / 2
                right = (image.size[0] + width) / 2
                image = image.crop((left, 0, right, image.size[1]))
            else:
                height = image.size[0] / ratio
                top = (image.size[1] - height) / 2
                bottom = (image.size[1] + height) / 2
                image = image.crop((0, top, image.size[0], bottom))
        image = image.resize((1024, 576))
    image = image.convert("RGB")
    generator = torch.manual_seed(seed)
    # Perform GPU memory cleanup
    frames = instance.pipe(image, decode_chunk_size=14, generator=generator, num_frames=num_frames, motion_bucket_id=motion_bucket_id, noise_aug_strength=noise_aug_strength).frames[0]
    export_to_video(frames, "output_.mp4", fps=targetFPS)
    import ffmpy
    ff = ffmpy.FFmpeg(
                        inputs = {"output_.mp4" : None},
                        outputs = {outputPath :  f" -y -c:v h264 " }    
                    ) 
    ff.cmd
    ff.run()
    torch.cuda.empty_cache()

    return outputPath

def Image2VideoProcessStableVideo(request):

        request = SanitizeRequest(request)
        image_ = Image.open(io.BytesIO(base64.b64decode(request.input)))
        output = StableVideoText2VideoProcess(  image_,  request.seed, num_frames=request.num_frames, targetFPS=request.fps, motion_bucket_id=request.motion, noise_aug_strength=request.guidance, debug=False, outputPath=request.output )
        torch.cuda.empty_cache()
        outputDict = {"media":[{"media":request.output,"prompt": request.prompt,
                "seed": request.seed}]}
        return json.dumps(outputDict)

def loadAnimateLCMModel(request):

    adapter = MotionAdapter.from_pretrained("wangfuyun/AnimateLCM", torch_dtype=torch.float32)
    pipe = AnimateDiffPipeline.from_pretrained(request.model, motion_adapter=adapter, torch_dtype=torch.float32)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config, beta_schedule="linear")

    pipe.load_lora_weights("wangfuyun/AnimateLCM", weight_name="AnimateLCM_sd15_t2v_lora.safetensors", adapter_name="lcm-lora")
    pipe.set_adapters(["lcm-lora"], [0.8])
    pipe.to("cuda")
    # pipe.enable_vae_slicing()
    # pipe.enable_model_cpu_offload()
    return pipe

def Text2VideoProcessAnimateLCM( prompt, negative_prompt, generator, num_frames=16, encodeSize=[512, 512]):
    instance, needsRegister = RegisterAI("StableDiffusion_Video", "VG:SDXLL")
    
    if needsRegister:
        instance.pipe = loadAnimateLCMModel()
        
    output = instance.pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_frames=num_frames,
        guidance_scale=2.0,
        num_inference_steps=6,
        generator=generator,
        width=encodeSize[0],
        height=encodeSize[1],
    )
    frames = output.frames
    return frames


def ExportFramesToVideo(frames, fps = 8, encodedFps=8, exportGif = False):
    from diffusers.utils import export_to_gif, export_to_video
    import ffmpy
    if exportGif:
        export_to_gif(frames, "output.gif")
        return  "output.gif"
    else:
        inputFile = "output_.mp4"
        outputFile = "output.mp4"
        export_to_video(frames,inputFile, fps)
        if fps != encodedFps:
            ff = ffmpy.FFmpeg(
                inputs = {inputFile : None},
                outputs = {outputFile :  f" -y -c:v h264 -filter:v minterpolate -r {encodedFps} " }    
            ) 
            print(ff.cmd) #optional
            ff.cmd
            ff.run()
        else:
            ff = ffmpy.FFmpeg(
                inputs = {inputFile : None},
                outputs = {outputFile :  f" -y -c:v h264 " }    
            ) 
            print(ff.cmd) #optional
            ff.cmd
            ff.run()
            # outputFile = inputFile
        return outputFile

def TI2VXLLProcess(request, callbacks = None):

    if callbacks is None:
        callbacks = UAICallBacks()
        
    request = SanitizeRequest(request)
    
    open("P:/temp/TI2VXLLProcess.json", "w").write(request.JSON())
    
    callbacks.sendProgress(0, "Starting", False)

    device = "cuda"
    dtype = torch.float16

    step = 8  # Options: [1,2,4,8]
    repo = "ByteDance/AnimateDiff-Lightning"
    ckpt = f"animatediff_lightning_{step}step_diffusers.safetensors"
    base = "SG161222/Realistic_Vision_V6.0_B1_noVAE"  # Choose to your favorite base model.
    vaeModel = "stabilityai/sd-vae-ft-mse"
    
    
    
    instance, needsRegister = RegisterAI("StableDiffusion_Video", "VG:SDXLL")
    callbacks.sendProgress(0.25, "Loading Model", False)
    
    if needsRegister:
        callbacks.sendProgress(0.27, "Loading Adapter Model", False)
        
        adapter = MotionAdapter().to(device, dtype)
        adapter.load_state_dict(load_file(hf_hub_download(repo ,ckpt), device=device))
        callbacks.sendProgress(0.29, "Loading VAE Model", False)
        
        vae = AutoencoderKL.from_pretrained(vaeModel, torch_dtype=dtype).to("cuda")
        # callbacks.sendProgress(0.3, "Loading Base Model", False)

        instance.pipe = AnimateDiffPipeline.from_pretrained(base, vae=vae, motion_adapter=adapter,  torch_dtype=dtype).to(device)
        instance.pipe.scheduler = EulerDiscreteScheduler.from_config(instance.pipe.scheduler.config, timestep_spacing="trailing", beta_schedule="linear")
        instance.pipe.to(device)
    callbacks.sendProgress(0.35, "Generating Media", False)
    
    generator = torch.manual_seed(request.seed)
    generatedFrames = []
    
    
    # if request['inputMode'] == True:
    #     image_ = Image.open(io.BytesIO(base64.b64decode(outRequest['image'])))
    # if base64Image:
    #     image = Image.open(io.BytesIO(base64.b64decode(request['image'])))
    # else:
    #     image = Image.open(request['image'])
    # image = loadImage(image, ratio,  maxSize, topOffset)

    frames = instance.pipe(prompt=request.prompt, negative_prompt=request.neg_prompt, guidance_scale=request.guidance_scale, num_inference_steps=request.steps, num_frames=request.num_frames, width=request.encodeSize[0], height=request.encodeSize[1], generator=generator, callback_on_step_end=callbacks.imageProgressCallback)
    
    for frame in frames.frames[0]:
        generatedFrames.append(frame.resize((request.size[0], request.size[1])))
    # outputFile = ExportFramesToVideo(generatedFrames, fps = request['fps'], encodedFps=request['encodedFps'], exportGif = request['exportGif'])
    callbacks.sendProgress(0.75, "Encoding Media", False)
    root, ext = os.path.splitext(request.output)
    if not ext:
        ext = '.mp4'
    request.output = root + ext
    callbacks.sendProgress(0.75, request.output, False)
    
    from diffusers.utils import export_to_video
    
    export_to_video(generatedFrames, "output_.mp4", fps=request.encodedFps)
    import ffmpy
    ff = ffmpy.FFmpeg(
                        inputs = {"output_.mp4" : None},
                        outputs = {request.output :  f" -y -c:v h264 " }    
                    ) 
    ff.cmd
    ff.run()
    torch.cuda.empty_cache()
    
    outputDict = {"media":[{"media":request.output,"prompt": request.prompt,
            "seed": request.seed}]}
    return json.dumps(outputDict)

    return [request.output, ImageToBase64(generatedFrames[0])]
    


def RunI2VGenXLProcess(request, baseLoad= False, base64Image=True):
    import torch
    import ffmpy
    from diffusers import I2VGenXLPipeline
    from diffusers.utils import export_to_gif, load_image, export_to_video
    import os
    
    request = SanitizeRequest(request)
    instance, needsRegister = RegisterAI("StableDiffusion_Video", "VG:SDI2VGenXL")
    

    ratio = request.ratio
    maxSize = request.maxSize
    seed = request.seed
    steps = request.steps
    topOffset = request.topOffset
    fps = request.fps
    loops = request.loops
    num_frames  = request.num_frames
    encodedFps  = request.encodedFps
    exportGif = request.exportGif
    outputSize = request.outputSize

    # print(request)
    
    repo_id = "ali-vilab/i2vgen-xl"
    if needsRegister:
        instance.pipe = I2VGenXLPipeline.from_pretrained(repo_id, torch_dtype=torch.float16).to("cuda")
        
        
    if base64Image:
        image = Image.open(io.BytesIO(base64.b64decode(request.input)))
    else:
        image = Image.open(request.input)
    image = loadImage(image, ratio,  maxSize, topOffset)
    lastFrame = image
    generator = torch.manual_seed(seed)
    
    if baseLoad:
        outputImages = {"media": []}
        generatedFrames = []
        for loop in range(loops):
            print(f"Loop: {loop} / {loops} ")
            frames = Text2VideoProcess( request.prompt, lastFrame, steps, request.negative_prompt, generator, num_frames)
            for frame in frames[0]:
                
                generatedFrames.append(frame.resize((outputSize[0], outputSize[1])))
            lastFrame = frames[0][-1]
            print(f"generatedFrames: {len(generatedFrames)}")
            
        if generatedFrames is not None:
            firstFrameBase64 = ImageToBase64(generatedFrames[0])
            if exportGif:
                export_to_gif(generatedFrames, "output.gif")
                return  "output.gif"
            else:
                inputFile = "output_.mp4"
                outputFile = request.output
                export_to_video(generatedFrames,inputFile, fps)
                if fps != encodedFps:
                    ff = ffmpy.FFmpeg(
                        inputs = {inputFile : None},
                        outputs = {outputFile :  f" -y -c:v h264 -filter:v minterpolate -r {encodedFps} " }    
                    ) 
                    print(ff.cmd) #optional
                    ff.cmd
                    ff.run()
                else:
                    ff = ffmpy.FFmpeg(
                        inputs = {inputFile : None},
                        outputs = {outputFile :  f" -y -c:v h264 " }    
                    ) 
                    print(ff.cmd) #optional
                    ff.cmd
                    ff.run()
                    # outputFile = inputFile
                                
                outputDict = {"media":[{"media":request.output,"prompt": request.prompt,
                        "seed": request.seed}]}
                return json.dumps(outputDict)
                return [outputFile, firstFrameBase64]


def RunAnimateLCMProcess(request, baseLoad= False, base64Image=True):
    import ffmpy

    import os
    global pipe
    global modelName
    request = SanitizeRequest(request)
    instance, needsRegister = RegisterAI("StableDiffusion_Video", "VG:SDXLL")
    
    if needsRegister:
        instance.pipe = loadAnimateLCMModel()
        modelName = "AnimateLCM"
    generator = torch.manual_seed(request.seed)
    if baseLoad:
        outputImages = {"media": []}
        generatedFrames = []
        frames = Text2VideoProcessAnimateLCM( request.prompt,  request.negative_prompt, generator,request.num_images_per_prompt, request.encodeSize)
        for frame in frames[0]:
            
            generatedFrames.append(frame.resize((request.size[0], request.size[1])))
            
        if generatedFrames is not None:
            firstFrameBase64 = ImageToBase64(generatedFrames[0])
            if request.exportGif:
                export_to_gif(generatedFrames, "output.gif")
                return  "output.gif"
            else:
                inputFile = "output_.mp4"
                outputFile = request.output
                export_to_video(generatedFrames,inputFile, request.fps)
                if request.fps != request.encodedFps:
                    ff = ffmpy.FFmpeg(
                        inputs = {inputFile : None},
                        outputs = {outputFile :  f" -y -c:v h264 -filter:v minterpolate -r {request.encodedFps} " }    
                    ) 
                    print(ff.cmd) #optional
                    ff.cmd
                    ff.run()
                else:
                    ff = ffmpy.FFmpeg(
                        inputs = {inputFile : None},
                        outputs = {outputFile :  f" -y -c:v h264 " }    
                    ) 
                    print(ff.cmd) #optional
                    ff.cmd
                    ff.run()

            outputDict = {"media":[{"media":request.output,"prompt": request.prompt,
                    "seed": request.seed}]}
            return json.dumps(outputDict)
            return [outputFile, firstFrameBase64]


if __name__ == "__main__":
    import base64
    request = {
    "player": "Matthew Tkachuk",
    "image": base64.b64encode(open("Justin.png", "rb").read()).decode("utf-8")
}
    response = ImageToBase64(RunCompositePersonProcess(request))
    

    
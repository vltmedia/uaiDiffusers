import argparse
import cv2
import glob
import numpy as np
import os
import PIL
import torch
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    sys.path.append("/root")
except:
    pass
import base64

from basicsr.utils import imwrite

from gfpgan import GFPGANer

from uaiDiffusers.constants import GetServerMode, GetDiffusersCachePath


class FaceFix:


    def __init__(self, image = None, version = 1.3, upscale = 2, bg_upsampler = "realesrgan_", bg_tile = 400, suffix = "None", only_center_face = False, aligned = False, ext = "auto", weight = 0, modelPath=""):
        self.version = version
        self.upscale = upscale
        self.bg_upsampler = bg_upsampler
        self.bg_tile = bg_tile
        self.suffix = suffix
        self.only_center_face = only_center_face
        self.aligned = aligned
        self.ext = ext
        self.modelPath = modelPath
        self.weight = weight
        if image is not None:
            self.loadImage(image)
        else:
            self.image = None
        

    
# -------------- Help --------------

        self.version_Help = "GFPGAN model version. Option: 1 | 1.2 | 1.3. Default: 1.3"
        self.upscale_Help = "The final upsampling scale of the image. Default: 2"
        self.bg_upsampler_Help = "background upsampler. Default: realesrgan"
        self.bg_tile_Help = "Tile size for background sampler"
        self.suffix_Help = "Suffix of the restored faces"
        self.only_center_face_Help = "Only restore the center face"
        self.aligned_Help = "Input are aligned faces"
        self.ext_Help = "Image extension. Options: auto | jpg | png"
        self.weight_Help = "Adjustable weights."
        
    def loadImageAsBase64(self, image):
        bytesIO = None
        if isinstance(image, PIL.Image.Image):
            import io
            bytesIO = io.BytesIO()
            image.save(bytesIO, format='PNG')
        elif isinstance(image, str):
            self.inputimageString = image
            if os.path.isfile(image):
                bytesIO = open(image, "rb")
            else:
                bytesIO = io.BytesIO(base64.b64decode(image))
        elif isinstance(image, bytes):
            bytesIO = io.BytesIO(image)
        elif isinstance(image, np.ndarray):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = PIL.Image.fromarray(image)
            bytesIO = io.BytesIO()
            image.save(bytesIO, format='PNG')
        self.inputimage = base64.b64encode(bytesIO.getvalue()).decode('utf-8')

    def loadImageAsCV2(self, image):
        if isinstance(image, PIL.Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        elif isinstance(image, str):
            if os.path.isfile(image):
                image = cv2.imread(image)
            else:
                import base64
                import io
                image = base64.b64decode(image)
                image = cv2.imdecode(np.frombuffer(image, np.uint8), -1)
        elif isinstance(image, bytes):
            image = cv2.imdecode(np.frombuffer(image, np.uint8), -1)
        self.inputimage = image
        
    def loadImage(self, image):
        self.loadImageAsCV2(image)
        
        
    def loadImageFromDict(self, data):
        if "inputimage" in data:
            self.loadImage(data["inputimage"])
                
                
    def loadFromDict(self, data):
        self.version = data["version"]
        self.upscale = data["upscale"]
        self.bg_upsampler = data["bg_upsampler"]
        self.bg_tile = data["bg_tile"]
        self.suffix = data["suffix"]
        self.only_center_face = data["only_center_face"]
        self.aligned = data["aligned"]
        self.ext = data["ext"]
        self.weight = data["weight"]
        if "facefixModelPath" in data:
            self.modelPath = data["facefixModelPath"]
        
        self.loadImageFromDict(data)

    def RunProcessWithInput(self, request = {},  image = None):
            if request != {}:
                self.loadFromDict(request)
            if image is not None:
                self.loadImage(image)
            self.RunProcess()

    def RunProcess(self, req = {}):
        """Inference demo for GFPGAN (for users).
        """
        if req != {}:
            self.loadFromDict(req)
        
        input_img = self.inputimage
        # ------------------------ set up background upsampler ------------------------
        if self.bg_upsampler == 'realesrgan' :
            if not torch.cuda.is_available():  # CPU
                print('The unoptimized RealESRGAN is slow on CPU. We do not use it. '
                            'If you really want to use it, please modify the corresponding codes.')
                bg_upsampler = None
            else:
                from basicsr.archs.rrdbnet_arch import RRDBNet
                from realesrgan import RealESRGANer
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
                bg_upsampler = RealESRGANer(
                    scale=2,
                    model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
                    model=model,
                    tile=self.bg_tile,
                    tile_pad=10,
                    pre_pad=0,
                    half=True)  # need to set False in CPU mode
        else:
            bg_upsampler = None
        # ------------------------ set up GFPGAN restorer ------------------------
        if self.version == '1':
            arch = 'original'
            channel_multiplier = 1
            model_name = 'GFPGANv1'
            url = 'https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/GFPGANv1.pth'
        elif self.version == '1.2':
            arch = 'clean'
            channel_multiplier = 2
            model_name = 'GFPGANCleanv1-NoCE-C2'
            url = 'https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth'
        elif self.version == '1.3':
            arch = 'clean'
            channel_multiplier = 2
            model_name = 'GFPGANv1.3'
            url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth'
        elif self.version == '1.4':
            arch = 'clean'
            channel_multiplier = 2
            model_name = 'GFPGANv1.4'
            url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'
        elif self.version == 'RestoreFormer':
            arch = 'RestoreFormer'
            channel_multiplier = 2
            model_name = 'RestoreFormer'
            url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth'
        else:
            print(f'AI Exiting: Wrong model version {self.version}.')
            
            raise ValueError(f'Wrong model version {self.version}.')

        isServer = GetServerMode()
        inputPath = "/root/Models/facefix"
        if not isServer:
            if self.modelPath != "":
                inputPath = os.path.dirname( self.modelPath)
            else:
                inputPath = GetDiffusersCachePath( "/gfpgan/weights") 
            
        # determine model paths
 
        model_path = os.path.join(inputPath, model_name + '.pth')
        if self.modelPath != "":
            model_path =  self.modelPath
        if not os.path.isfile(model_path):
            model_path = os.path.join('gfpgan/weights', model_name + '.pth')
        if not os.path.isfile(model_path):
            # download pre-trained models from url
            model_path = url

        restorer = GFPGANer(
            model_path=model_path,
            upscale=self.upscale,
            arch=arch,
            channel_multiplier=channel_multiplier,
            bg_upsampler=bg_upsampler)
        indx = 0
        # ------------------------ restore ------------------------
        # restore faces and background if necessary
        cropped_faces, restored_faces, restored_img = restorer.enhance(
            input_img,
            has_aligned=self.aligned,
            only_center_face=self.only_center_face,
            paste_back=True,
            weight=self.weight)
        # torch.cuda.empty_cache()
        # return base64Img.decode('utf-8')  
        # cv2.imwrite('cropped_faces.jpg', cropped_faces[0])
        try:
            return PIL.Image.fromarray(cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)) 
        except:
            return PIL.Image.fromarray(self.inputimage) 
        return restored_img  

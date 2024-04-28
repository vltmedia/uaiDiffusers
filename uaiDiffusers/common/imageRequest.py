import os
from typing import List
from typing import Any
from dataclasses import dataclass
from uaiDiffusers.common.utils import SetCLRMaxSteps

import json
@dataclass
class ImageRequest:
    def __init__(self):
        self.ratio = 1
        self.configScale = 0
        self.guidance_scale = 7.5
        self.seed = 52125
        self.prompt = ""
        self.negative_prompt = ""
        self.neg_prompt = ""
        self.num_images_per_prompt = 1
        self.imagesToGenerate = 1
        self.num_frames = 1
        self.size = [1920, 1080]
        self.encodeSize = [1024, 576]
        self.steps = 6
        self.faceFix = 1
        self.width = 1920
        self.height = 1080
        self.maxSize = 1080
        self.resolution = 256
        self.topOffset = 0
        self.altitude = 0
        self.distance = 0
        self.fov = 0
        self.fovx = 0
        self.fovy = 0
        self.range = [0,1]
        self.range2 = [0,1]
        self.topOffset = 0
        self.version = "1.3"
        self.upscale = 2
        self.bg_upsampler = "realesrgan"
        self.chunkSize = 8192
        self.bg_tile = 400
        self.suffix = "None"
        self.only_center_face = False
        self.aligned = False
        self.ext = "auto"
        self.foregroundRatio = 1.0
        self.weight = 0.75
        self.s_scale = 0.75
        self.scale = 0.75
        self.encodedFps = 8
        self.fps = 24
        self.loops = 0
        self.maskIndex = 0
        self.removeBg = False
        self.isXL = False
        self.isXLLightning = False
        self.maskIndex   = False
        self.watermarked   = False
        self.exportGif   = False
        self.exportTextures   = False
        self.exportMeshes   = False
        self.exportFiles   = False
        self.exportOther   = False
        self.export   = False
        self.saveFaceEmbeddings   = False
        self.overrideForm   = "False"
        
        self.submode = "default"
        self.mode = "default"
        self.model = "default"
        self.modelType = "default"
        self.ipAdapterModel = ""
        self.onnxModel = ""
        self.inpaintModel = ""
        self.controlnetModel = ""
        self.vaeRepo = ""
        self.schedueler = ""
        self.device = ""
        self.tempPath = ""
        self.paramsData = ""
        self.exportSize = ""
        self.faceEmbeddings = ""
        self.input = ""
        self.mask = ""
        self.inputs = []
        self.masks = []
        self.mediaPaths = []
        self.urls = []
        self.styleImages = []
        self.vertecies = []
        self.points = []
        self.primitives = []
        self.meshes = []
        self.objects = []
        self.output = ""
        self.customSDBinWeights = ""
        self.textualInversionWeights = ""
        self.refinerRepo = ""
        self.metadata = ""
        self.url = ""
        self.shareuser = ""
        self.user = ""
        self.ipAdapters = []
        self.loras = []
        
    def DetectSizeFromSize(self):
        if isinstance(self.size, str):
            split = self.size.split(",")
            width = int(split[0])
            height = int(split[1])
            self.ratio = float(width) / float(height)
            self.size = [width, height]
            self.width = width
            self.height = height
            
    def DetectEncodeSize(self):
        if isinstance(self.encodeSize, str):
            split = self.encodeSize.split(" ")
            sizeSplit = split[1].lower().split("x")
            width = int(sizeSplit[0])
            height = int(sizeSplit[1])
            self.encodeSize = [width, height]
            
    def DetectSizeFromExportSize(self):
        split = self.exportSize.split(" ")
        ratio = split[0]
        sizeSplit = split[1].lower().split("x")
        width = int(sizeSplit[0])
        height = int(sizeSplit[1])
        self.ratio = float(width) / float(height)
        self.size = [width, height]
        self.width = width
        self.height = height
        
    def DetectXL(self):
        self.isXL = "XL" in self.model or "xl" in self.model
        if self.isXL:
            self.isXLLightning =  "lightning" in self.model or "Lightning" in self.model
        
    def JSON(self):
        return json.dumps(self.__dict__)
    
    def SetOutputExtension(self, extension = "png"):
        root, ext = os.path.splitext(self.output)
        if not ext:
            ext = f'.{extension}'
        self.output = root + ext
        
    def CreateResponse(self, outputFiles = [], objectType = "image", isBase64 = False):
            
        outputDict = {"media":[]}
        for file in outputFiles:
            outputDict["media"].append({"media":file,"prompt": self.prompt,
                "seed": self.seed, "objectType": objectType, "isBase64": isBase64})
        
        outputDict["media"].append({"media":self.input,"prompt": self.prompt,
                "seed": self.seed, "objectType": objectType, "isBase64": True})
        return outputDict

    @staticmethod
    def FromDict(obj: Any) -> 'ImageRequest':
        imgRequest              = ImageRequest()
        try:
            imgRequest.chunkSize = int(obj["chunkSize"])
        except:
            pass
        try:
            imgRequest.steps = int(obj["steps"])
            SetCLRMaxSteps(imgRequest.steps)
        except:
            pass
        try:
            imgRequest.weight =float( obj["weight"])
        except:
            pass
        try:
            imgRequest.ratio = float(obj["ratio"])
        except:
            pass
        try:
            imgRequest.s_scale = float(obj["s_scale"])
        except:
            pass
        try:
            imgRequest.scale = float(obj["scale"])
        except:
            pass
        try:
            imgRequest.configScale = float(obj["configScale"])
            imgRequest.guidance_scale = float(obj["configScale"])
        except:
            pass
        try:
            imgRequest.guidance_scale = float(obj["guidance_scale"])
            imgRequest.configScale = float(obj["guidance_scale"])
            
        except:
            pass
        try:
            imgRequest.seed = int(obj["seed"])
        except:
            pass
        
        try:
            imgRequest.exportTextures = bool(obj["exportTextures"])
        except:
            pass
        
        try:
            imgRequest.exportOther = bool(obj["exportOther"])
        except:
            pass
        
        try:
            imgRequest.export = bool(obj["export"])
        except:
            pass
        
        try:
            imgRequest.exportFiles = bool(obj["exportFiles"])
        except:
            pass
        
        try:
            imgRequest.vertecies = obj["vertecies"]
        except:
            pass
        
        try:
            imgRequest.points = obj["points"]
        except:
            pass
        
        
        try:
            imgRequest.objects = obj["objects"]
        except:
            pass
        
        
        try:
            imgRequest.meshes = obj["meshes"]
        except:
            pass
        
        
        try:
            imgRequest.primitives = obj["primitives"]
        except:
            pass
        
        try:
            imgRequest.exportMeshes = bool(obj["exportMeshes"])
        except:
            pass
        
        try:
            imgRequest.faceEmbeddings = obj["faceEmbeddings"]
        except:
            pass
        try:
            imgRequest.encodeSize = obj["encodeSize"]
            imgRequest.DetectEncodeSize()
            
        except:
            pass
        try:
            imgRequest.removeBg = int(obj["removeBg"])
        except:
            pass
        try:
            imgRequest.ipAdapterModel = obj["ipAdapterModel"]
        except:
            pass
        try:
            imgRequest.topOffset = int(obj["topOffset"])
        except:
            pass
        try:
            imgRequest.foregroundRatio = float(obj["foregroundRatio"])
        except:
            pass
        try:
            imgRequest.resolution = int(obj["resolution"])
        except:
            pass
        try:
            imgRequest.maxSize = int(obj["maxSize"])
        except:
            pass
        try:
            imgRequest.prompt = obj["prompt"]
        except:
            pass
        try:
            imgRequest.negative_prompt = obj["negative_prompt"]
        except:
            pass
        try:
            imgRequest.negative_prompt = obj["neg_prompt"]
            imgRequest.neg_prompt = obj["neg_prompt"]
        except:
            pass
        try:
            imgRequest.negative_prompt = obj["negative_prompt"]
            imgRequest.neg_prompt = obj["negative_prompt"]
            
        except:
            pass
        try:
            imgRequest.num_images_per_prompt = int(obj["num_images_per_prompt"])
            imgRequest.imagesToGenerate = int(obj["num_images_per_prompt"])
        except:
            pass
        try:
            imgRequest.num_images_per_prompt = int(obj["imagesToGenerate"])
            imgRequest.imagesToGenerate = int(obj["imagesToGenerate"])
            imgRequest.num_frames = int(obj["imagesToGenerate"])
        except:
            pass
        try:
            imgRequest.num_images_per_prompt = int(obj["num_frames"])
            imgRequest.imagesToGenerate = int(obj["num_frames"])
            imgRequest.num_frames = int(obj["num_frames"])
        except:
            pass
        try:
            imgRequest.size = obj["size"]
        except:
            pass
        try:
            imgRequest.faceFix = int(obj["faceFix"])
        except:
            pass
        try:
            imgRequest.version = str(obj["version"])
        except:
            pass
        try:
            imgRequest.weight =float( obj["facefixweight"])
        except:
            pass
        try:
            imgRequest.upscale =int( obj["facefixupscale"])
        except:
            pass
        try:
            imgRequest.upscale = int(obj["upscale"])
        except:
            pass
        try:
            imgRequest.bg_upsampler = str(obj["bg_upsampler"])
        except:
            pass
        try:
            imgRequest.bg_tile = int(obj["bg_tile"])
        except:
            pass
        try:
            imgRequest.loops = int(obj["loops"])
        except:
            pass
        try:
            imgRequest.suffix = int(obj["suffix"])
        except:
            pass
        try:
            imgRequest.only_center_face = bool(obj["only_center_face"])
        except:
            pass
        try:
            imgRequest.exportGif =bool( obj["exportGif"])
        except:
            pass
        try:
            imgRequest.saveFaceEmbeddings =bool( obj["saveFaceEmbeddings"])
        except:
            pass
        try:
            imgRequest.aligned =bool( obj["aligned"])
        except:
            pass
        try:
            imgRequest.ext = str(obj["ext"])
        
        except:
            pass
        try:
            imgRequest.submode = obj["submode"]
        except:
            pass
        try:
            imgRequest.mode = obj["mode"]
        except:
            pass
        try:
            imgRequest.model = obj["model"]
        except:
            pass
        try:
            imgRequest.modelType = obj["modelType"]
        except:
            pass
        try:
            imgRequest.onnxModel = obj["onnxModel"]
        except:
            pass
        try:
            imgRequest.inpaintModel = obj["inpaintModel"]
        except:
            pass
        try:
            imgRequest.controlnetModel = obj["controlnetModel"]
        except:
            pass
        try:
            imgRequest.vaeRepo = obj["vaeRepo"]
        except:
            pass
        
        
        try:
            imgRequest.altitude = float(obj["altitude"])
        except:
            pass
        
        try:
            imgRequest.distance = float(obj["distance"])
        except:
            pass
        
        try:
            imgRequest.fov = float(obj["fov"])
        except:
            pass
        try:
            imgRequest.fovc = float(obj["fovc"])
        except:
            pass
        try:
            imgRequest.fovy = float(obj["fovy"])
        except:
            pass
        
        try:
            imgRequest.range = obj["range"]
        except:
            pass
        
        try:
            imgRequest.range2 = obj["range2"]
        except:
            pass
        
        try:
            imgRequest.schedueler = obj["schedueler"]
        except:
            pass
        try:
            imgRequest.device = obj["device"]
        except:
            pass
        try:
            imgRequest.tempPath = obj["tempPath"]
        except:
            pass
        try:
            imgRequest.paramsData = obj["paramsData"]
        except:
            pass
        try:
            imgRequest.exportSize = obj["exportSize"]
        except:
            pass
        try:
            imgRequest.input = obj["input"]
        except:
            pass
        try:
            import flask
            import base64
            inputImage = flask.request.files['inputImage']
            image_string = base64.b64encode(inputImage.read())
            imgRequest.input = image_string
        except:
            pass
        try:
            import flask
            import base64
            inputImage = flask.request.files['input']
            image_string = base64.b64encode(inputImage.read())
            imgRequest.input = image_string
        except:
            pass
        try:
            import flask
            import base64
            inputs = flask.request.files['inputs']
            for inputImage in inputs:
                image_string = base64.b64encode(inputImage.read())
                imgRequest.inputs.append(image_string)
        except:
            pass
        try:
            import flask
            import base64
            inputs = flask.request.files['masks']
            for inputImage in inputs:
                image_string = base64.b64encode(inputImage.read())
                imgRequest.masks.append(image_string)
        except:
            pass
        try:
            import flask
            import base64
            inputs = flask.request.files['styleImages']
            for inputImage in inputs:
                image_string = base64.b64encode(inputImage.read())
                imgRequest.styleImages.append(image_string)
        except:
            pass
        try:
            import flask
            import pickle
            import base64
            inputs = flask.request.files['faceEmbeddings']
            for inputImage in inputs:
                loaded = pickle.loads(inputImage.read())
                imgRequest.faceEmbeddings.append(loaded)
        except:
            pass
        try:
            imgRequest.mask = obj["mask"]
        except:
            pass
        try:
            imgRequest.masks = obj["masks"]
        except:
            pass
        try:
            imgRequest.mediaPaths = obj["mediaPaths"]
        except:
            pass
        try:
            imgRequest.urls = obj["urls"]
        except:
            pass
        try:
            imgRequest.styleImages = obj["styleImages"]
        except:
            pass
        try:
            imgRequest.output = obj["output"]
        except:
            pass
        try:
            imgRequest.customSDBinWeights = obj["customSDBinWeights"]
        except:
            pass
        try:
            imgRequest.textualInversionWeights = obj["textualInversionWeights"]
        except:
            pass
        try:
            imgRequest.refinerRepo = obj["refinerRepo"]
        except:
            pass
        try:
            imgRequest.metadata = obj["metadata"]
        except:
            pass
        try:
            imgRequest.url = obj["url"]
        except:
            pass
        try:
            imgRequest.shareuser = obj["shareuser"]
        except:
            pass
        try:
            imgRequest.user = obj["user"]
        except:
            pass
        try:
            imgRequest.ipAdapters = obj["ipAdapters"]
        except:
            pass
        try:
            imgRequest.loras = obj["loras"]
        except:
            pass
        try:
            imgRequest.maskIndex = int(obj["maskIndex"])
        except:
            pass
        try:
            imgRequest.watermarked = bool(obj["watermarked"])
        except:
            pass
        try:
            imgRequest.overrideForm = obj["overrideForm"]
        except:
            pass
        try:
            imgRequest.isXL = bool(obj["isXL"])
        except:
            pass
        try:
            imgRequest.isXLLightning = bool(obj["isXLLightning"])
        except:
            pass
        try:
            imgRequest.width = int(obj["width"])
        except:
            pass
        try:
            imgRequest.height = int(obj["height"])
        except:
            pass
        try:
            imgRequest.fps = float(obj["fps"])
        
        except:
            pass
        try:
            imgRequest.encodedFps = float(obj["encodedFps"])
        
        except:
            pass
        if imgRequest.isXL == False:
            imgRequest.DetectXL()
        return imgRequest

# Example Usage
# jsonstring = json.loads(myjsonstring)
# root = Root.from_dict(jsonstring)


from PIL import Image, ImageFilter
import PIL
def PruneList(inputList = [], maxImages = 100):
    import glob
    from random import shuffle
    shuffle(inputList)
    inputList = inputList[:len(inputList) - maxImages]

    return inputList


def PruneFilesInFolder(inputDirectory = "johnSmith/johnSmithDataset", extension="png", maxImages = 100):
    import os
    import glob
    files = glob.glob(inputDirectory + "/*."+ extension)
    files = PruneList(files, maxImages)
    for file in files:
        os.remove(file)
    return files
    
def ConvertDirectoryRGBAToRGB(inputDirectory = "johnSmith/johnSmithDataset", extension="png", replacementColor = (0,0,0,255)):
    import os
    import glob
    files = glob.glob(inputDirectory + "/*."+ extension)
    for indx, file in enumerate(files):
        print(f"Processing image {indx+1} / {len(files)}")
        img = Image.open(file)
        img = img.convert("RGB")
        img.save(file)
    return files
    

def ForceResizDirectoryImages(inputPath,  outputPath,extension = "png" , size = (256, 256)):
    import cv2
    import os
    import glob
    import numpy as np
    from PIL import Image, ImageOps
    imgs = glob.glob(inputPath + "/*."+extension)
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
    for indx, img in enumerate(imgs):
        print(f"Processing image {indx+1} / {len(imgs)}")
        newImage = ForceResizeImage(img, size)
        newImage.save(os.path.join(outputPath, os.path.basename(img)))
        
    

def ForceResizeImage(imgPath, size = (256, 256)):
    from PIL import Image, ImageOps
    with Image.open(imgPath) as im:
        im = ImageOps.pad(im, (size[0], size[1]), color='black')
    return im
    
def CreateImageDatasetFromVideo(inputVideo, outputPath, prefix="img" ,extension="png", maxImages=100):
    import cv2
    import os
    import glob
    import numpy as np
    import random
    vid = cv2.VideoCapture(inputVideo)
    success,image = vid.read()
    count = 0
    videoLength = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    keyFrames = np.linspace(0, videoLength - 1, maxImages, dtype=int)
    while success:
        success,image = vid.read()
        if count in keyFrames:
            cv2.imwrite(os.path.join(outputPath, f"{prefix}_{count:04d}.{extension}"), image)
        count += 1
            

import base64
import io
import os
import glob
import sys
import numpy as np
from tokenizers import Tokenizer, pre_tokenizers
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.pre_tokenizers import Digits
from PIL import Image
import cv2
# Take in base64 string and return cv image
def stringToRGB(base64_string, size = (512, 512)):
    # convert string to bytes
    message_bytes = base64_string.encode('ascii')
    imgdata = base64.b64encode(message_bytes)
    # convert bytes to numpy array at the size of size
    img = np.frombuffer(imgdata, dtype=np.uint8)
    # resize numpy array to size x size with 3 channels and padding if needed
    img = np.resize(img, size + (3,))
    # convert numpy array to cv2 image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # return cv2 image
    vv = 0
    return img
    
    




def readObj(inputPath,skipPattern = "#"):
    text = open(inputPath, "r")
    lines = text.readlines()
    text.close()
    # remove any lines that start with skipPattern
    lines = [line for line in lines if not line.startswith(skipPattern)]
    combinedLines = "".join(lines)
    return combinedLines


def tokenizeToImage(inputPath,skipPattern = "#", size = (512, 512)):
    textLines = readObj(inputPath, skipPattern)
    v= 0
    img = stringToRGB(textLines, size)
    return img



def decodeReadImageToString(inputPath, size = (512, 512)):
    img = cv2.imread(inputPath)
    # reverse everything above to get back to string
    imgR = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgR = np.resize(imgR, (1, size[0]*size[1]*3))
    imgR = imgR.reshape(size[0]*size[1]*3)
    imgR = imgR.astype(np.uint8)
    imgR = base64.b64decode(imgR)
    imgR = imgR.decode(encoding="utf-8", errors='ignore')
    return imgR
    

def decodeImageToString(cv2img, size = (512, 512)):
    img = cv2img
    # reverse everything above to get back to string
    imgR = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgR = np.resize(imgR, (1, size[0]*size[1]*3))
    imgR = imgR.reshape(size[0]*size[1]*3)
    imgR = imgR.astype(np.uint8)
    imgR = base64.b64decode(imgR)
    imgR = imgR.decode(encoding="utf-8", errors='ignore')
    return imgR
    
    

    
    
def objToImage(inputPath, outputPath,skipPattern = "#" ,size = (512, 512)):
    img = tokenizeToImage(inputPath,skipPattern, size)
    cv2.imwrite(outputPath, img)

def encodeDirectoryOfTextToImage(inputDir, outputDir):
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    files = glob.glob(inputDir + "/*.obj")
    for indx, file in enumerate(files):
        inputPath = file
        outputPath = outputDir + "/" + os.path.basename(inputPath)[:-4] + ".png"
        print(f"Processing {indx} of {len(files)} files")
        objToImage(inputPath, outputPath,skipPattern = "#", size = (512, 512))
    
def decodeDirectoryOfTextToImage(inputDir, outputDir):
    
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    files = glob.glob(inputDir + "/*.png")
    for indx, file in enumerate(files):
        inputPath = file
        outputPath = outputDir + "/" + os.path.basename(inputPath)[:-4] + ".obj"
        exportImageToString(inputPath, outputPath)
    
    
    
def exportImageToString(inputPath, outputPath):
    decodedString = decodeReadImageToString(inputPath)
    # decodedString = decodeImageToString(inputPath)
    text = open(outputPath, "w")
    text.write(decodedString)
    text.close()
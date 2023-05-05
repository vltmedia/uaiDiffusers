
from PIL import Image, ImageFilter
import PIL
def PruneList(inputList = [], maxImages = 100):
    import glob
    from random import shuffle
    shuffle(inputList)
    inputList = inputList[:len(inputList) - maxImages]

    return inputList


def PruneFilesInFolder(inputDirectory = "johnSmith\johnSmithDataset", extension="png", maxImages = 100):
    import os
    import glob
    files = glob.glob(inputDirectory + "/*."+ extension)
    files = PruneList(files, maxImages)
    for file in files:
        os.remove(file)
    return files
    
def ConvertDirectoryRGBAToRGB(inputDirectory = "johnSmith\johnSmithDataset", extension="png", replacementColor = (0,0,0,255)):
    import os
    import glob
    files = glob.glob(inputDirectory + "/*."+ extension)
    for indx, file in enumerate(files):
        print(f"Processing image {indx+1} / {len(files)}")
        img = Image.open(file)
        img = img.convert("RGB")
        img.save(file)
    return files
    
    
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
            

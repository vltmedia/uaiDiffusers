from glob import glob
import shutil
import torch
from time import  strftime
import os, sys, time
from argparse import ArgumentParser
import numpy as np

np.float = float  
np.complex = complex  
from sadTalksrc.utils.preprocess import CropAndExtract
from sadTalksrc.test_audio2coeff import Audio2Coeff  
from sadTalksrc.facerender.animate import AnimateFromCoeff
from sadTalksrc.generate_batch import get_data
from sadTalksrc.generate_facerender_batch import get_facerender_data
from sadTalksrc.utils.init_path import init_path


# Make me a class based on the parameters in the parser above
class sadTalkArgs:
    def __init__(self):
        self.driven_audio =''
        self.source_image =''
        self.outputPath =''
        self.ref_eyeblink = ''
        self.ref_pose = ''
        self.checkpoint_dir = ''
        self.result_dir = ''
        self.pose_style = 0
        self.batch_size = 2
        self.size = 256
        self.expression_scale = 1.0
        self.input_yaw = 0
        self.input_pitch = 0
        self.input_roll = 0
        self.enhancer = False
        self.background_enhancer = False
        self.cpu = False
        self.face3dvis = False
        self.still = False
        self.device = 'cuda'
        self.preprocess = 'crop'
        self.verbose = False
        self.old_version = False
        self.net_recon = 'resnet50'
        self.init_path = 'None'
        self.use_last_fc = False
        self.bfm_folder = './checkpoints/BFM_Fitting/'
        self.bfm_model = 'BFM_model_front.mat'
        self.focal = 1015.0
        self.center = 112.0
        self.camera_d = 10.0
        self.z_near = 5.0
        self.z_far = 15.0
    
    def fromJson(self, jsonData):
        """
        Load data from a json object
        Example:
            >>> args = sadTalkArgs()
            >>> args.fromJson(jsonData)
        Args:
            jsonData (dict): A dictionary containing the data to load
        Returns:
            None
        
        """
        self.driven_audio = jsonData['driven_audio']
        self.source_image = jsonData['source_image']
        self.outputPath = jsonData['outputPath']
        self.ref_eyeblink = jsonData['ref_eyeblink']
        self.ref_pose = jsonData['ref_pose']
        self.checkpoint_dir = jsonData['checkpoint_dir']
        self.result_dir = jsonData['result_dir']
        self.pose_style = jsonData['pose_style']
        self.batch_size = jsonData['batch_size']
        self.size = jsonData['size']
        self.expression_scale = jsonData['expression_scale']
        self.input_yaw = jsonData['input_yaw']
        self.input_pitch = jsonData['input_pitch']
        self.input_roll = jsonData['input_roll']
        self.enhancer = jsonData['enhancer']
        self.background_enhancer = jsonData['background_enhancer']
        self.cpu = jsonData['cpu']
        self.face3dvis = jsonData['face3dvis']
        self.still = jsonData['still']
        self.device = jsonData['device']
        self.preprocess = jsonData['preprocess']
        self.verbose = jsonData['verbose']
        self.old_version = jsonData['old_version']
        self.net_recon = jsonData['net_recon']
        self.init_path = jsonData['init_path']
        self.use_last_fc = jsonData['use_last_fc']
        self.bfm_folder = jsonData['bfm_folder']
        self.bfm_model = jsonData['bfm_model']
        self.focal = jsonData['focal']
        self.center = jsonData['center']
        self.camera_d = jsonData['camera_d']
        self.z_near = jsonData['z_near']
        self.z_far = jsonData['z_far']
        



def AnimateFaceWithAudio(args):
    #torch.backends.cudnn.enabled = False

    pic_path = args.source_image
    audio_path = args.driven_audio
    save_dir = os.path.join(os.path.dirname( args.outputPath), strftime("%Y_%m_%d_%H.%M.%S"))
    os.makedirs(save_dir, exist_ok=True)
    pose_style = args.pose_style
    device = args.device
    batch_size = args.batch_size
    input_yaw_list = args.input_yaw if args.input_yaw != [] else None
    input_pitch_list = args.input_pitch if args.input_pitch != [] else None
    input_roll_list = args.input_roll if args.input_roll != [] else None
    ref_eyeblink = args.ref_eyeblink
    ref_pose = args.ref_pose

    current_root_path = os.path.split(sys.argv[0])[0]

    sadtalker_paths = init_path(args.checkpoint_dir, os.path.join(current_root_path, 'src/config'), args.size, args.old_version, args.preprocess)

    #init model
    preprocess_model = CropAndExtract(sadtalker_paths, device)

    audio_to_coeff = Audio2Coeff(sadtalker_paths,  device)
    
    animate_from_coeff = AnimateFromCoeff(sadtalker_paths, device)

    #crop image and extract 3dmm from image
    first_frame_dir = os.path.join(save_dir, 'first_frame_dir')
    os.makedirs(first_frame_dir, exist_ok=True)
    print('3DMM Extraction for source image')
    first_coeff_path, crop_pic_path, crop_info =  preprocess_model.generate(pic_path, first_frame_dir, args.preprocess, source_image_flag=True, pic_size=args.size)
    if first_coeff_path is None:
        print("Can't get the coeffs of the input")
        return

    if ref_eyeblink !=  '':
        ref_eyeblink_videoname = os.path.splitext(os.path.split(ref_eyeblink)[-1])[0]
        ref_eyeblink_frame_dir = os.path.join(save_dir, ref_eyeblink_videoname)
        os.makedirs(ref_eyeblink_frame_dir, exist_ok=True)
        print('3DMM Extraction for the reference video providing eye blinking')
        ref_eyeblink_coeff_path, _, _ =  preprocess_model.generate(ref_eyeblink, ref_eyeblink_frame_dir, args.preprocess, source_image_flag=False)
    else:
        ref_eyeblink_coeff_path=None

    if ref_pose != '':
        if ref_pose == ref_eyeblink: 
            ref_pose_coeff_path = ref_eyeblink_coeff_path
        else:
            ref_pose_videoname = os.path.splitext(os.path.split(ref_pose)[-1])[0]
            ref_pose_frame_dir = os.path.join(save_dir, ref_pose_videoname)
            os.makedirs(ref_pose_frame_dir, exist_ok=True)
            print('3DMM Extraction for the reference video providing pose')
            ref_pose_coeff_path, _, _ =  preprocess_model.generate(ref_pose, ref_pose_frame_dir, args.preprocess, source_image_flag=False)
    else:
        ref_pose_coeff_path=None

    #audio2ceoff
    batch = get_data(first_coeff_path, audio_path, device, ref_eyeblink_coeff_path, still=args.still)
    coeff_path = audio_to_coeff.generate(batch, save_dir, pose_style, ref_pose_coeff_path)

    # 3dface render
    if args.face3dvis:
        from sadTalksrc.face3d.visualize import gen_composed_video
        gen_composed_video(args, device, first_coeff_path, coeff_path, audio_path, os.path.join(save_dir, '3dface.mp4'))
    
    #coeff2video
    data = get_facerender_data(coeff_path, crop_pic_path, first_coeff_path, audio_path, 
                                batch_size, input_yaw_list, input_pitch_list, input_roll_list,
                                expression_scale=args.expression_scale, still_mode=args.still, preprocess=args.preprocess, size=args.size)
    
    result = animate_from_coeff.generate(data, save_dir, pic_path, crop_info, \
                                enhancer=args.enhancer, background_enhancer=args.background_enhancer, preprocess=args.preprocess, img_size=args.size)
    
    shutil.move(result,  args.outputPath)
    print('The generated video is named:',  args.outputPath)

    if not args.verbose:
        shutil.rmtree(save_dir)
        

def main():
    args = sadTalkArgs()
    args.driven_audio = "P:/datasets/Voices/voices/angie/2.wav"
    args.source_image = "F:/Temp/AI/image_0000.png"
    args.checkpoint_dir = "F:/Temp/AI/checkpoints"
    args.result_dir = 'F:/Temp/AI/results'
    args.outputPath = 'F:/Temp/AI/results/img2.mp4'
    args.batch_size = 2
    args.size = 512
    args.expression_scale = 1.0
    args.enhancer = False
    args.background_enhancer = False
    args.cpu = False
    args.face3dvis = False
    args.still = False
    args.preprocess = 'crop'
    args.device = 'cuda'
    args.focal = 1015.0
    args.center = 112.0
    args.camera_d = 10.0
    args.z_near = 5.0
    args.z_far = 15.0
    args.z_far = 15.0
    args.input_yaw = []
    args.input_pitch = []
    args.input_roll = []
    args.ref_eyeblink = ''
    args.ref_pose = ''
    import json
    file = open("outtext.json", "w")
    file.write(json.dumps(args.__dict__))
    AnimateFaceWithAudio(args)
    
main()
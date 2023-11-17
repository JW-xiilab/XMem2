import os
import re
import argparse
os.environ["CUDA_VISIBLE_DEVICES"]='4'
from pathlib import Path

from util import mask_img_loader
from sam_hq.sam_controller import SamController
from inference.run_on_video import run_on_video, _inference_on_video
from inference.run_on_video_kjw import run_on_video as run_on_video2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process video frames given a few (1+) existing annotation masks')
    parser.add_argument('--video', default='/DATA_17/kjw/01-SM_project/input_vid/jenny_01.mp4', type=str, help='Path to the video file or directory with .jpg video frames to process')
    parser.add_argument('--masks', default='/DATA_17/kjw/01-SM_project/XMem2/workspace/masks/jenny_01/frame_000000.png', type=str, help='Path to the directory with individual .png masks  for corresponding video frames, named `frame_000000.png`, `frame_000123.png`, ... or similarly (the script searches for the first integer value in the filename).'
                        'Will use all masks int the directory.')
    parser.add_argument('-vis','--visualize', action='store_true', help='Either Save video file from overlayed images')
    parser.add_argument('--vid_save', action='store_true', help='Either Save video file from overlayed images')
    parser.add_argument('--output', default='workspace', type=str, help='Path to the output directory where to save the resulting segmentation masks and overlays. '
                        'Will be automatically created if does not exist')

    args = parser.parse_args()

    sam = SamController('sam_hq/sam_hq_vit_h.pth', 'vit_h') 
    
    # frames_with_masks = []
    # for file_path in (p for p in Path(args.masks).iterdir() if p.is_file()):
    #     frame_number_match = re.search(r'\d+', file_path.stem)
    #     if frame_number_match is None:
    #         print(f"ERROR: file {file_path} does not contain a frame number. Cannot load it as a mask.")
    #         exit(1)
    #     frames_with_masks.append(int(frame_number_match.group()))

    # print("Using masks for frames: ", frames_with_masks)
    mask_img = mask_img_loader(args.mask)
    mask, _, p_img = sam.first_frame_click(image=mask_img,
                                           points=points,
                                           lables=labels,
                                           box=box,
                                           multimask=multimask)
    
    p_out = Path(args.output)
    p_out.mkdir(parents=True, exist_ok=True)
    run_on_video2(args.video, args.masks, args.output, frames_with_masks, save_overlay=args.visualize, vid_save=args.vid_save)

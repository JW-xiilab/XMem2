import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Union
from PIL import Image, ImageDraw, ImageOps

from .hq_segmenter import BaseSegmenter
from .painter import mask_painter, point_painter
# from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
# from .mask_painter import mask_painter as mask_painter2

mask_color = 3
mask_alpha = 0.7
contour_color = 1
contour_width = 5
point_color_ne = 8
point_color_ps = 50
point_alpha = 0.9
point_radius = 15
contour_color = 2
contour_width = 5

class SamController():
    def __init__(self, SAM_checkpoint, model_type):
        '''
        initialize sam controler
        '''
        self.sam_controler = BaseSegmenter(SAM_checkpoint, model_type)

    def first_frame_click(self, image:np.ndarray, points:np.ndarray=None, labels:np.ndarray=None, box:np.ndarray=None, multimask=True,mask_color=3):
        '''
        it is used in first frame in video
        return: mask, logit, painted image(mask+point)
        '''
        self.sam_controler.set_image(image)
        origal_image = self.sam_controler.orignal_image
        if isinstance(labels, np.ndarray):
            neg_flag = labels[-1]
        if isinstance(box, np.ndarray):
            prompts = {'box_coords': box}
            masks, scores, logits = self.sam_controler.predict(prompts, 'box', multimask=False)  # masks (n, h, w), scores (n,), logits (n, 256, 256)
            mask, logit = masks[np.argmax(scores)], logits[np.argmax(scores), :, :]
        else:
            if neg_flag==1:
                #find neg
                prompts = {
                    'point_coords': points,
                    'point_labels': labels,
                }
                masks, scores, logits = self.sam_controler.predict(prompts, 'point', multimask)
                mask, logit = masks[np.argmax(scores)], logits[np.argmax(scores), :, :]
                prompts = {
                    'point_coords': points,
                    'point_labels': labels,
                    'mask_input': logit[None, :, :]
                }
                masks, scores, logits = self.sam_controler.predict(prompts, 'both', multimask)
                mask, logit = masks[np.argmax(scores)], logits[np.argmax(scores), :, :]
            else:
                #find positive
                prompts = {
                    'point_coords': points,
                    'point_labels': labels,
                }
                masks, scores, logits = self.sam_controler.predict(prompts, 'point', multimask)
                mask, logit = masks[np.argmax(scores)], logits[np.argmax(scores), :, :]
                
            
            assert len(points)==len(labels)
        
        painted_image = mask_painter(image, mask.astype('uint8'), mask_color, mask_alpha, contour_color, contour_width)
        if isinstance(box, np.ndarray):
            return
        else:
            painted_image = point_painter(painted_image, np.squeeze(points[np.argwhere(labels>0)],axis = 1), point_color_ne, point_alpha, point_radius, contour_color, contour_width)
            painted_image = point_painter(painted_image, np.squeeze(points[np.argwhere(labels<1)],axis = 1), point_color_ps, point_alpha, point_radius, contour_color, contour_width)
        painted_image = Image.fromarray(painted_image)
        
        return mask, logit, painted_image
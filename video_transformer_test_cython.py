'''
This script contains the codes that we recently developed, and is used to compile a Cython code.

Steps:
1. Copy this file to a new .pyx file.
2. Compile the code by running python3 setup.py build_ext --inplace.
3. Then run test_cython.py.

'''
import torch
import os
from itertools import product
import numpy as np
import face_recognition
import cv2
import matplotlib.patches as patches
from IPython.display import clear_output
from matplotlib.pyplot import imshow
import matplotlib.pylab as plt
from PIL import Image, ImageDraw
import imageio
from itertools import product
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import mmcv
import numpy as np
import pandas as pd
import os
from time import time
from array_mp4_conversion import array_to_mp4, mp4_to_array
from numba import jit, prange
from collections import defaultdict
# cimport numpy as np
# ctypedef np.uint8_t D_TYPE

path = os.getcwd()

class video_transformer_base:
    '''
    This is the base of video_transformer, containing basic information about the video.
    '''
    def __init__(self,
                path, 
                save_path, 
                file_name, 
                device='cpu',
                display=False):
        
        self.video_path = os.path.join(path, "data", file_name)
        self.video_array, self.fps = mp4_to_array(self.video_path)
        self.display = display
        self.save_path = save_path
        self.file_name = file_name
        self.num_frames = 0
        if device == 'cpu':
          self.device = 'cpu'
        else:
          self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    def main_transformation(self, 
                            face_detection_model, 
                            filter_effect):
        '''
        For each frame, do:
        1. detect the face;
        2. apply the filter;
        3. save the processed frame.
        '''                    
        video_capture = cv2.VideoCapture(self.video_path)
        frame_count = 0
        output_frames = []
        while video_capture.isOpened():    
            # Grab a single frame of video
            ret, frame = video_capture.read()
            # try:
            #   frame = torch.from_numpy(frame).to(self.device)
            # except:
            #   print(ret)

            # Bail out when the video file ends
            if not ret:
                video_capture.release()
                break
            
            frame_count += 1
            # print(frame_count)
            # print(type(frame))
            # detect faces
            if face_detection_model != "mtcnn":
                face_locations = self.face_detection(frame, 
                                                     model=face_detection_model)
            else:
                face_locations = self.face_detection_mtcnn(frame)
            # print(f"{len(face_locations)} face(s) detected at frame {frame_count}.")

            # add effect
            after_effect_frame = filter_effect(frame, face_locations)

            # print(frame_count)
            if self.display and frame_count % 2 == 0:
               # If faces were found, we will mark it on frame with blue dots
                for face_location in face_locations:
                    top, right, bottom, left = face_location
                    cv2.rectangle(after_effect_frame,(left, top), (right, bottom), (0, 0, 255), 2)
                plt.imshow(after_effect_frame)
                plt.show()
                clear_output(wait=True)
            # im = Image.fromarray(after_effect_frame)
            # im.save(os.path.join(self.save_path, f"{self.file_name}_prcs_{frame_count}.png"))
            output_frames.append(after_effect_frame)
        self.num_frames = frame_count
        self.des_arr = np.array(output_frames)

    def face_detection(self, frame):
        '''
        Face detection with package face_recognition.
        Models includes: svm, knn, cnn.
        Currently fixed as model='svm' because model ='cnn' is slow.
        '''
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame = frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_frame, model='svm')
        # print(f"{len(face_locations)} face(s) detected.")
        
        return face_locations
    def face_detection_mtcnn(self, frame):
        '''
        Face detection with package facenet_pytorch.
        MTCNN implemented in Pytorch, so also support CUDA.
        '''       

        mtcnn = MTCNN(keep_all=True, device=self.device)
        boxes, _ = mtcnn.detect(frame)
        
        if boxes is None:
            boxes = []
            
        boxes = np.array([[box[1], box[2], box[3], box[0]] for box in boxes]).astype(np.int)
        # print(f"{len(boxes)} face(s) detected.")
        return boxes
    def oil_effect(self, frame):
        pass
    
    def negative_effect(self, frame, locations):
        des_img = np.copy(frame)
        try:
            for location in locations:
                x_, y_, w_, h_ = location
                t_ = int(y_)
                r_ = int(x_+w_)
                b_ = int(y_+h_)
                l_ = int(x_)

                des_img[t_:b_,l_:r_] = 255 - frame[t_:b_,l_:r_]
        except:
            pass
        
        return des_img

    def mean_blur(self, frame, locations, radius=5):
        '''
        Apply simple mosaic effect to specified regions. 
        '''
        k = 1 / (radius*2+1)**2
        des_img = np.copy(frame)
        height, width, _ = des_img.shape
        # try:
        for location in locations:
            top, right, bottom, left = location
            t_ = max(top+radius,0)
            b_ = min(bottom-radius, height)
            l_ = max(left+radius,0)
            r_ = min(right-radius, width)
            if t_ >= b_ or l_ >= r_:
                continue

            for i, j in product(range(t_, b_), range(l_, r_)):
                kernel = frame[i-radius:i+radius+1, j-radius:j+radius+1, :]
                sumed = np.sum(kernel, axis = (0,1)) * k
                des_img[i, j] = sumed.astype(np.uint8)
        # except:
        #     pass
        
        return des_img    
    
    # construct transformed gif
    def output(self):
        images = []
        frames_count = list(range(1,self.num_frames))
        
        for i in frames_count:
            try:
                images.append(imageio.imread(
                    os.path.join(self.save_path, f"{self.file_name}_prcs_{i}.png")))
            except:
                pass
        imageio.mimsave(os.path.join(self.save_path, f"{self.file_name}_prcs.gif"), images)
    
    def write_to_video(self, output_filename):
        '''
        Write out the video with filter to mp4.
        '''
        array_to_mp4(output_filename, self.des_arr, self.fps)

class video_transformer_parallel(video_transformer_base):
    '''
    This version views the video as an array for easier parallelization.
    '''
    def __init__(self, path, save_path, file_name, device='cpu',display=False):
        video_transformer_base.__init__(self, path, save_path, file_name, device, display)
        
        self.locations = None
        self.des_arr = None

        torch.from_numpy(self.video_array).to(self.device)
    
    #@jit(nopython=False, fastmath=True)
    def mean_blur(self, image, des_img, locations, radius):
        '''
        mean_blur function with a source and destination image, the logic remains the same.
        '''
        # radius has to be even
        if len(locations) == 0:
            return
        k = 1 / (radius*2+1)**2
        height, width, _ = des_img.shape
        for location in locations:
            top, right, bottom, left = location
            t_ = max(top+radius,0)
            b_ = min(bottom-radius, height)
            l_ = max(left+radius,0)
            r_ = min(right-radius, width)
            if t_ >= b_ or l_ >= r_:
                continue
            for i in range(t_, b_):
                for j in range(l_, r_):
                    kernel = image[i-radius:i+radius+1, j-radius:j+radius+1, :]
                    sumed = np.sum(kernel, axis= 0, dtype=np.uint32)
                    sumed = np.sum(sumed, axis=0)
                    des_img[i, j, :] = (sumed * k).astype(np.uint8)
    #@jit(nopython=False, parallel=True)
    def get_face_locations(self, face_detection_model):
        '''
        get face_locations on entire video as an array.
        '''
        des_arr = torch.from_numpy(self.video_array.copy()).to(self.device)
        
        if face_detection_model != 'mtcnn':
            locations = list(map(self.face_detection, des_arr))
        else:
            locations = list(map(self.face_detection_mtcnn, des_arr))    

        return locations
    #@jit(nopython=False, parallel=True)
    def filter_on_video(self, filter_func, face_detection_model = 'mtcnn', radius = 15):
        '''
        Produce filter on the video.
        '''
        self.des_arr = self.video_array.copy()
        frame_size = self.video_array.shape[0]
        self.locations = self.get_face_locations(face_detection_model)
        for i in prange(frame_size):
            filter_func(self.video_array[i], self.des_arr[i], self.locations[i], radius)

      
import cv2
import numpy as np
import os  
import glob 
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from torch.utils.data import Dataset
from random import random, choice
from io import BytesIO
from PIL import Image
from PIL import ImageFile
from scipy.ndimage.filters import gaussian_filter
from copy import deepcopy

def extract_phase_spectrum(rgb_array):
    gray = 0.299 * rgb_array[:, :, 0] + 0.587 * rgb_array[:, :, 1] + 0.114 * rgb_array[:, :, 2]
    
    fft_result = torch.fft.fft2(torch.tensor(gray).float())
    fft_shifted = torch.fft.fftshift(fft_result)
    
    phase = torch.angle(fft_shifted)  
    phase_normalized = phase / np.pi  
    return phase_normalized.numpy()

class CMPDataset(Dataset):

    def __init__(self, opt):
        self.opt = opt
        self.data = self.__load_data__()
        self.rz_dict = {'bilinear': Image.BILINEAR,
            'bicubic': Image.BICUBIC,
            'lanczos': Image.LANCZOS,
            'nearest': Image.NEAREST}
        self.jpeg_dict = {'cv2': self.cv2_jpg, 'pil': self.pil_jpg}
    
    def __getitem__(self, index):
        aug_input, no_aug_input, cmp_phase, orig_phase, tf_label, cmp_label = self.preprocess(self.data[index]) 
        return aug_input, no_aug_input, cmp_phase, orig_phase, tf_label, cmp_label
    
    def __len__(self):
        return len(self.data)
    
    def __load_data__(self):
        dataset = []
        tf = None
        for cls in self.opt['classes']:
            root = os.path.join(self.opt['dataroot'], cls)
            if '0_real' in root or '1_fake' in root:
                for root, dirs, files in os.walk(root): 
                    if '0_real' in root.split('/')[-1]:
                        tf = 0
                    elif '1_fake' in root.split('/')[-1]:
                        tf = 1
                    for file in files:  
                        file_path = os.path.join(root, file)  
                        dataset.append([file_path, tf])  
            else:
                dataroot = root
                for item in os.listdir(dataroot):
                    root = os.path.join(dataroot, item)
                    for root, dirs, files in os.walk(root): 
                        if '0_real' in root.split('/')[-1]:
                            tf = 0
                        elif '1_fake' in root.split('/')[-1]:
                            tf = 1
                        for file in files:  
                            file_path = os.path.join(root, file)  
                            dataset.append([file_path, tf])  
        return dataset
      
    def data_augment(self,img):
        img = np.array(img)

        if random() < self.opt['blur_prob']:
            sig = self.sample_continuous(self.opt['blur_sig'])
            self.gaussian_blur(img, sig)
        
        return Image.fromarray(img)

    def sample_continuous(self,s):
        if len(s) == 1:
            return s[0]
        if len(s) == 2:
            rg = s[1] - s[0]
            return random() * rg + s[0]
        raise ValueError("Length of iterable s should be 1 or 2.")

    def sample_discrete(self,s):
        if len(s) == 1:
            return s[0]
        return choice(s)


    def gaussian_blur(self,img, sigma):
        gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
        gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
        gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)


    def cv2_jpg(self,img, compress_val):
        img_cv2 = img[:,:,::-1]
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
        result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
        decimg = cv2.imdecode(encimg, 1)
        return decimg[:,:,::-1]


    def pil_jpg(self,img, compress_val):
        out = BytesIO()
        img = Image.fromarray(img)
        img.save(out, format='jpeg', quality=compress_val)
        img = Image.open(out)
        # load from memory before ByteIO closes
        img = np.array(img)
        out.close()
        return img

    def jpeg_from_key(self,img, compress_val, key):
        method = self.jpeg_dict[key]
        return method(img, compress_val)

    def custom_resize(self,img):
        interp = self.sample_discrete(self.opt['rz_interp'])
        return TF.resize(img, (self.opt['loadSize'], self.opt['loadSize']), interpolation=self.rz_dict[interp])

    def preprocess(self, data):
        
        image_path, tf_label = data
        cmp_path = image_path.replace("NoCmp", self.opt["mode"])
        image, cmp_label = Image.open(image_path), False
        
        if os.path.isfile(cmp_path):
            cmp_image = Image.open(cmp_path)
            cmp_label = True
        else:
            cmp_image = image
        
        if image.mode == 'L':  
            image = image.convert('RGB')
            cmp_image = cmp_image.convert('RGB')

        if self.opt['isTrain'] and not self.opt['no_crop']:
            crop_func = transforms.RandomCrop(self.opt['cropSize'])
        else:
            crop_func = transforms.CenterCrop(self.opt['cropSize'])

        if self.opt['isTrain'] and not self.opt['no_flip']:
            flip_func = transforms.RandomHorizontalFlip()
        else:
            flip_func = transforms.Lambda(lambda img: img)

        if not self.opt['isTrain'] and self.opt['no_resize']:
            rz_func = transforms.Lambda(lambda img: img)
        else:
            rz_func = transforms.Lambda(lambda img: self.custom_resize(img))

        if self.opt['isTrain']:
            color_func = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5)
        else:
            color_func = transforms.Lambda(lambda img: img)

        if self.opt['isTrain']:
            rota_func = transforms.RandomRotation(degrees=180)
        else:
            rota_func = transforms.Lambda(lambda img: img)
        
        aug_func = transforms.Lambda(lambda img: self.data_augment(img))

        no_aug_transform =  transforms.Compose([
                    rz_func,
                    crop_func,
                    flip_func,
                    rota_func,
                    color_func
                ])

        transform = transforms.Compose([
                    rz_func,
                    crop_func,
                    flip_func,
                    aug_func,
                    rota_func,
                    color_func
                ])

        rng_state = torch.get_rng_state()
        cmp_image_pil = transform(cmp_image)
        torch.set_rng_state(rng_state)
        image_pil = no_aug_transform(image)
        
        cmp_rgb_np = np.array(cmp_image_pil)
        orig_rgb_np = np.array(image_pil)
        
        cmp_phase = extract_phase_spectrum(cmp_rgb_np)
        orig_phase = extract_phase_spectrum(orig_rgb_np)
        
        rgb_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        cmp_image_trans = rgb_transform(cmp_image_pil)
        img_trans = rgb_transform(image_pil)
        cmp_phase_tensor = torch.tensor(cmp_phase).unsqueeze(0).float()  # [1, H, W]
        orig_phase_tensor = torch.tensor(orig_phase).unsqueeze(0).float()  # [1, H, W]

        return cmp_image_trans, img_trans, cmp_phase_tensor, orig_phase_tensor, torch.tensor(tf_label, dtype=torch.long), torch.tensor(cmp_label, dtype=torch.bool)

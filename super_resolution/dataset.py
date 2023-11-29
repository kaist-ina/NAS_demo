#Python
import shutil, glob, random, time, os, ntpath, sys, re, logging, math, random
import numpy as np
import cv2
from PIL import Image, ImageFile
#ImageFile.LOAD_TRUNCATED_IMAGES = True

#Project
import utility as util

#PyTorch
import torch.utils.data as data
from torchvision.transforms import Compose, ToTensor, Resize

#Usage: dataset for training and testing DASH videos
#If fps is given, use it to pick frames
#If total_num is given, use it to pick frames

def get_resolution(resolution):
    assert resolution in [240, 360, 480, 720]

    if resolution == 720:
        t_w = 1920
        t_h = 1080
    elif resolution == 480:
        t_w = 960
        t_h = 540
    elif resolution == 360:
        t_w = 640
        t_h = 360
    elif resolution == 240:
        t_w = 480
        t_h = 270

    return (t_w, t_h)

class DatasetForDASH(data.Dataset):
    def __init__(self, opt, base_dir, video_dir, is_trained, total_num=-1, fps=-1):
        super(DatasetForDASH, self).__init__()
        assert not (fps != -1 and total_num != -1)
        assert os.path.exists(video_dir)
        os.makedirs(base_dir, exist_ok=True)

        self.base_dir = base_dir
        self.video_dir = video_dir
        self.lr = opt.dashLR
        self.hr = opt.dashHR
        self.target_lr = self.lr[0] #TODO: remove this defualt value
        self.scale = []
        for lr in self.lr:
            self.scale.append(math.floor(self.hr / lr))
        self.interpolation = opt.resize
        self.patch_size = opt.patchSize
        self.batch_size = opt.batchSize
        self.fps = fps
        self.total_num = total_num
        self.lazy_load = opt.lazy_load
        self.pre_upscaled = opt.pre_upscaled
        self.is_trained = is_trained

        self.input_transform = Compose([ToTensor(),])
        self.target_tarnsform = Compose([ToTensor(),])

        #Prepare datset from video
        self.prepareFromVideo()

    def getScaleList(self):
        return self.scale

    def setScaleIdx(self, idx_scale):
        assert idx_scale < len(self.scale)
        self.target_lr = self.lr[idx_scale]

    def setScale(self, scale):
        #print(self.scale)
        assert scale in self.scale
        self.target_lr = self.lr[self.scale.index(scale)]

    def setTargetLR(self, lr):
        self.target_lr = lr

    def bitrate(self, resolution):
        return self.bitrates[resolution]

    def getPathInfo(self, resolution, upscale):
        if self.fps != -1:
            checkfile = os.path.join(self.base_dir, 'new_prepare-{}p-{}fps'.format(resolution, self.fps))
            frames_dir = os.path.join(self.base_dir, '{}p-{}fps'.format(resolution, self.fps))
        elif self.total_num != -1:
            checkfile = os.path.join(self.base_dir, 'new_prepare-{}p-{}'.format(resolution, self.total_num))
            frames_dir = os.path.join(self.base_dir, '{}p-{}'.format(resolution, self.total_num))
        else:
            checkfile = os.path.join(self.base_dir, 'new_prepare-{}p'.format(resolution))
            frames_dir = os.path.join(self.base_dir, '{}p'.format(resolution))

        if not upscale and resolution != self.hr:
            checkfile += '-no-upscale'
            frames_dir += '-no-upscale'

        return checkfile, frames_dir

    def prepareFromVideo(self):
        #Check videos exists
        self.lr_videos = []
        for resolution in self.lr:
            lr_dir = os.path.join(self.video_dir, '{}p'.format(resolution))
            assert os.path.exists(lr_dir)
            filenames = glob.glob('{}/output*k.mp4'.format(lr_dir))
            assert len(filenames) == 1
            self.lr_videos.append((filenames[0], resolution))

        hr_dir = os.path.join(self.video_dir, '{}p'.format(self.hr))
        assert os.path.exists(hr_dir)
        filenames = glob.glob('{}/output*k.mp4'.format(hr_dir))
        assert len(filenames) == 1
        self.hr_video = (filenames[0], self.hr)

        #Get video information
        if self.hr == 720:
            target_resolution = (720, 1280)
        elif self.hr == 1080:
            target_resolution = (1080, 1920)

        #Save videos' bitrate
        self.bitrates = {}
        for lr_video in self.lr_videos:
            bitrate = util.get_video_bitrate(lr_video[0])
            self.bitrates[lr_video[1]] = bitrate
        self.bitrates[self.hr_video[1]] = util.get_video_bitrate(self.hr_video[0])

        #Design choice
        #i. Load all frames on memory - 20min video need more than 100GB, so not good
        #ii. Save all frames on filesystem and lazy loading - extract all frames as ('png' or 'jpg')

        #Prepare (upscaled) high-resolution input: i) VDSR style network that needs pre-upscaled input or comparing PSNR/SSIM with interpolation methods (e.g., bicubic)
        self.lr_upscaled_frames_dir = []
        for lr_video in self.lr_videos:
            logging.info('===>Prepare {}p upscaled frames'.format(lr_video[1]))
            checkfile, frames_dir = self.getPathInfo(lr_video[1], True)
            self.lr_upscaled_frames_dir.append(frames_dir)
            if not os.path.exists(checkfile):
                if os.path.exists(frames_dir) and os.path.isdir(frames_dir):
                    shutil.rmtree(frames_dir)
                os.makedirs(frames_dir)
                video_info = util.get_video_info(lr_video[0])
                total_frame = util.get_video_frame_count(lr_video[0])
                print(frames_dir)
                util.write_frame(frames_dir, lr_video[0], 0, video_info['fps'], target_resolution[1], target_resolution[0], self.interpolation, self.total_num, self.fps, True)
                #Create 'prepare-train' file for logging dataset completion
                with open(checkfile, 'w+') as f:
                    f.write(str(total_frame['frames']))

        #Prepare low-resolution input (bicubic upscale only for training setup)
        self.lr_frames_dir = []
        for lr_video in self.lr_videos:
            t_w, t_h = get_resolution(lr_video[1])
            logging.info('===>Prepare {}p no-upscaled frames'.format(lr_video[1]))
            checkfile, frames_dir = self.getPathInfo(lr_video[1], False)
            self.lr_frames_dir.append(frames_dir)
            if not os.path.exists(checkfile):
                if os.path.exists(frames_dir) and os.path.isdir(frames_dir):
                    shutil.rmtree(frames_dir)
                os.makedirs(frames_dir)
                video_info = util.get_video_info(lr_video[0])
                total_frame = util.get_video_frame_count(lr_video[0])
                #Calculate target resolution
                util.write_frame(frames_dir, lr_video[0], 0, video_info['fps'], t_w, t_h, 'bicubic', self.total_num, self.fps, True)
                #Create 'prepare-train' file for logging dataset completion
                with open(checkfile, 'w+') as f:
                    f.write(str(total_frame['frames']))

        #Prepare high-resolution target
        checkfile, frames_dir = self.getPathInfo(self.hr_video[1], True)
        self.hr_frames_dir = frames_dir
        logging.info('===>Prepare {}p original frames'.format(self.hr_video[1]))
        if not os.path.exists(checkfile):
            if os.path.exists(frames_dir) and os.path.isdir(frames_dir):
                shutil.rmtree(frames_dir)
            os.makedirs(frames_dir)
            video_info = util.get_video_info(self.hr_video[0])
            total_frame = util.get_video_frame_count(self.hr_video[0])
            util.write_frame_noscale(frames_dir, self.hr_video[0], 0, video_info['fps'], self.total_num, self.fps)

            #Create 'prepare-train' file for logging dataset completion
            with open(checkfile, 'w+') as f:
                f.write(str(total_frame['frames']))

        #Read filenames
        filenames = glob.glob('{}/*.png'.format(self.hr_frames_dir))
        filenames.sort(key=util.natural_keys)
        self.hr_filenames = filenames
        self.lr_upscaled_filenames = []
        for frames_dir in self.lr_upscaled_frames_dir:
            filenames = glob.glob('{}/*.png'.format(frames_dir))
            assert len(filenames) == len(self.hr_filenames)
            filenames.sort(key=util.natural_keys)
            self.lr_upscaled_filenames.append(filenames)
        self.lr_filenames = []
        for frames_dir in self.lr_frames_dir:
            filenames = glob.glob('{}/*.png'.format(frames_dir))
            assert len(filenames) == len(self.hr_filenames)
            filenames.sort(key=util.natural_keys)
            self.lr_filenames.append(filenames)

        #(Optional) Load an image on memory
        if not self.lazy_load:
            print('===> Load input images on memory')
            if self.pre_upscaled or not self.is_trained:
                self.lr_upscaled_images = []
                for input_filenames in self.lr_upscaled_filenames:
                    img_list = []
                    for name in input_filenames:
                        img = Image.open(name)
                        img.load()
                        img_list.append(img)
                    self.lr_upscaled_images.append(img_list)

            if not self.pre_upscaled:
                self.lr_images = []
                for input_filenames in self.lr_filenames:
                    img_list = []
                    for name in input_filenames:
                        img = Image.open(name)
                        img.load()
                        img_list.append(img)
                    self.lr_images.append(img_list)

            print('===> Load target images on memory')
            self.hr_images = []
            for name in self.hr_filenames:
                img = Image.open(name)
                img.load()
                self.hr_images.append(img)

    def getItemTrain(self):
        frame_idx = random.randrange(0, len(self.hr_filenames))

        if not self.lazy_load:
            if self.pre_upscaled:
                input = self.lr_upscaled_images[self.lr.index(self.target_lr)][frame_idx]
                target = self.hr_images[frame_idx]
            else:
                input = self.lr_images[self.lr.index(self.target_lr)][frame_idx]
                target = self.hr_images[frame_idx]
        else:
            if self.pre_upscaled:
                input = Image.open(self.lr_upscaled_filenames[self.lr.index(self.target_lr)][frame_idx])
                input.load()
                target = Image.open(self.hr_filenames[frame_idx])
                target.load()
            else:
                input = Image.open(self.lr_filenames[self.lr.index(self.target_lr)][frame_idx])
                input.load()
                target = Image.open(self.hr_filenames[frame_idx])
                target.load()

        #Randomly select crop lcation
        if self.patch_size != -1:
            height, width = input.size

            scale = self.scale[self.lr.index(self.target_lr)]
            height_ = random.randrange(0, height - self.patch_size + 1)
            width_ = random.randrange(0, width - self.patch_size + 1)

            input = input.crop((width_ , height_, width_ + self.patch_size, height_ + self.patch_size))
            if self.pre_upscaled:
                target = target.crop((width_, height_, (width_ + self.patch_size), (height_ + self.patch_size)))
            else:
                target = target.crop((width_ * scale, height_ * scale, (width_ + self.patch_size) * scale, (height_ + self.patch_size) * scale))

        input = self.input_transform(input)
        target = self.input_transform(target)

        return input, target

    def getItemTest(self, index):
        if not self.lazy_load:
            if self.pre_upscaled:
                input = self.lr_upscaled_images[self.lr.index(self.target_lr)][index]
                target = self.hr_images[index]
            else:
                input = self.lr_images[self.lr.index(self.target_lr)][index]
                target = self.hr_images[index]
                upscaled = self.lr_upscaled_images[self.lr.index(self.target_lr)][index]
        else:
            if self.pre_upscaled:
                input = Image.open(self.lr_filenames[self.lr.index(self.target_lr)][index])
                input.load()
                target = Image.open(self.hr_filenames[index])
                target.load()
            else:
                input = Image.open(self.lr_filenames[self.lr.index(self.target_lr)][index])
                input.load()
                target = Image.open(self.hr_filenames[index])
                target.load()
                upscaled = Image.open(self.lr_upscaled_filenames[self.lr.index(self.target_lr)][index])
                upscaled.load()

        if self.pre_upscaled:
            input = self.input_transform(input)
            upscaled = input.clone()
            target = self.input_transform(target)
        else:
            input = self.input_transform(input)
            upscaled = self.input_transform(upscaled)
            target = self.input_transform(target)

        return input, upscaled, target

    def lenTrain(self):
        if self.patch_size == -1:
            return len(self.hr_filenames)
        else:
            #return self.batch_size * len(self.hr_filenames)
            return len(self.hr_filenames) * self.batch_size * 3
            #return self.batch_size * 1000
            #note - set to 10000 since patch size 64 computes too fast - balance btw train and valid

    def lenTest(self):
        return len(self.hr_filenames)

    def __getitem__(self, index):
        if self.is_trained:
            return self.getItemTrain()
        else:
            return self.getItemTest(index)

    def __len__(self):
        if self.is_trained:
            length = self.lenTrain()
        else:
            length = self.lenTest()
        return length

#Data-augmentation is enalbed by randomly selecting among 8 choices
class DatasetForDIV2K(data.Dataset):
    @staticmethod
    def input_transform():
        return Compose([
            ToTensor(),
        ])

    @staticmethod
    def target_transform():
        return Compose([
            ToTensor(),
        ])

    def __init__(self, opt, image_dir):
        super(DatasetForDIV2K, self).__init__()

        self.scale = opt.div2kLR
        self.patch_size = opt.patchSize
        self.batch_size = opt.batchSize
        self.image_dir = image_dir
        self.lr = [240, 360, 480, 720]
        self.target_lr = self.lr[0] #TODO: remove this defualt value
        self.interpolation = opt.resize

        self.input_transform = self.input_transform()
        self.target_transform = self.target_transform()

        #self.target_dir = os.path.join(self.image_dir, 'DIV2K_train_HR_modcrop{}.')
        self.target_dir= os.path.join(self.image_dir, 'train-1080p')
        print(self.target_dir)
        assert os.path.exists(self.target_dir)

        #self.target_filenames = [os.path.join(self.target_dir, x) for x in os.listdir(self.target_dir) if is_image_file(x)]
        self.target_filenames = glob.glob('{}/*.png'.format(self.target_dir))
        self.target_filenames.sort(key=util.natural_keys)

        #dataset creat for DIV2K image
        self.input_dirs = []
        for scale in self.scale:
            checkfile = os.path.join(self.image_dir, 'prepare-train-{}-{}'.format(scale, self.interpolation))
            upscaled_dir = os.path.join(image_dir, 'lr_x{}_{}'.format(scale, self.interpolation))

            if not os.path.exists(checkfile):
                logging.info('===>Prepare x{} images'.format(scale))
                if os.path.exists(upscaled_dir) and os.path.isdir(upscaled_dir):
                    shutil.rmtree(upscaled_dir)
                os.makedirs(upscaled_dir)

                for path in self.target_filenames:
                    util.resize_div2k(upscaled_dir, path, scale, self.interpolation, 1, True)

                with open(checkfile, 'w+') as f:
                    f.write('div2k dataset is prepared')

            self.input_dirs.append(upscaled_dir)

        #Make a filenames list and sort it
        self.input_filenames = []
        for input_dir in self.input_dirs:
            #input_filenames = [os.path.join(input_dir, x) for x in os.listdir(input_dir) if is_image_file(x)]
            input_filenames = glob.glob('{}/*.png'.format(input_dir))
            input_filenames.sort(key=util.natural_keys)
            self.input_filenames.append(input_filenames)

        #Lazy loading rather than load on memory
        #Load an image on memory
        print('===> Load input images on memory')
        self.input_images = []
        for input_filenames in self.input_filenames:
            img_list = []
            for name in input_filenames:
                img = Image.open(name)
                img.load()
                img_list.append(img)
            self.input_images.append(img_list)

        print('===> Load target images on memory')
        self.target_images = []
        for name in self.target_filenames:
            img = Image.open(name)
            img.load()
            self.target_images.append(img)

    def __getitem__(self, index):
        #input_index = np.random.randint(0, len(self.input_filenames))
        image_index = random.randrange(0, len(self.target_filenames))

        #get an image from memory
        #input = Image.open(self.input_filenames[input_index][image_index])
        #target = Image.open(self.target_filenames[image_index])

        input = self.input_images[self.lr.index(self.target_lr)][image_index]
        target = self.target_images[image_index]

        #crop
        height, width = target.size
        scale = self.scale[self.lr.index(self.target_lr)]
        height_ = random.randrange(0, height - self.patch_size + 1)
        width_ = random.randrange(0, width - self.patch_size + 1)

        input = input.crop((width_, height_, width_ + self.patch_size, height_ + self.patch_size))
        target = target.crop((width_ * scale, height_ * scale, (width_ + self.patch_size) * scale, (height_ + self.patch_size) * scale))

        #print(input.size, target.size)

        #input = self.input_transform(input)
        #upscaled = self.input_transform(upscaled)
        #target = self.input_transform(target)

        #transform
        input = self.input_transform(input)
        target = self.target_transform(target)

        return input, target

    def getScaleList(self):
        return self.scale

    def setScaleIdx(self, idx_scale):
        assert idx_scale < len(self.scale)
        self.target_lr = self.lr[idx_scale]

    def setScale(self, scale):
        #print(self.scale)
        assert scale in self.scale
        self.target_lr = self.lr[self.scale.index(scale)]

    def setTargetLR(self, lr):
        self.target_lr = lr

    def __len__(self):
        #return self.batch_size * 100
        return self.batch_size * 1000
        #return self.batch_size * 10000 #iterate 1000 mini-batches per epoch

#Data-augmentation is enalbed by randomly selecting among 8 choices
class TestDatasetForDIV2K(data.Dataset):
    @staticmethod
    def input_transform():
        return Compose([
            ToTensor(),
        ])

    @staticmethod
    def target_transform():
        return Compose([
            ToTensor(),
        ])

    def __init__(self, image_dir, scale, interpolation, image_format):
        super(TestDatasetForDIV2K, self).__init__()
        self.scale = scale
        self.current_scale = scale[0]
        self.image_dir = image_dir
        self.image_format = image_format
        self.interpolation = interpolation

        self.input_transform = self.input_transform()
        self.target_transform = self.target_transform()

        self.target_dir = os.path.join(image_dir, 'DIV2K_valid_HR')
        assert os.path.exists(self.target_dir)

        self.target_filenames = [os.path.join(self.target_dir, x) for x in os.listdir(self.target_dir) if is_image_file(x)]
        self.target_filenames.sort(key=natural_keys)

        #dataset creat for DIV2K image
        self.input_dirs = {}
        for scale in self.scale:
            checkfile = os.path.join(self.image_dir, 'prepare-valid-{}-{}-{}'.format(scale, self.image_format, self.interpolation))
            upscaled_dir = os.path.join(image_dir, 'lr_x{}_upscaled_{}_{}_valid'.format(scale, self.image_format, self.interpolation))

            if not os.path.exists(checkfile):
                logging.info('===>Prepare x{} images'.format(scale))
                if os.path.exists(upscaled_dir) and os.path.isdir(upscaled_dir):
                    shutil.rmtree(upscaled_dir)
                os.makedirs(upscaled_dir)

                for path in self.target_filenames:
                    util.resize_div2k(upscaled_dir, path, scale, self.interpolation, self.image_format)

                with open(checkfile, 'w+') as f:
                    f.write('div2k dataset is prepared')

            self.input_dirs[scale] = upscaled_dir

        #Make a filenames list and sort it
        self.input_filenames = {}
        for scale, input_dir in self.input_dirs.items():
            input_filenames = [os.path.join(input_dir, x) for x in os.listdir(input_dir) if is_image_file(x)]
            input_filenames.sort(key=natural_keys)
            self.input_filenames[scale] = input_filenames

        #Lazy loading rather than load on memory
        #Load an image on memory
        #print('===> Load input images on memory')
        self.input_images = {}
        for scale, input_filenames in self.input_filenames.items():
            img_list = []
            for name in input_filenames:
                img = Image.open(name)
                img.load()
                img_list.append(img)
            self.input_images[scale] = img_list

        #print('===> Load target images on memory')
        self.target_images = []
        for name in self.target_filenames:
            img = Image.open(name)
            img.load()
            self.target_images.append(img)

    def setscale(self, scale):
        self.current_scale = scale

    def __getitem__(self, index):
        input = self.input_images[self.current_scale][index]
        target = self.target_images[index]

        #transform
        input = self.input_transform(input)
        target = self.target_transform(target)

        return input, target

    def __len__(self):
        return len(self.target_images)

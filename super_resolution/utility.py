#System
import logging, os, math, re, ntpath, sys, xlsxwriter, subprocess, datetime, shlex, json, pprint, time

#Python
import numpy as np
from PIL import Image
from bisect import bisect_left

#Pytorch
import torch.nn as nn

THREAD_NUM=4

class videoInfo:
    def __init__(self, fps, duration, quality):
        self.fps = fps
        self.duration = duration
        self.quality = quality

class discrete_cdf:
    def __init__(self, data):
        self._data = data # must be sorted
        self._data_len = float(len(data))

    def __call__(self, point):
        #return (len(self._data[:bisect_left(self._data, point)]) / self._data_len)
        return (sum(i <= point for i in self._data) / self._data_len)

def psnr(pred, gt, shave_border=0, max_value=255.0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(max_value / rmse)

#Unnormalize 0-1 to 0-255
def unnormalize(img):
    img = img*255.
    img = np.clip(img, 0., 255.)

    return img

class timer():
    def __init__(self):
        self.acc = 0
        self.total_time = 0
        self.tic()

    def tic(self):
        self.t0 = time.perf_counter()

    def toc(self):
        return time.perf_counter() - self.t0

    def toc_total_sum(self):
        elapsed_time = time.perf_counter() - self.t0
        return self.total_time + elapsed_time

    def toc_total_add(self):
        elapsed_time = time.perf_counter() - self.t0
        self.total_time += elapsed_time
        return elapsed_time

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0

    def add_total(time):
        self.total_time += time

def atoi(text):
        return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]


#python-ffmpeg: https://gist.github.com/hiwonjoon/035a1ead72a767add4b87afe03d0dd7b
def get_video_info(fileloc) :
    command = ['ffprobe',
               '-v', 'fatal',
               '-show_entries', 'stream=width,height,r_frame_rate,duration',
               '-of', 'default=noprint_wrappers=1:nokey=1',
               fileloc, '-sexagesimal']
    ffmpeg = subprocess.Popen(command, stderr=subprocess.PIPE ,stdout = subprocess.PIPE )
    out, err = ffmpeg.communicate()
    if(err) : print(err)
    out = out.decode().split('\n')
    return {'file' : fileloc,
            'width': int(out[0]),
            'height' : int(out[1]),
            'fps': float(out[2].split('/')[0])/float(out[2].split('/')[1]),
            'duration' : out[3] }

def get_video_bitrate(fileloc):
    cmd = "ffprobe -v quiet -print_format json -show_streams"
    args = shlex.split(cmd)
    args.append(fileloc)
    # run the ffprobe process, decode stdout into utf-8 & convert to JSON
    ffprobeOutput = subprocess.check_output(args).decode('utf-8')
    ffprobeOutput = json.loads(ffprobeOutput)

    # prints all the metadata available:
    #pp = pprint.PrettyPrinter(indent=2)
    #pp.pprint(ffprobeOutput)

    # for example, find height and width
    bitrate = int(ffprobeOutput['streams'][0]['bit_rate']) / 1000

    return bitrate

def get_video_frame_count(fileloc) : # This function is spearated since it is slow.
    command = ['ffprobe',
               '-v', 'fatal',
               '-count_frames',
               '-show_entries', 'stream=nb_read_frames',
               '-of', 'default=noprint_wrappers=1:nokey=1',
               fileloc, '-sexagesimal']
    ffmpeg = subprocess.Popen(command, stderr=subprocess.PIPE ,stdout = subprocess.PIPE )
    out, err = ffmpeg.communicate()
    if(err) : print(err)
    out = out.decode().split('\n')
    if out[0].isdigit():
        return {'file' : fileloc,
                'frames' : int(out[0])}
    else:
        return None

def read_frame(fileloc,frame,fps,num_frame,t_w,t_h, interpolation) :
    command = ['ffmpeg',
               '-loglevel', 'fatal',
               '-ss', str(datetime.timedelta(seconds=frame/fps)),
               '-i', fileloc,
               #'-vf', '"select=gte(n,%d)"'%(frame),
               '-threads', str(THREAD_NUM),
               #'-threads', 0,
               '-vf', 'scale=%d:%d'%(t_w,t_h),
               '-sws_flags', '%s'%(interpolation),
               '-vframes', str(num_frame),
               '-f', 'image2pipe',
               '-pix_fmt', 'rgb24',
               '-vcodec', 'rawvideo', '-']
    #print(command)
    ffmpeg = subprocess.Popen(command, stderr=subprocess.PIPE ,stdout = subprocess.PIPE )
    out, err = ffmpeg.communicate()
    if(err) : print('error',err); return None;
    if num_frame == 1:
        video = np.fromstring(out, dtype='float32').reshape((3, t_h, t_w)) #NCHW
    else:
        video = np.fromstring(out, dtype='float32').reshape((num_frame, 3, t_h, t_w)) #NCHW
    return video

def read_frame_noscale(fileloc,frame,fps,num_frame,t_w,t_h, interpolation) :
    command = ['ffmpeg',
               '-loglevel', 'fatal',
               '-ss', str(datetime.timedelta(seconds=frame/fps)),
               '-i', fileloc,
               #'-vf', '"select=gte(n,%d)"'%(frame),
               '-threads', str(THREAD_NUM),
               '-vframes', str(num_frame),
               '-f', 'image2pipe',
               '-pix_fmt', 'rgb24',
               '-vcodec', 'rawvideo', '-']
    #print(command)
    ffmpeg = subprocess.Popen(command, stderr=subprocess.PIPE ,stdout = subprocess.PIPE )
    out, err = ffmpeg.communicate()
    if(err) : print('error',err); return None;
    if num_frame == 1:
        video = np.fromstring(out, dtype='float32').reshape((3, t_h, t_w)) #NCHW
    else:
        video = np.fromstring(out, dtype='float32').reshape((num_frame, 3, t_h, t_w)) #NCHW
    return video

#Resize tool for frame by frame interpolation
#lr: low resolution (width, height)
#hr: high resolution (width, height)
def resize_dash(targetloc, inputloc, lr, hr, interpolation, image_format, start_number=1):
    command = ['ffmpeg',
               '-loglevel', 'fatal',
               '-start_number', '%d'%(start_number),
               '-i', '{}/%d.png'.format(inputloc),
               '-threads', str(THREAD_NUM),
               '-vf', 'scale=%d:%d'%(lr[1], lr[0]) + ',scale=%d:%d'%(hr[1], hr[0]),
               '-sws_flags', '%s'%(interpolation),
               '-start_number', '%d'%(start_number),
               '{}/%d.{}'.format(targetloc, image_format)
               ]

    ffmpeg = subprocess.Popen(command, stderr=subprocess.PIPE ,stdout = subprocess.PIPE )
    out, err = ffmpeg.communicate()
    if(err) : print('error',err); return None;

#Assumption: original high-resolutin is 1080p
def resize_div2k(targetloc, path, scale, interpolation, start_number=1, pre_upscaled=False):
    with Image.open(path) as img:
        width, height = img.size

    filename = ntpath.basename(path)
    filename, _ = os.path.splitext(filename)

    if scale == 4:
        pre_width = 426
        pre_height = 240
        pos_width = 480
        pos_height = 270
    elif scale == 3:
        pre_width = 640
        pre_height = 360
        pos_width = 640
        pos_height = 360
    elif scale == 2:
        pre_width = 854
        pre_height = 480
        pos_width = 960
        pos_height = 540
    elif scale == 1:
        pre_width = 1280
        pre_height = 720
        pos_width = 1920
        pos_height = 1080
    else:
        print('unsupported scale')
        sys.exit(-1)

    if pre_upscaled:
        command = ['ffmpeg',
                   '-loglevel', 'fatal',
                   '-i', '{}'.format(path),
                   '-threads', str(THREAD_NUM),
                   '-vf', 'scale={}:{}'.format(pre_width, pre_height) + ',scale={}:{}'.format(pos_width, pos_height),
                   #'-vf', 'scale={}:{}'.format(width // scale, height // scale),
                   '-sws_flags', '%s'%(interpolation),
                   '{}/{}.png'.format(targetloc, filename)
                   ]
    else:
        command = ['ffmpeg',
                   '-loglevel', 'fatal',
                   '-i', '{}'.format(path),
                   '-threads', str(THREAD_NUM),
                   '-vf', 'scale={}:{}'.format(pre_width, pre_height),
                   #'-vf', 'scale={}:{}'.format(width // scale, height // scale),
                   '-sws_flags', '%s'%(interpolation),
                   '{}/{}.png'.format(targetloc, filename)
                   ]

    ffmpeg = subprocess.Popen(command, stderr=subprocess.PIPE ,stdout = subprocess.PIPE )
    out, err = ffmpeg.communicate()
    if(err) : print('error',err); return None;

"""
def resize_div2k(targetloc, path, scale, interpolation, start_number=1):
    with Image.open(path) as img:
        width, height = img.size

    filename = ntpath.basename(path)
    filename, _ = os.path.splitext(filename)

    command = ['ffmpeg',
               '-loglevel', 'fatal',
               '-i', '{}'.format(path),
               '-threads', str(THREAD_NUM),
               #'-vf', 'scale={}:{}'.format(width // scale, height // scale) + ',scale={}:{}'.format(width, height),
               '-vf', 'scale={}:{}'.format(width // scale, height // scale),
               '-sws_flags', '%s'%(interpolation),
               '{}/{}.png'.format(targetloc, filename)
               ]

    ffmpeg = subprocess.Popen(command, stderr=subprocess.PIPE ,stdout = subprocess.PIPE )
    out, err = ffmpeg.communicate()
    if(err) : print('error',err); return None;
"""

def write_frame(imageloc, videoloc, frame, fps, t_w, t_h, interpolation, extract_num, extract_fps, upscale=True):
    if extract_num != -1:
        print('[utility.py] Reimplement: frames should space equally')
        sys.exit()
        command = ['ffmpeg',
                   '-loglevel', 'fatal',
                   '-ss', str(datetime.timedelta(seconds=frame/fps)),
                   '-i', videoloc,
                   #'-vf', '"select=gte(n,%d)"'%(frame),
                   '-threads', str(THREAD_NUM),
                   '-vf', 'scale=%d:%d'%(t_w,t_h),
                   '-sws_flags', '%s'%(interpolation),
                   '-vframes', str(extract_num),
                   '{}/%d.png'.format(imageloc)]
    elif extract_fps != -1:
        if upscale:
            vf_args = 'scale=%d:%d, fps=%f'%(t_w, t_h, extract_fps)
        else:
            vf_args = 'fps=%f'%(extract_fps)
        command = ['ffmpeg',
                   '-loglevel', 'fatal',
                   '-ss', str(datetime.timedelta(seconds=frame/fps)),
                   '-i', videoloc,
                   #'-vf', '"select=gte(n,%d)"'%(frame),
                   '-threads', str(THREAD_NUM),
                   '-vf', vf_args,
                   '-sws_flags', '%s'%(interpolation),
                   '{}/%d.png'.format(imageloc)]
    else:
        #Check if this command extract all frames
        if upscale:
            vf_args = 'scale=%d:%d, fps=%f'%(t_w, t_h, fps)
        else:
            vf_args = 'fps=%f'%(fps)
        command = ['ffmpeg',
                   '-loglevel', 'fatal',
                   '-ss', str(datetime.timedelta(seconds=frame/fps)),
                   '-i', videoloc,
                   #'-vf', '"select=gte(n,%d)"'%(frame),
                   '-threads', str(THREAD_NUM),
                   '-vf', vf_args,
                   '-sws_flags', '%s'%(interpolation),
                   '{}/%d.png'.format(imageloc)]

    ffmpeg = subprocess.Popen(command, stderr=subprocess.PIPE ,stdout = subprocess.PIPE )
    out, err = ffmpeg.communicate()
    if(err) : print('error',err); return None;


def write_frame_noscale(imageloc, videoloc, frame, fps, extract_num, extract_fps):
    if extract_num != -1:
        print('Reimplement: frames should space equally')
        sys.exit()
        command = ['ffmpeg',
                   '-loglevel', 'fatal',
                   '-ss', str(datetime.timedelta(seconds=frame/fps)),
                   '-i', videoloc,
                   '-threads', str(THREAD_NUM),
                   '-vf', 'fps=%f'%(extract_fps),
                   '-vframes', str(extract_num),
                   '{}/%d.png'.format(imageloc)]
    elif extract_fps != -1:
        command = ['ffmpeg',
                   '-loglevel', 'fatal',
                   '-ss', str(datetime.timedelta(seconds=frame/fps)),
                   '-i', videoloc,
                   #'-vf', '"select=gte(n,%d)"'%(frame),
                   '-threads', str(THREAD_NUM),
                   '-vf', 'fps=%f'%(extract_fps),
                   '{}/%d.png'.format(imageloc)]
    else:
        #Check if this command extract all frames
        command = ['ffmpeg',
                   '-loglevel', 'fatal',
                   '-ss', str(datetime.timedelta(seconds=frame/fps)),
                   '-i', videoloc,
                   #'-vf', '"select=gte(n,%d)"'%(frame),
                   '-threads', str(THREAD_NUM),
                   '-vf', 'fps=%f'%(fps),
                   '{}/%d.png'.format(imageloc)]

    ffmpeg = subprocess.Popen(command, stderr=subprocess.PIPE ,stdout = subprocess.PIPE )
    out, err = ffmpeg.communicate()
    if(err) : print('error',err); return None;

def getLogger(save_dir, save_name):
    Logger = logging.getLogger(save_name)
    Logger.setLevel(logging.INFO)
    Logger.propagate = False

    filePath = os.path.join(save_dir, save_name)
    if os.path.exists(filePath):
        os.remove(filePath)

    fileHandler = logging.FileHandler(filePath)
    #fileHandler = logging.FileHandler("{}.log".format(save_name))
    logFormatter = logging.Formatter('%(message)s')
    fileHandler.setFormatter(logFormatter)
    Logger.addHandler(fileHandler)
    #consoleHandler = logging.StreamHandler()
    #consoleHandler.setFormatter(logFormatter)
    #Logger.addHandler(consoleHandler)

    return Logger

def getAct(name):
    if name == 'relu':
        return nn.ReLU(inplace=True)
    elif name == 'prelu':
        return nn.PReLU()
    elif name == 'lrelu':
        return nn.LeakyReLU(0.2, inplace=True)
    else:
        print('Invalid Activation Name')
        sys.exit()

"""
TODO: resize frame to LR resolution and upscale
def write_frame_resized(imageloc, videoloc, frame, fps, t_w, t_h, interpolation, extract_num, extract_fps):
    if extract_num != -1:
        print('Reimplement: frames should space equally')
        sys.exit()
        command = ['ffmpeg',
                   '-loglevel', 'fatal',
                   '-ss', str(datetime.timedelta(seconds=frame/fps)),
                   '-i', videoloc,
                   #'-vf', '"select=gte(n,%d)"'%(frame),
                   '-threads', str(THREAD_NUM),
                   '-vf', 'scale=%d:%d'%(t_w,t_h),
                   '-sws_flags', '%s'%(interpolation),
                   '-vframes', str(extract_num),
                   '{}/%d.png'.format(imageloc)]
    elif extract_fps != -1:
        command = ['ffmpeg',
                   '-loglevel', 'fatal',
                   '-ss', str(datetime.timedelta(seconds=frame/fps)),
                   '-i', videoloc,
                   #'-vf', '"select=gte(n,%d)"'%(frame),
                   '-threads', str(THREAD_NUM),
                   '-vf', 'scale=%d:%d, fps=%f'%(t_w,t_h, extract_fps),
                   '-sws_flags', '%s'%(interpolation),
                   '{}/%d.png'.format(imageloc)]
    else:
        #Check if this command extract all frames
        command = ['ffmpeg',
                   '-loglevel', 'fatal',
                   '-ss', str(datetime.timedelta(seconds=frame/fps)),
                   '-i', videoloc,
                   #'-vf', '"select=gte(n,%d)"'%(frame),
                   '-threads', str(THREAD_NUM),
                   '-vf', 'scale=%d:%d, fps=%f'%(t_w,t_h, fps),
                   '-sws_flags', '%s'%(interpolation),
                   '{}/%d.png'.format(imageloc)]

    ffmpeg = subprocess.Popen(command, stderr=subprocess.PIPE ,stdout = subprocess.PIPE )
    out, err = ffmpeg.communicate()
    if(err) : print('error',err); return None;
"""

def random_gradual_03(elem_list):
    random_list = []

    for i in range(len(elem_list)):
        if i == len(elem_list) - 1:
            random_list.extend([elem_list[i]] * len(random_list))
        else:
            random_list.extend([elem_list[i]] *  1)

    return random_list

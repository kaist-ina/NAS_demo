#Python
import argparse, os, sys, logging, random, time, queue, signal, copy
from subprocess import Popen
import subprocess as sub
from shutil import copyfile
import numpy as np
from skimage.io import imsave
import threading
#Project
# from dataloader import *
from dataset import *
import utility as util
from option import *
from model import NAS
#from model import NAS_old as NAS #TODO: rollback to first model

#PyTorch
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
# from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Resize
import torch.multiprocessing as mp

import cv2 #Import Error when import cv2 before torch

TMP_DIR = 'tmp_video/'
INPUT_VIDEO_NAME = 'input.mp4'
MAX_FPS =  30
MAX_SEGMENT_LENGTH = 4
SHARED_QUEUE_LEN = MAX_FPS * MAX_SEGMENT_LENGTH #Regulate GPU memory usage (> 3 would be fine)

def get_resolution(quality):
    assert quality in [0, 1, 2, 3]

    if quality == 3:
        t_w = 1920
        t_h = 1080
    elif quality == 2:
        t_w = 960
        t_h = 540
    elif quality == 1:
        t_w = 640
        t_h = 360
    elif quality == 0:
        t_w = 480
        t_h = 270

    return (t_h, t_w)

def decode(decode_queue, encode_queue, data_queue, shared_tensor_list):
    logger = util.getLogger(opt.resultDir, 'decode.log')

    while True:
        try:
            input = decode_queue.get()
            start_time = time.time()
            #print('decode [start]: {}sec'.format(start_time))

            header_file = input[0]
            video_file = input[1]
            output_input = input[2]
            video_info = input[3]

            print(header_file)
            if not os.path.exists(header_file):
                print('decode: header does not exist')
                continue
            if not os.path.exists(video_file):
                print('decode: video does not exist')
                continue

            video_file_name, _  = os.path.splitext(os.path.basename(video_file))
            process_dir = os.path.join(TMP_DIR, '{}_{}_{}'.format(opt.contentType, video_file_name, video_info.quality))
            os.makedirs(process_dir, exist_ok=True)

            #Merge Header.m4s and X.m4s
            input_video = os.path.join(process_dir, INPUT_VIDEO_NAME)
            with open(input_video, 'wb')as outfile:
                with open(header_file, 'rb') as infile:
                    outfile.write(infile.read())

                with open(video_file, 'rb') as infile:
                    outfile.write(infile.read())

            #Call super-resolution / encoder processes to start
            data_queue.put(('process_dir', process_dir)) #TODO: remove this (just for timeline figure)
            data_queue.join()
            encode_queue.put(('start', process_dir, video_info))
            t_h, t_w = get_resolution(video_info.quality)
            targetScale = int(1080/t_h)
            targetHeight = t_h
            data_queue.put(('configure', targetScale, targetHeight))
            data_queue.join()
            print('decode [configuration]: {}sec'.format(time.time() - start_time))

            #Read frame and prepare PyTorch CUDA tensors
            vc = cv2.VideoCapture(input_video)
            frame_count = 0
            print('decode [video read prepare]: {}sec'.format(time.time() - start_time))
            while True:
                #Read frames
                rval, frame = vc.read()
                if rval == False:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (t_w, t_h), interpolation=cv2.INTER_CUBIC) #add bicubic resize
                input_t_ = torch.from_numpy(frame).byte().cuda()

                #Copy tensors to shared CUDA memory
                shared_tensor_list[t_h][frame_count % SHARED_QUEUE_LEN].copy_(input_t_)
                data_queue.put(('frame', frame_count))
                frame_count += 1
            vc.release()

            print('decode [prepare_frames-{} frames]: {}sec'.format(frame_count, time.time() - start_time))
            data_queue.join()
            print('decode [super-resolution end]: {}sec'.format(time.time() - start_time))
            encode_queue.join() #wait for encode to be ended
            encode_queue.put(('end', output_input))
            encode_queue.join()
            print('decode [encode end] : {}sec'.format(time.time() - start_time))

        except (KeyboardInterrupt, SystemExit):
            print('exiting...')
            break

inference_idx = 0 #inference layer of a scalable DNN #used in load_dnn_chunk(), process_video_chunk()
model = NAS.Multi_Network(opt.quality)
model = model.cuda()
model = model.half()
#if opt.processMode != 'runtime':

def load_dnn_chunk(dnn_queue):
    global inference_idx
    global model
    while True:
        try:
            input = dnn_queue.get()
            #load a pretrained model of which path is given
            if input[0] == 'model':
                start_time = time.time()
                #print('model_loading [start]: {}sec'.format(start_time))

                pretrained_path = input[1]
                print(pretrained_path)
                if not os.path.exists(pretrained_path):
                    print('sr: Model does not exist')
                    continue

                weights = torch.load(pretrained_path)
                model.load_state_dict(weights['model'].state_dict(), strict=True)
                model = model.half() #temporary for float-32 model upload
                #inference_idx# = 4 - deprecated

                #TODO - check whether it works
                scale_list = [1,2,3,4]
                for scale in scale_list:
                    inference_idx = max(inference_idx, len(model.getOutputNodes(scale))-1)
                elapsed_time = time.time()
                #print('model_loading [end] : {}sec'.format(elapsed_time))
                print('model_loading [elapsed] : {}sec'.format(elapsed_time - start_time))

            elif input[0] == 'test_figure16':
                inference_idx = input[1]

            elif input[0] == 'test_runtime':
                scale_list = [1,2,3,4]
                for scale in scale_list:
                    inference_idx = max(inference_idx, len(model.getOutputNodes(scale))-1)

            #TODO 1. DNN chunk load 2. NAS-FULL vs NAS - implement when we evaluate this
            elif input[0] == 'dnn_chunk':
                start_time = time.time()
                dnn_chunk_path = input[1]
                dnn_chunk_idx = input[2] #idx : 0,1,2,3,4

                weights = torch.load(dnn_chunk_path)
                model.load_state_dict(weights, strict=False)
                inference_idx = dnn_chunk_idx
                print('inference_idx: {}'.format(inference_idx))
                end_time = time.time()
                print('dnn_chunk_loading [elapsed] : {}sec'.format(end_time - start_time))

            #input: DNN_list, fps, duration
            elif input[0] == 'test_dnn':
                DNN_list = input[1]
                fps = input[2]
                duration = input[3]
                test_input = input[4]
                is_break = False

                #Prepare a random input
                start_time = time.time()
                random_tensor = torch.HalfTensor(1, 3, 480, 270) #TODO: test with resolution which takes the longest time
                input = Variable(random_tensor, volatile=True)
                input = input.cuda()

                #Prepare and test a mock DNN
                for DNN in DNN_list:
                    inference_time_list = []
                    layer = DNN[0]
                    feature = DNN[1]
                    #mock_DNN = NAS.Single_Network(nLayer=layer, nFeatBase=feature // 2, nFeatBody=feature, nChannel=3, scale=3, outputFilter=1, bias=True, act=nn.ReLU(True))
                    mock_DNN = NAS.Single_Network(nLayer=layer, nFeat=feature, nChannel=3, scale=4, outputFilter=2, bias=True, act=nn.ReLU(True))
                    mock_DNN = mock_DNN.cuda()
                    mock_DNN = mock_DNN.half()
                    output_node = mock_DNN.getOutputNodes()[-1]

                    #Dummy-inference: initial CUDA run has overhead
                    for _ in range(10):
                        mock_DNN(input, output_node)
                    torch.cuda.synchronize()

                    #Real-inference
                    for _ in range(10):
                        start_inference = time.time()
                        mock_DNN(input, output_node)
                        torch.cuda.synchronize()
                        end_inference = time.time()
                        inference_time_list.append(end_inference - start_inference)

                        if np.mean(inference_time_list) < (duration) / (fps * duration): #TODO: replace 0.1 with reasonable value
                            break


                    print('DNN inference [{}]: {}sec'.format(DNN_list.index(DNN),np.mean(inference_time_list)))
                    #if end_inference - start_inference > (duration - 0.1) / fps: #TODO: replace 0.1 with reasonable value

                    if np.mean(inference_time_list) > (duration) / (fps * duration): #TODO: replace 0.1 with reasonable value
                        is_break = True
                        break

                end_time = time.time()
                print('test mock DNNs [elapsed] : {}sec'.format(end_time - start_time))
                if is_break:
                    test_input.send((DNN_list.index(DNN) - 1,))
                else:
                    test_input.send((DNN_list.index(DNN),))
            else:
                print('sr: Invalid input')

            dnn_queue.task_done()

        except (KeyboardInterrupt, SystemExit):
            print('exiting...')
            break

def process_video_chunk(encode_queue, shared_tensor_list, data_queue):
    global inference_idx
    global model
    targetHeight = None
    process_dir = None

    while True:
        try:
            input = data_queue.get()

            if input[0] == 'configure':
                targetScale = input[1]
                targetHeight = input[2]
                model.setScale(targetScale)
                if targetScale != 1:
                    inference_idx_ = inference_idx * 2
                else:
                    inference_idx_ = min(inference_idx, len(model.getOutputNodes(targetScale))-1)
                inference_time_list = []
                encode_queue.put(('index', inference_idx_))

            elif input[0] == 'process_dir':
                process_dir = input[1]

            elif input[0] == 'frame':
                start_time = time.time()
                frame_count = input[1]

                input_tensor_ = shared_tensor_list[targetHeight][frame_count % SHARED_QUEUE_LEN]
                input_tensor_ = input_tensor_.permute(2,0,1).half()
                input_tensor_.div_(255) #byte tensor/255
                input_tensor_.unsqueeze_(0)
                input_ = Variable(input_tensor_, volatile=True)

                output_ = model(input_, inference_idx_)
                output_ = output_.data[0].permute(1,2,0)
                output_ = output_ * 255
                output_ = torch.clamp(output_, 0, 255)
                output_ = output_.byte()
                shared_tensor_list[1080][frame_count % SHARED_QUEUE_LEN].copy_(output_)
                torch.cuda.synchronize()

                encode_queue.put(('frame', frame_count % SHARED_QUEUE_LEN))
                end_time = time.time()

                if opt.enable_debug:
                    output_np = output_.float().cpu().numpy().astype(np.uint8)
                    imsave(os.path.join('{}/sr_{}p_{}.png'.format(process_dir, targetHeight, frame_count % SHARED_QUEUE_LEN)), output_np)

                inference_time_list.append(end_time - start_time)

                #For measuring a DNN run-time
                if frame_count == 95:
                    print('process [index: {}, total-{}frames]: {}sec'.format(inference_idx_, len(inference_time_list), np.sum(inference_time_list)))
            else:
                print('sr: Invalid input')

            data_queue.task_done()

        except (KeyboardInterrupt, SystemExit):
            print('exiting...')
            break

def super_resolution_threading(encode_queue, dnn_queue, data_queue, shared_tensor_list):
    dnn_load_thread = threading.Thread(target=load_dnn_chunk, args=(dnn_queue,))
    video_process_thread = threading.Thread(target=process_video_chunk, args=(encode_queue, shared_tensor_list, data_queue))
    dnn_load_thread.start()
    video_process_thread.start()
    dnn_load_thread.join()
    video_process_thread.join()

#TODO (minor) : 1. remove all images
def encode(encode_queue, shared_tensor_list):
    pipe = None
    process_dir = None
    infer_idx = None

    while(1):
        try:
            input = encode_queue.get()

            if input[0] == 'start':
                encode_start_time = time.time()
                #print('encode [start]: {}sec'.format(encode_start_time))

                process_dir = input[1]
                video_info = input[2]

                fps = video_info.fps
                duration = video_info.duration
                total_frames = duration * fps

                print('encode [after video info]: {}sec'.format(time.time() - encode_start_time))

                command = [ '/usr/bin/ffmpeg',
                            '-r', str(fps), # frames per second
                            '-y',
                            '-loglevel', 'error',
                            '-f', 'rawvideo',
                            '-vcodec','rawvideo',
                            '-s', '1920x1080', # size of one frame
                            #'-s', '1280x720', # size of one frame
                            '-pix_fmt', 'rgb24',
                            '-i', '-', # The imput comes from a pipe
                            #'-s', '1920x1080', # size of one frame
                            '-vcodec', 'libx264',
                            #'crf', '0',
                            '-preset', 'ultrafast',
                            '-movflags', 'empty_moov+omit_tfhd_offset+frag_keyframe+default_base_moof',
                            '-pix_fmt', 'yuv420p',
                            #'-an', # Tells FFMPEG not to expect any audio
                            '{}'.format(os.path.join(process_dir, 'output.mp4'))]

                pipe = sub.Popen(command, stdin=sub.PIPE, stderr=sub.PIPE)
                end_time_ = time.time()
                print('encode [start]: {}sec'.format(end_time_ - encode_start_time))

            elif input[0] == 'frame':
                #start_time_ = time.time()
                idx = input[1]
                img = shared_tensor_list[1080][idx].cpu().numpy()

                if img is None:
                    print(idx)

                pipe.stdin.write(img.tobytes())
                pipe.stdin.flush()
                #end_time_ = time.time()
                #print('encode [frame]: {}sec'.format(end_time_ - start_time_))

            elif input[0] == 'end':
                start_time_ = time.time()
                pipe.stdin.flush()
                pipe.stdin.close()
                pipe = None
                output_input = input[1]
                #infer_idx = input[2]
                #infer_idx = -1 #TODO

                #print('encode [end] : {}sec'.format(end_time))
                encode_end_time = time.time()
                print('encode [end]: {}sec'.format(encode_end_time - start_time_))
                print('encode [elapsed] / index [{}]: {}sec'.format(infer_idx, encode_end_time - encode_start_time))

                output_input.send(('output', os.path.join(process_dir, 'output.mp4'), infer_idx))
                process_dir = None

            elif input[0] == 'dummy':
                print('dummy run')
                img = input[1].cpu()
                continue

            elif input[0] == 'index':
                infer_idx = input[1]

            else:
                print('encode: Invalid input')
                continue

            encode_queue.task_done()

        except (KeyboardInterrupt, SystemExit):
            print('exiting...')
            break

#Test for off-line : request [resolution]/[index].mp4
def request(decode_queue, resolution, index):
    res2bitrate = {240: 400, 360: 800, 480: 1200, 720: 2400, 1080: 4800}
    res2quality = {240: 0, 360: 1, 480: 2, 720: 3, 1080: 4}
    video_dir = os.path.join(opt.videoDir, '{}p'.format(resolution))
    #print('overall [start]: {}sec'.format(start_time))

    start_time = time.time()
    video_info = util.videoInfo(24.0, 4.0, res2quality[resolution])
    output_output, output_input = mp.Pipe(duplex=False)
    decode_queue.put((os.path.join(video_dir, 'segment_init.mp4'), os.path.join(video_dir, 'segment_{}.m4s'.format(index)), output_input, video_info))
    #decode_queue.put((os.path.join(video_dir, 'output_{}k_dashinit.mp4'.format(res2bitrate[resolution])), os.path.join(video_dir, 'output_{}k_dash{}.m4s'.format(res2bitrate[resolution], index)), output_input, video_info)) #temporary for old encoding formats

    while(1):
        input = output_output.recv()
        if input[0] == 'output':
            end_time = time.time()
            #print('overall [end] : {}sec'.format(end_time))
            print('overall [elapsed], resolution [{}p] : {}sec'.format(resolution, end_time - start_time))
            break
        else:
            print('request: Invalid input')
            break

def test():
    tensor = torch.FloatTensor(96, 3, 1920, 1080).random_(0,1)
    tensor = tensor.cuda()

    start_time = time.time()
    tensor_cpu = tensor.cpu()
    end_time = time.time()

    print('gpu2cpu: {}sec'.format(end_time - start_time))

def signal_handler(signal, frame):
    print('You pressed Ctrl+C!')
    sys.exit(0)

#Test for off-line : request [resolution]/[index].mp4
def test_figure16(clock, index=1):
    cudnn.benchmark = True
    signal.signal(signal.SIGINT, signal_handler)

    #Queue, Pipe
    decode_queue = mp.Queue()
    dnn_queue = mp.JoinableQueue()
    data_queue = mp.JoinableQueue()
    encode_queue = mp.JoinableQueue()
    output_output, output_input = mp.Pipe(duplex=False)

    #Shared tensor
    shared_tensor_list = {}
    res_list = [(270, 480), (360, 640), (540, 960), (1080, 1920)]
    for res in res_list:
        shared_tensor_list[res[0]] = []
        for _ in range(SHARED_QUEUE_LEN):
            shared_tensor_list[res[0]].append(torch.ByteTensor(res[0], res[1], 3).cuda().share_memory_())
            #shared_tensor_list[res[0]].append(torch.ByteTensor(res[0], res[1], 3).cuda().share_memory_())

    #Creat processes
    decode_process = mp.Process(target=decode, args=(decode_queue, encode_queue, data_queue, shared_tensor_list))
    sr_process = mp.Process(target=super_resolution_threading, args=(encode_queue, dnn_queue, data_queue, shared_tensor_list))
    encode_process = mp.Process(target=encode, args=(encode_queue, shared_tensor_list))

    dummy_process_list = []
    request_process_list = []

    #Configuration
    res2bitrate = {240: 400, 360: 800, 480: 1200, 720: 2400, 1080: 4800}
    res2quality = {240: 0, 360: 1, 480: 2, 720: 3, 1080: 4}
    #resolution_list = [240, 360, 480, 720]
    resolution_list = [240]
    segment_fps = 30
    segment_size = 4

    decode_process.start()
    sr_process.start()
    encode_process.start()

    print('=====dummy start=====')
    for resolution in resolution_list:
        video_dir = os.path.join(opt.videoDir, '{}p'.format(resolution))
        video_info = util.videoInfo(segment_fps, segment_size, res2quality[resolution])
        output_output, output_input = mp.Pipe(duplex=False)
        decode_queue.put((os.path.join(video_dir, 'segment_init.mp4'), os.path.join(video_dir, 'segment_{}.m4s'.format(index)), output_input, video_info))

        while(1):
            input = output_output.recv()
            if input[0] == 'output':
                break
            else:
                print('request: Invalid input')
                break
    print('=====dummy_end=====')

    #Iterate over xx times - report min/average/max
    elapsed_time_list = {}
    fps_list = {}

    output_list = [0,1,2,3,4]
    for output in output_list:
        elapsed_time_list[output] = {}
        fps_list[output] = {}
        for resolution in resolution_list:
            elapsed_time_list[output][resolution] = []
            fps_list[output][resolution] = []

    #Set inference index
    for _ in range(opt.runtimeNum):
        for output in output_list:
            dnn_queue.put(('test_figure16',output,))
            dnn_queue.join()
            for resolution in resolution_list:
                video_dir = os.path.join(opt.videoDir, '{}p'.format(resolution))
                video_info = util.videoInfo(segment_fps, segment_size, res2quality[resolution])
                output_output, output_input = mp.Pipe(duplex=False)
                start_time = time.time()
                decode_queue.put((os.path.join(video_dir, 'segment_init.mp4'), os.path.join(video_dir, 'segment_{}.m4s'.format(index)), output_input, video_info))
                while(1):
                    input = output_output.recv()
                    if input[0] == 'output':
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        fps = segment_fps * segment_size / (end_time - start_time)
                        print('overall [elapsed], resolution [{}p] : {} second, {} fps'.format(resolution, elapsed_time, fps))
                        elapsed_time_list[output][resolution].append(elapsed_time)
                        fps_list[output][resolution].append(fps)
                        break
                    else:
                        print('request: Invalid input')
                        break

    #Log
    os.makedirs('dnn_runtime_{}'.format(clock), exist_ok=True)
    runtimeLogger = util.getLogger('dnn_runtime_{}'.format(clock), opt.quality)

    #Print statistics
    for output in output_list:
        for resolution in resolution_list:
            print('[output: {}][{}p]: minmum {} fps, average {} fps, maximum {} fps'.format(output, resolution, np.min(fps_list[output][resolution]), np.average(fps_list[output][resolution]), np.max(fps_list[output][resolution])))
            log_str = "\t".join(map(str, fps_list[output][resolution]))
            log_str += "\t{}".format(np.average(fps_list[output][resolution]))
            runtimeLogger.info(log_str)

    sr_process.terminate()
    decode_process.terminate()
    encode_process.terminate()

#Test for off-line : request [resolution]/[index].mp4
def test_runtime(index=1):
    cudnn.benchmark = True
    signal.signal(signal.SIGINT, signal_handler)

    #Queue, Pipe
    decode_queue = mp.Queue()
    dnn_queue = mp.JoinableQueue()
    data_queue = mp.JoinableQueue()
    encode_queue = mp.JoinableQueue()
    output_output, output_input = mp.Pipe(duplex=False)

    #Shared tensor
    shared_tensor_list = {}
    res_list = [(270, 480), (360, 640), (540, 960), (1080, 1920)]
    for res in res_list:
        shared_tensor_list[res[0]] = []
        for _ in range(SHARED_QUEUE_LEN):
            shared_tensor_list[res[0]].append(torch.ByteTensor(res[0], res[1], 3).cuda().share_memory_())
            #shared_tensor_list[res[0]].append(torch.ByteTensor(res[0], res[1], 3).cuda().share_memory_())

    #Creat processes
    decode_process = mp.Process(target=decode, args=(decode_queue, encode_queue, data_queue, shared_tensor_list))
    sr_process = mp.Process(target=super_resolution_threading, args=(encode_queue, dnn_queue, data_queue, shared_tensor_list))
    encode_process = mp.Process(target=encode, args=(encode_queue, shared_tensor_list))

    dummy_process_list = []
    request_process_list = []

    #Configuration
    res2bitrate = {240: 400, 360: 800, 480: 1200, 720: 2400, 1080: 4800}
    res2quality = {240: 0, 360: 1, 480: 2, 720: 3, 1080: 4}
    #resolution_list = [240, 360, 480, 720]
    resolution_list = [240, 360, 480, 720]
    segment_fps = 30
    segment_size = 4

    decode_process.start()
    sr_process.start()
    encode_process.start()

    #Set inference index
    dnn_queue.put(('test_runtime',))
    time.sleep(1)

    print('=====dummy start=====')
    for resolution in resolution_list:
        video_dir = os.path.join(opt.videoDir, '{}p'.format(resolution))
        video_info = util.videoInfo(segment_fps, segment_size, res2quality[resolution])
        output_output, output_input = mp.Pipe(duplex=False)
        decode_queue.put((os.path.join(video_dir, 'segment_init.mp4'), os.path.join(video_dir, 'segment_{}.m4s'.format(index)), output_input, video_info))

        while(1):
            input = output_output.recv()
            if input[0] == 'output':
                break
            else:
                print('request: Invalid input')
                break
    print('=====dummy_end=====')

    #Iterate over xx times - report min/average/max
    elapsed_time_list = {}
    fps_list = {}

    for resolution in resolution_list:
        elapsed_time_list[resolution] = []
        fps_list[resolution] = []

    for _ in range(opt.runtimeNum):
        for resolution in resolution_list:
            video_dir = os.path.join(opt.videoDir, '{}p'.format(resolution))
            video_info = util.videoInfo(segment_fps, segment_size, res2quality[resolution])
            output_output, output_input = mp.Pipe(duplex=False)
            start_time = time.time()
            decode_queue.put((os.path.join(video_dir, 'segment_init.mp4'), os.path.join(video_dir, 'segment_{}.m4s'.format(index)), output_input, video_info))
            while(1):
                input = output_output.recv()
                if input[0] == 'output':
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    fps = segment_fps * segment_size / (end_time - start_time)
                    print('overall [elapsed], resolution [{}p] : {} second, {} fps'.format(resolution, elapsed_time, fps))
                    elapsed_time_list[resolution].append(elapsed_time)
                    fps_list[resolution].append(fps)
                    break
                else:
                    print('request: Invalid input')
                    break

    #Log
    os.makedirs('dnn_runtime', exist_ok=True)
    runtimeLogger = util.getLogger('dnn_runtime', opt.quality)

    #Print statistics
    for resolution in resolution_list:
        print('[{}p]: minmum {} fps, average {} fps, maximum {} fps'.format(resolution, np.min(fps_list[resolution]), np.average(fps_list[resolution]), np.max(fps_list[resolution])))
        log_str = "\t".join(map(str, fps_list[resolution]))
        log_str += "\t{}".format(np.average(fps_list[resolution]))
        runtimeLogger.info(log_str)

    sr_process.terminate()
    decode_process.terminate()
    encode_process.terminate()

def test_multi_resolution():
    cudnn.benchmark = True
    signal.signal(signal.SIGINT, signal_handler)

    #Queue, Pipe
    decode_queue = mp.Queue()
    dnn_queue = mp.JoinableQueue()
    data_queue = mp.JoinableQueue()
    encode_queue = mp.JoinableQueue()
    output_output, output_input = mp.Pipe(duplex=False)

    #Shared tensor
    shared_tensor_list = {}
    res_list = [(270, 480), (360, 640), (540, 960), (1080, 1920)]
    for res in res_list:
        shared_tensor_list[res[0]] = []
        for _ in range(SHARED_QUEUE_LEN):
            shared_tensor_list[res[0]].append(torch.ByteTensor(res[0], res[1], 3).cuda().share_memory_())
            #shared_tensor_list[res[0]].append(torch.ByteTensor(res[0], res[1], 3).cuda().share_memory_())

    #Creat processes
    decode_process = mp.Process(target=decode, args=(decode_queue, encode_queue, data_queue, shared_tensor_list))
    sr_process = mp.Process(target=super_resolution_threading, args=(encode_queue, dnn_queue, data_queue, shared_tensor_list))
    encode_process = mp.Process(target=encode, args=(encode_queue, shared_tensor_list))

    dummy_process_list = []
    request_process_list = []

    #Configuration
    res2bitrate = {240: 400, 360: 800, 480: 1200, 720: 2400, 1080: 4800}
    res2quality = {240: 0, 360: 1, 480: 2, 720: 3, 1080: 4}
    #resolution_list = [240, 360, 480, 720]
    resolution_list = [240, 360, 480, 720]
    segment_fps = 30
    segment_size = 4
    cudnn.benchmark = True
    signal.signal(signal.SIGINT, signal_handler)

    request_process_list = []

    print ('test start')
    sr_process.start()
    decode_process.start()
    encode_process.start()

    pretrained_path = os.path.join(opt.modelDir, 'epoch_%d.pth' % (opt.testEpoch))
    dnn_queue.put(('model', pretrained_path))
    dnn_queue.join()

    dnn_idx = 4
    chunk_idx = [6, 7, 8, 9, 10]
    for _ in range(1):
        dnn_queue.put(('test_figure16',dnn_idx,)) #output = 0,1,2,3,4 - 4 is a full model
        dnn_queue.join() # wait for done
        request_process_list.append(mp.Process(target=request, args=(decode_queue, 240, chunk_idx[dnn_idx])))
        dnn_idx += 1
        #request_process_list.append(mp.Process(target=request, args=(decode_queue, 360, 1)))
        #request_process_list.append(mp.Process(target=request, args=(decode_queue, 480, 1)))
        #request_process_list.append(mp.Process(target=request, args=(decode_queue, 720, 1)))



    print('============INFERENCE START==============')
    for request_process in request_process_list:
        request_process.start()
        request_process.join()
    print('============INFERENCE END==============')

    sr_process.terminate()
    decode_process.terminate()
    encode_process.terminate()

#TODO: partial model load
def test_scalable_dnn():
    cudnn.benchmark = True
    signal.signal(signal.SIGINT, signal_handler)

    #Queue, Pipe
    decode_queue = mp.Queue()
    dnn_queue = mp.Queue()
    process_output, process_input= mp.Pipe(duplex=False)
    encode_output, encode_input = mp.Pipe(duplex=False)
    output_output, output_input = mp.Pipe(duplex=False)

    #Shared tensor
    shared_tensor_list = []
    for _ in range(SHARED_QUEUE_LEN):
        shared_tensor_list.append(torch.ByteTensor(1080, 1920, 3).cuda().share_memory_())
        #shared_tensor_list.append(torch.ByteTensor(720, 1280, 3).cuda().share_memory_())

    #Lock
    mp_lock_list = []
    for _ in range(SHARED_QUEUE_LEN):
        mp_lock_list.append(mp.Lock())

    decode_process = mp.Process(target=decode, args=(decode_queue, process_input))
    sr_process = mp.Process(target=super_resolution_threading, args=(process_output, encode_input, dnn_queue, mp_lock_list, shared_tensor_list))
    #sr_process = mp.Process(target=super_resolution, args=(process_output, encode_input))
    encode_process = mp.Process(target=encode, args=(encode_output, mp_lock_list, shared_tensor_list))

    dummy_process_list = []
    request_process_list = []

    for _ in range(1):
        dummy_process_list.append(mp.Process(target=request, args=(decode_queue, 240, 1)))

    for _ in range(1):
        request_process_list.append(mp.Process(target=request, args=(decode_queue, 240, 1)))
        request_process_list.append(mp.Process(target=request, args=(decode_queue, 240, 2)))
        request_process_list.append(mp.Process(target=request, args=(decode_queue, 240, 3)))
        request_process_list.append(mp.Process(target=request, args=(decode_queue, 240, 4)))
        request_process_list.append(mp.Process(target=request, args=(decode_queue, 240, 5)))

    sr_process.start()
    decode_process.start()
    encode_process.start()

    print('============DUMMY START==============')
    for dummy_process in dummy_process_list:
        dummy_process.start()
        dummy_process.join()
    print('============DUMMY END==============')

    print('============INFERENCE START==============')
    count = 0
    for request_process in request_process_list:
        pretrained_path = os.path.join(opt.modelDir, 'DNN_chunk_{}.pth'.format(count))
        dnn_queue.put(('dnn_chunk', pretrained_path, count))
        time.sleep(0.5)
        request_process.start()
        request_process.join()
        count += 1
    print('============INFERENCE END==============')

#Assumption: 1) encode + decode overhead: prefixed constant (x sec) 2) resolution of which has the most overhead is known 3) 'feature' represets the number of body channels (~2x of pre-process and post-process)
#Test with low to ultra-high
#TODO: find 2) - currently using 1 / pass fps / duration

def run_dummy():
    cudnn.benchmark = True
    signal.signal(signal.SIGINT, signal_handler)

    decode_queue = mp.Queue()
    process_output, process_input= mp.Pipe(duplex=False)
    encode_output, encode_input = mp.Pipe(duplex=False)
    output_output, output_input = mp.Pipe(duplex=False)

    decode_process = mp.Process(target=decode, args=(decode_queue, process_input))
    sr_process = mp.Process(target=super_resolution, args=(process_output, encode_input))
    encode_process = mp.Process(target=encode, args=(encode_output, 1))

    dummy_process_list = []
    request_process_list = []

    for _ in range(1):
        dummy_process_list.append(mp.Process(target=request, args=(decode_queue, 240, 1)))
        dummy_process_list.append(mp.Process(target=request, args=(decode_queue, 360, 1)))
        dummy_process_list.append(mp.Process(target=request, args=(decode_queue, 480, 1)))
        dummy_process_list.append(mp.Process(target=request, args=(decode_queue, 720, 1)))

    sr_process.start()
    decode_process.start()
    encode_process.start()

    print('============DUMMY START==============')
    for dummy_process in dummy_process_list:
        dummy_process.start()
        dummy_process.join()
    print('============DUMMY END==============')

    decode_process.terminate()
    sr_process.terminate()
    encode_process.terminate()

def test_mock_DNNs():
    cudnn.benchmark = True
    signal.signal(signal.SIGINT, signal_handler)

    #Queue, Pipe
    decode_queue = mp.Queue()
    dnn_queue = mp.JoinableQueue()
    data_queue = mp.JoinableQueue()
    encode_queue = mp.JoinableQueue()
    output_output, output_input = mp.Pipe(duplex=False)
    test_output, test_input= mp.Pipe(duplex=False)

    #Shared tensor
    shared_tensor_list = {}
    res_list = [(270, 480), (360, 640), (540, 960), (1080, 1920)]
    for res in res_list:
        shared_tensor_list[res[0]] = []
        for _ in range(SHARED_QUEUE_LEN):
            shared_tensor_list[res[0]].append(torch.ByteTensor(res[0], res[1], 3).cuda().share_memory_())
            #shared_tensor_list[res[0]].append(torch.ByteTensor(res[0], res[1], 3).cuda().share_memory_())

    #Creat processes
    decode_process = mp.Process(target=decode, args=(decode_queue, encode_queue, data_queue, shared_tensor_list))
    sr_process = mp.Process(target=super_resolution_threading, args=(encode_queue, dnn_queue, data_queue, shared_tensor_list))
    encode_process = mp.Process(target=encode, args=(encode_queue, shared_tensor_list))

    dummy_process_list = []
    request_process_list = []

    dummy_process_list = []
    request_process_list = []

    for _ in range(1):
        dummy_process_list.append(mp.Process(target=request, args=(decode_queue, 240, 1)))
        dummy_process_list.append(mp.Process(target=request, args=(decode_queue, 360, 1)))
        dummy_process_list.append(mp.Process(target=request, args=(decode_queue, 480, 1)))
        dummy_process_list.append(mp.Process(target=request, args=(decode_queue, 720, 1)))

    sr_process.start()
    decode_process.start()
    encode_process.start()

    """
    print('============DUMMY START==============')
    for dummy_process in dummy_process_list:
        dummy_process.start()
        dummy_process.join()
    print('============DUMMY END==============')
    """

    print('============TEST START==============')
    DNN_list = [(18, 9), (18, 21), (18, 32), (18, 48)]
    fps = 30
    duration = 4
    dnn_queue.put(('test_dnn', DNN_list, fps, duration, test_input))

    input = test_output.recv()
    print('Selected DNN index is {}'.format(input[0]))
    print('============TEST END==============')

    decode_process.terminate()
    sr_process.terminate()
    encode_process.terminate()

if __name__ == "__main__":
    mp.set_start_method('spawn')
    torch.multiprocessing.set_sharing_strategy('file_descriptor')

    if opt.processMode == 'multi':
        test_multi_resolution()
    elif opt.processMode == 'scalable':
        test_scalable_dnn()
    elif opt.processMode == 'mock':
        test_mock_DNNs()
    elif opt.processMode == 'runtime':
        test_runtime()
    elif opt.processMode == 'figure16':
        CLOCK_INFO = {}
        #TITAN_XP_INFO = [1404, 1303, 1202, 1101, 999, 898, 797, 696, 506]
        #TITAN_XP_INFO = [1404, 1303, 1202, 1101, 999, 898, 797, 696]
        TITAN_XP_INFO = [949]
        TITAN_XP_INFO.reverse()
        CLOCK_INFO['titanxp'] = TITAN_XP_INFO

        for clock in CLOCK_INFO['titanxp']:
            os.system('echo ina8024 | sudo -S nvidia-smi -i 0 --applications-clocks=5705,{}'.format(clock))
            test_figure16(clock)
        os.system('echo ina8024 nvidia-smi -i 0 --reset-applications-clocks')

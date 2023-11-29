import json
import flask
import os
import signal
from io import StringIO
from werkzeug.utils import secure_filename
from flask import Flask, make_response, render_template, redirect, url_for, request, send_file
import subprocess
import sys
sys.path.insert(0, '../super_resolution')

from option import *
#import process_ver2
import process as process
import utility as util
import common
#import torch.multiprocessing as mp
import torch.multiprocessing as mp
import time
import torch


#resolution = ['1280x720', '1280x720','640x480', '640x480', '320x180', '320x180']

DOWNLOAD_PATH = 'downloads'
VIDEO_NAME = 'MediaSegment'
INIT_NAME = 'InitializationSegment'

# TODO: mock DNN test
"""
def test_DNN():
"""

app = Flask(__name__)  # Flask


@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    data = json.loads(request.form['jsondata'])
    #print('fps!!! :' + str(data['fps']))
    videodata = request.files['videofile'].read()
    quality = str(data['quality'])
    index = str(data['index'])
    # TODO: get resolution & framenumber

    file_path = os.path.join(os.getcwd(), DOWNLOAD_PATH,
                             quality + '_' + index + '_' + data['segmentType'])
    video_chunk_path = os.path.join(
        os.getcwd(), DOWNLOAD_PATH, quality + '_' + index + '_' + VIDEO_NAME)
    init_path = os.path.join(os.getcwd(), DOWNLOAD_PATH,
                             quality + '_None_' + INIT_NAME)

    # Download a file - TODO: measure time
    newfile = open(file_path, "wb")
    newfile.write(videodata)
    newfile.close()

    print('dnn_server (input): {}'.format(
        quality + '_' + index + '_' + data['segmentType']))

    # TODO: remove 'and' - just for debugging
    if str(data['segmentType']) == 'MediaSegment' and int(index) > 0:
        try:
            output_input, output_output = mp.Pipe()
            video_info = util.videoInfo(float(data['fps']), float(
                data['duration']), int(data['quality']))
            #print(video_info.fps)
            #print(video_info.duration)
            #print(video_info.quality)
            #app.config['dnn_log'].write('video recv: ' + str(index) + '\n')
            start_time = time.time()
            app.config['decode_queue'].put(
                (init_path, video_chunk_path, output_input, video_info))

            while(1):
                input = output_output.recv()

                if input[0] == 'output':
                    # change base media decode time #TODO: put into process.py encode()
                    print('output video: {}'.format(input[1]))
                    cmd = 'python set_hdr.py {} {}'.format(input[1], index)
                    print(cmd)
                    os.system(cmd)

                    cmd = 'cat {} >> {}'.format(init_path, input[1])
                    os.system(cmd)

                    #swap_file = input[1]
                    swap_file = os.path.join(os.getcwd(), input[1])


                    end_time = time.time()
                    print('{} is done'.format(input[1]))
                    print('thread [elapsed]: {}sec'.format(
                        end_time - start_time))

                    infer_idx = input[2]
                    if quality != '3':
                        infer_idx = int(infer_idx/2)

                    app.config['dnn_log'].write(str(input[1]) + ' thread [elapsed]: ' + str(end_time - start_time) + ' sec inference_idx: ' + str(infer_idx) +'\n')
                    app.config['dnn_log'].flush()
                    break
                else:
                    print('thread: Invalid income')
                    break

        except KeyboardInterrupt:
            resp = flask.Response('hello')
            resp.headers['Access-Control-Allow-Origin'] = '*'
            return resp

        filename = str(index) + ',' + str(infer_idx)
        resp = send_file(swap_file, download_name=filename,
                         mimetype='application/octet-stream', as_attachment=True)
        response = make_response(resp)
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Expose-Headers'] = 'Content-Disposition'
        response.headers['Cache-Control'] = 'no-cache, no-store'
        return response
    else:
        resp = flask.Response('Init')
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp


@app.route('/swap_log', methods=['GET', 'POST'])
def write_log():
    log = json.loads(request.data)
    print('\nswap_log: start-' +
          str(log['start']) + '\t' + 'time-' + str(log['time']) + '\n')
    resp = flask.Response('Init')
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp

@app.route('/dnn_chunk', methods=['GET', 'POST'])
def notify_path():
    data = json.loads(request.form['jsondata'])
    #print('fps!!! :' + str(data['fps']))
    dnn_chunk = request.files['dnn'].read()

    dnn_path = os.path.join(os.getcwd(), DOWNLOAD_PATH, 'DNN_chunk_{}.pth'.format(data['chunk_num']))

    newfile = open(dnn_path, "wb")
    newfile.write(dnn_chunk)

    #path = json.loads(data)
    print(str(data))
    app.config['dnn_queue'].put(('dnn_chunk', dnn_path, int(data['chunk_num']) - 1))
    app.config['dnn_log'].write('dnn_chunk recv: ' + str(data['chunk_num']) + '\n')
    resp = flask.Response('Init')
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp

@app.route('/dnn_config', methods=['GET', 'POST'])
def config():
    global fps
    #Representation index : 0-low, 1-medium, 2-high, 3-ultra, 4-single(whole)
    #Quality index : 0-240p, 1-360p, 2-480p, 3-720p, 4-1080p
    config = json.loads(request.data)
    fps = int(config['frameRate']['fps'])
    config_start = time.time()

    #process mock DNNs test
    DNN_list = []
    for idx in range(4):
        dnn_quality = config['Representation'][idx]['Quality']
        DNN_list.append((int(dnn_quality[0]['layer']), int(dnn_quality[0]['feature'])))

    print (DNN_list)
    print ('fps returned is {}'.format(fps))
    duration = 4
    app.config['dnn_queue'].put(('test_dnn', DNN_list, fps, duration, test_input))

    input = test_output.recv()
    print('Selected DNN index is {}'.format(input[0]))
    dnn_selection = input[0]
    #dnn_selection = 2
    print ('config eplased:' + str(time.time() - config_start))

    resp = flask.Response('DNN Quality select: ' + str(dnn_selection))
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp


@app.route('/')
def index():
    return render_template('index.html')

"""
Total process: 4
1 decode process
1 super-resolution process : use multiple threads to load a data
2 encoding preocess

Input: a raw video
Output: a enhanced video
"""

if __name__ == '__main__':
    mp.set_start_method('spawn')
    torch.multiprocessing.set_sharing_strategy('file_descriptor')

    process.cudnn.benchmark = True
    signal.signal(signal.SIGINT, process.signal_handler)

    #Queue, Pipe
    decode_queue = mp.Queue()
    dnn_queue = mp.JoinableQueue()
    data_queue = mp.JoinableQueue()
    encode_queue = mp.JoinableQueue()
    output_output, output_input = mp.Pipe(duplex=False)
    test_output, test_input = mp.Pipe(duplex=False)

    #Shared tensor
    shared_tensor_list = {}
    res_list = [(270, 480), (360, 640), (540, 960), (1080, 1920)]
    for res in res_list:
        shared_tensor_list[res[0]] = []
        for _ in range(process.SHARED_QUEUE_LEN):
            shared_tensor_list[res[0]].append(torch.ByteTensor(res[0], res[1], 3).cuda().share_memory_())
            #shared_tensor_list[res[0]].append(torch.ByteTensor(res[0], res[1], 3).cuda().share_memory_())


    #Creat processes
    decode_process = mp.Process(target=process.decode, args=(decode_queue, encode_queue, data_queue, shared_tensor_list))
    sr_process = mp.Process(target=process.super_resolution_threading, args=(encode_queue, dnn_queue, data_queue, shared_tensor_list))
    encode_process = mp.Process(target=process.encode, args=(encode_queue, shared_tensor_list))

    sr_process.start()
    decode_process.start()
    encode_process.start()

    pretrained_path = os.path.join(opt.modelDir, 'epoch_%d.pth' % (opt.testEpoch))

    # Model upload: for pre-load DNN chunks
    """
    dnn_queue.put(('model', pretrained_path))
    time.sleep(5)

    # Run dummy jobs
    print('============DUMMY START==============')
    for dummy_process in dummy_process_list:
        dummy_process.start()
        dummy_process.join()
    print('============DUMMY END==============')
    """

    # Setup
    os.makedirs(DOWNLOAD_PATH, exist_ok=True)

    dnn_log = open('./dnn_log', "w")

    # Run flask
    app.config['decode_queue'] = decode_queue
    app.config['dnn_queue'] = dnn_queue
    app.config['dnn_log'] = dnn_log
    app.run(host='0.0.0.0', threaded=True, debug=True, use_reloader=False)#, ssl_context='adhoc')

    # Join other processes
    sr_process.join()
    decode_process.join()
    encode_process.join()

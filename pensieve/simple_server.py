#!/usr/bin/env python
from http.server import BaseHTTPRequestHandler, HTTPServer
import socketserver
import base64
import urllib
import sys
import os
import json
import time
os.environ['CUDA_VISIBLE_DEVICES']=''

import numpy as np
import time
import itertools
from option import *
# import utility as util
import template
import time
######################## FAST MPC #######################

S_INFO = 5  # bit_rate, buffer_size, rebuffering_time, bandwidth_measurement, chunk_til_video_end
S_LEN = 8  # take how many frames in the past
MPC_FUTURE_CHUNK_COUNT = 5
RANDOM_SEED = 42
RAND_RANGE = 1000
SUMMARY_DIR = './results'
LOG_FILE = 'log_default'
# in format of time_stamp bit_rate buffer_size rebuffer_time video_chunk_size download_time reward
NN_MODEL = None
LOG_FILE_DEFAULT = 'log'
LOG_FILE_SR_QUALITY = 'log_sr_'+opt.reward+'_'
LOG_FILE_SR_AWARE = 'log_sr_aware_'+opt.reward+'_'


global video_size

past_errors = []
past_bandwidth_ests = []

dummy = {0: 197337, 1: 394881, 2: 844676, 3: 972253, 4: 2034051, 5: 844676}


def get_chunk_size(quality, index):
    #if ( index < 0 or index > 48 ):
    #    return 0
    # note that the quality and video labels are inverted (i.e., quality 8 is highest and this pertains to video1)
    
    #sizes = {5: size_video1[index], 4: size_video2[index], 3: size_video3[index], 2: size_video4[index], 1: size_video5[index], 0: size_video6[index]}
    global video_size 
    return dummy[quality] #jh TODO: fix this
        
    return video_size[quality][index]

def create_logfile(sr_quality, sr_aware, name):
        parse=name.split('_')
        algo=parse[0]
        name='_'.join(parse[1:])
        if sr_quality:
            log_path = os.path.join(opt.train_logfile[:-3], '{}_{}'.format(LOG_FILE_SR_QUALITY, name))
        elif sr_aware:
            log_path = os.path.join(opt.train_logfile[:-3], '{}_{}'.format(LOG_FILE_SR_AWARE, name))
        else:
            #log_path = os.path.join(opt.train_logfile[:-3], '{}_{}_{}_{}_'.format(LOG_FILE_DEFAULT, name,opt.content,opt.quality))
            
            log_path = os.path.join(opt.realworld_path, '{}_{}_{}_{}_PARSEHERE{}'.format(LOG_FILE_DEFAULT,algo,opt.reward,opt.content, name))
        print(log_path)
        log_file = open(log_path, 'w')
            
        return log_file
                        
def get_index_from_size(size):
    global video_size
    
    for key,list_ in video_size.items():
        if size in list_:
            return key
    
    return -1#error not found
def make_request_handler(input_dict):

    class Request_Handler(BaseHTTPRequestHandler):
        exp_finished=0
        def __init__(self, *args, **kwargs):
            self.input_dict = input_dict
            self.log_file = input_dict['log_file']
            
            #self.saver = input_dict['saver']
            self.s_batch = input_dict['s_batch']
            #self.a_batch = input_dict['a_batch']
            #self.r_batch = input_dict['r_batch']
            BaseHTTPRequestHandler.__init__(self, *args, **kwargs)

        def do_POST(self):
            content_length = int(self.headers['Content-Length'])
            post_data = json.loads(self.rfile.read(content_length))
            print (post_data)
            post_data_f=open('./post_log/simple_post_data_'+opt.run_vid_trace,'a')
            post_data_f.write(str(post_data['lastquality']))
            for i in post_data:
                post_data_f.write(i+': '+str(post_data[i])+'\t')
            post_data_f.write('\n')
            post_data_f.flush()
            try:
                print(post_data['finished'])
            except:
                post_data['finished']=0


            try:
                rebuffer_time = float(post_data['RebufferTime'] -self.input_dict['last_total_rebuf'])
                time_stamp=float(time.time())*1000
                #self.input_dict['last_bit_rate']= state_bitrate[post_data['lastquality']]
                self.input_dict['last_total_rebuf'] = post_data['RebufferTime']
                # compute bandwidth measurement
                video_chunk_fetch_time = post_data['lastChunkFinishTime'] - post_data['lastChunkStartTime']
                video_chunk_size = post_data['lastChunkSize']
                video_chunk_remain = opt.total_chunk - self.input_dict['video_chunk_count']
                self.input_dict['video_chunk_count'] += 1
                #video_resolution=get_index_from_size(video_chunk_size)
            except Exception as e:
                print('ERROR IN BLOCK 1 '+str(e)+'')
            # log wall_time, bit_rate, buffer_size, rebuffer_time, video_chunk_size, download_time, reward
            '''
            if video_resolution ==-1:
                post_data_f.write('VIDEO SIZE NOT FOUND\n')
                post_data_f.flush()
                self.log_file.write('VIDEO SIZE NOT FOUND\n')
                self.log_file.flush()
            '''
            if Request_Handler.exp_finished!=1:
                self.log_file.write(
                    str(time_stamp / M_IN_K) + '\t' +
                    str(opt.bitrate[post_data['lastquality']]) + '\t' +
                    str(post_data['buffer'] ) + '\t' +
                    str(rebuffer_time/M_IN_K) + '\t' +
                    str(video_chunk_size) + '\t' +
                    str(video_chunk_fetch_time) + '\t' +
                    str(int(post_data['finished'])) + '\n'
                )
                
                self.log_file.flush()
                
            # pick bitrate according to MPC           
            # first get harmonic mean of last 5 bandwidths
            send_data=''
            end_of_video = False
            if ( post_data['lastRequest'] +1 >= opt.total_chunk ):
                send_data = "REFRESH"
                #stop when video download ends
                end_of_video = True
                Request_Handler.exp_finished=1
                self.input_dict['last_total_rebuf'] = 0
                self.input_dict['last_bit_rate'] = opt.default_quality#DEFAULT_QUALITY
                self.input_dict['video_chunk_count'] = 0
                #self.log_file.write('\n')  # so that in the log we know where video ends
                #self.log_file.flush()
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain')
            self.send_header('Content-Length', len(send_data))
            self.send_header('Access-Control-Allow-Origin', "*")
            self.end_headers()
            self.wfile.write(send_data.encode('utf-8'))
            
        def do_GET(self):
            #print >> sys.stderr, 'GOT REQ'
            self.send_response(200)
            #self.send_header('Cache-Control', 'Cache-Control: no-cache, no-store, must-revalidate max-age=0')
            self.send_header('Cache-Control', 'max-age=3000')
            self.send_header('Content-Length', 20)
            self.end_headers()
            self.wfile.write("console.log('here');".encode('utf-8'))
        def log_message(self, format, *args):
            return

    return Request_Handler


def run(server_class=HTTPServer, port=8333, log_file_path=LOG_FILE):

    np.random.seed(RANDOM_SEED)

    if not os.path.exists(SUMMARY_DIR):
        os.makedirs(SUMMARY_DIR)

    

    #with open(log_file_path, 'wb') as log_file:
    log_file = create_logfile(opt.sr_quality, opt.sr_aware, log_file_path)
    s_batch = [np.zeros((S_INFO, S_LEN))]
    
    last_bit_rate = opt.default_quality#DEFAULT_QUALITY
    last_total_rebuf = 0
    # need this storage, because observation only contains total rebuffering time
    # we compute the difference to get

    video_chunk_count = 0
    
    input_dict = {'log_file': log_file,
                  'last_bit_rate': last_bit_rate,
                  'last_total_rebuf': last_total_rebuf,
                  'video_chunk_count': video_chunk_count,
                  's_batch': s_batch}
    global video_size
    #jh TODO: fix this
    # video_size = template.get_video_sizes(opt.video_dir, opt.content, opt.total_chunk)
    #print(video_size)
    # interface to abr_rl server
    handler_class = make_request_handler(input_dict=input_dict)
    
    server_address = ('0.0.0.0', port)
    httpd = server_class(server_address, handler_class)
    print ('Listening on port ' + str(port))
    httpd.serve_forever()


def main():
    
    trace_file = opt.run_vid_trace
    run(log_file_path=trace_file)
    



if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Keyboard interrupted.")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

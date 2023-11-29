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
LOG_FILE_DEFAULT = 'log_mpc'
LOG_FILE_SR_QUALITY = 'log_sr_'+opt.reward+'_quality_mpc'
LOG_FILE_SR_AWARE = 'log_sr_aware_'+opt.reward+'_mpc'
CHUNK_COMBO_OPTIONS = []

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
        if sr_quality or opt.dnn_mode ==3:
            log_path = os.path.join(opt.realworld_path, '{}_{}'.format(LOG_FILE_SR_QUALITY, name))
        elif sr_aware:
            log_path = os.path.join(opt.realworld_path, '{}_{}'.format(LOG_FILE_SR_AWARE, name))
        else:
            log_path = os.path.join(opt.realworld_path, '{}_{}_{}_PARSEHERE{}'.format(LOG_FILE_DEFAULT,opt.reward, opt.content , name))
        log_file = open(log_path, 'w')
            
        return log_file
                        


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
            post_data_f=open('./post_log/mpc_post_data_'+opt.run_vid_trace,'a')
            post_data_f.write(str(post_data['lastquality']))
            for i in post_data:
                post_data_f.write(i+': '+str(post_data[i])+'\t')
            post_data_f.write('\n')
            post_data_f.flush()
            if ( 'pastThroughput' in post_data ):
                # @Hongzi: this is just the summary of throughput/quality at the end of the load
                # so we don't want to use this information to send back a new quality
                print ("Summary: ", post_data)
            else:
                try:
                    # option 1. reward for just quality
                    # reward = post_data['lastquality']
                    # option 2. combine reward for quality and rebuffer time
                    #           tune up the knob on rebuf to prevent it more
                    # reward = post_data['lastquality'] - 0.1 * (post_data['RebufferTime'] - self.input_dict['last_total_rebuf'])
                    # option 3. give a fixed penalty if video is stalled
                    #           this can reduce the variance in reward signal
                    # reward = post_data['lastquality'] - 10 * ((post_data['RebufferTime'] - self.input_dict['last_total_rebuf']) > 0)
                    
                    # option 4. use the metric in SIGCOMM MPC paper
                    try:
                        print(post_data['finished'])
                    except:
                        post_data['finished']=0
                    try:
                        time_stamp=float(time.time())*1000

                        if opt.sr_quality or opt.dnn_mode ==3:
                            assert sr_bitrate != None
                            assert len(sr_bitrate) == len(opt.bitrate)
                            sr_bitrate = template.get_partial_sr_quality(len(template.get_dnn_chunk_size(opt.quality)),opt.content,opt.quality)
                            
                            reward_bitrate = sr_bitrate
                            state_bitrate = sr_bitrate
                        elif opt.sr_aware:
                            assert sr_bitrate !=None
                            assert len(sr_bitrate) == len(opt.bitrate)
                            sr_bitrate = template.get_partial_sr_quality(len(template.get_dnn_chunk_size(opt.quality)),opt.content,opt.quality)
                            
                            reward_bitrate = sr_bitrate
                            state_bitrate = sr_bitrate
                        else:
                            reward_bitrate = opt.bitrate
                            state_bitrate = opt.bitrate
                    

                        rebuffer_time = float(post_data['RebufferTime'] -self.input_dict['last_total_rebuf'])
                
                
                        self.input_dict['last_bit_rate']= state_bitrate[post_data['lastquality']]
                        self.input_dict['last_total_rebuf'] = post_data['RebufferTime']
                    
                        # retrieve previous state
                        if len(self.s_batch) == 0:
                            state = [np.zeros((S_INFO, S_LEN))]
                        else:
                            state = np.array(self.s_batch[-1], copy=True)
                            
                        # compute bandwidth measurement
                        video_chunk_fetch_time = post_data['lastChunkFinishTime'] - post_data['lastChunkStartTime']
                        video_chunk_size = post_data['lastChunkSize']
                            
                        # compute number of video chunks left
                        video_chunk_remain = opt.total_chunk - self.input_dict['video_chunk_count']
                        self.input_dict['video_chunk_count'] += 1
                    except Exception as e:
                        post_data_f.write('MPC_SERVER ERROR in BLOCK 1 '+str(e)+'\n')
                        post_data_f.flush()
                        print(e)
                        exit(1)
                    # dequeue history record
                    state = np.roll(state, -1, axis=1)

                    # this should be S_INFO number of terms
                    try:
                        state[0, -1] = state_bitrate[post_data['lastquality']] / float(np.max(state_bitrate))
                        state[1, -1] = post_data['buffer'] / BUFFER_NORM_FACTOR
                        state[2, -1] = rebuffer_time / M_IN_K
                        state[3, -1] = float(video_chunk_size) / float(video_chunk_fetch_time) / M_IN_K  # kilo byte / ms
                        state[4, -1] = np.minimum(video_chunk_remain,opt.total_chunk) / float(opt.total_chunk)
                        curr_error = 0
                        if ( len(past_bandwidth_ests) > 0 ):
                            curr_error  = abs(past_bandwidth_ests[-1]-state[3,-1])/float(state[3,-1])
                        past_errors.append(curr_error)
                    except ZeroDivisionError:
                        # this should occur VERY rarely (1 out of 3000), should be a dash issue

                        # in this case we ignore the observation and roll back to an eariler one
                        past_errors.append(0)
                        if len(self.s_batch) == 0:
                            state = [np.zeros((S_INFO, S_LEN))]
                        else:
                            state = np.array(self.s_batch[-1], copy=True)




                    # log wall_time, bit_rate, buffer_size, rebuffer_time, video_chunk_size, download_time, reward
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
                        past_bandwidths = state[3,-5:]
                        while past_bandwidths[0] == 0.0:
                            past_bandwidths = past_bandwidths[1:]
                        #if ( len(state) < 5 ):
                        #    past_bandwidths = state[3,-len(state):]
                        #else:
                        #    past_bandwidths = state[3,-5:]
                        bandwidth_sum = 0
                        for past_val in past_bandwidths:
                            bandwidth_sum += (1/float(past_val))
                        harmonic_bandwidth = 1.0/(bandwidth_sum/len(past_bandwidths))



                        # future bandwidth prediction
                        
                        max_error = 0
                        error_pos = -5
                        if ( len(past_errors) < 5 ):
                            error_pos = -len(past_errors)
                        max_error = float(max(past_errors[error_pos:]))
                        future_bandwidth = harmonic_bandwidth/(1+max_error)
                        past_bandwidth_ests.append(harmonic_bandwidth)

                
                
                        # future chunks length (try 4 if that many remaining)
                        last_index = int(post_data['lastRequest'])
                        future_chunk_length = MPC_FUTURE_CHUNK_COUNT
                        if ( opt.total_chunk-1 - last_index < 5 ):
                            future_chunk_length = opt.total_chunk-1 - last_index

                        # all possible combinations of 5 chunk bitrates (9^5 options)
                        # iterate over list and for each, compute reward and store max reward combination
                        max_reward = -100000000
                        best_combo = ()
                        start_buffer = float(post_data['buffer'])
                        #start = time.time()
                        for full_combo in CHUNK_COMBO_OPTIONS:
                            combo = full_combo[0:future_chunk_length]
                            # calculate total rebuffer time for this combination (start with start_buffer and subtract
                            # each download time and add 2 seconds in that order)
                            curr_rebuffer_time = 0
                            curr_buffer = start_buffer
                            bitrate_sum = 0
                            smoothness_diffs = 0
                            last_quality = int(post_data['lastquality'])
                            for position in range(0, len(combo)):
                                chunk_quality = combo[position]
                                index = last_index + position + 1 # e.g., if last chunk is 3, then first iter is 3+0+1=4
                                download_time = (get_chunk_size(chunk_quality, index)/1000000.)/future_bandwidth # this is MB/MB/s --> seconds
                                if ( curr_buffer < download_time ):
                                    curr_rebuffer_time += (download_time - curr_buffer)
                                    curr_buffer = 0
                                else:
                                    curr_buffer -= download_time
                                curr_buffer += 4
                        
                                # linear reward
                                #bitrate_sum += VIDEO_BIT_RATE[chunk_quality]
                                #smoothness_diffs += abs(VIDEO_BIT_RATE[chunk_quality] - VIDEO_BIT_RATE[last_quality])
                                
                                # log reward
                                # log_bit_rate = np.log(VIDEO_BIT_RATE[chunk_quality] / float(VIDEO_BIT_RATE[0]))
                                # log_last_bit_rate = np.log(VIDEO_BIT_RATE[last_quality] / float(VIDEO_BIT_RATE[0]))
                                # bitrate_sum += log_bit_rate
                                # smoothness_diffs += abs(log_bit_rate - log_last_bit_rate)
                                
                                # hd reward

                                if opt.reward == 'linear':
                                    current_bitrate = state_bitrate[chunk_quality]
                                    last_bitrate = state_bitrate[last_quality]
                                elif opt.reward == 'log':
                                    """
                                    current_bitrate= np.log(state_bitrate[chunk_quality] / float(state_bitrate[0]))
                                    last_bitrate = np.log(state_bitrate[last_quality] / float(state_bitrate[0]))
                                    """
                                    current_bitrate= np.log(state_bitrate[chunk_quality] / float(opt.bitrate[0]))
                                    last_bitrate = np.log(state_bitrate[last_quality] / float(opt.bitrate[0]))
                                elif opt.reward == 'hd':
                                    current_bitrate = util.bitrate_to_hdreward(state_bitrate[chunk_quality])
                                    last_bitrate = util.bitrate_to_hdreward(state_bitrate[last_quality])
                                    
                                bitrate_sum += current_bitrate#BITRATE_REWARD[chunk_quality]
                                smoothness_diffs += abs(current_bitrate - last_bitrate)#abs(BITRATE_REWARD[chunk_quality] - BITRATE_REWARD[last_quality])
                        
                                last_quality = chunk_quality
                            # compute reward for this combination (one reward per 5-chunk combo)
                            # bitrates are in Mbits/s, rebuffer in seconds, and smoothness_diffs in Mbits/s
                            
                            # linear reward 
                            #reward = (bitrate_sum/1000.) - (4.3*curr_rebuffer_time) - (smoothness_diffs/1000.)
                            
                            # log reward
                            # reward = (bitrate_sum) - (4.3*curr_rebuffer_time) - (smoothness_diffs)
                            
                            # hd reward
                            if opt.reward == 'linear':
                                reward = (bitrate_sum / M_IN_K) - (opt.linear_rebuf * curr_rebuffer_time) - opt.smooth * (smoothness_diffs / M_IN_K)
                            elif opt.reward == 'log':
                                reward = (bitrate_sum) - (opt.log_rebuf * curr_rebuffer_time) - opt.smooth * (smoothness_diffs)
                            elif opt.reward == 'hd':
                                reward = (bitrate_sum) - (opt.hd_rebuf * curr_rebuffer_time) -  opt.smooth * (smoothness_diffs)
                            else:
                                exit(-1)
                    
                    
                            
                            if ( reward > max_reward ):
                                max_reward = reward
                                best_combo = combo
                        # send data to html side (first chunk of best combo)
                        send_data = str(0) # no combo had reward better than -1000000 (ERROR) so send 0
                        if ( best_combo != () ): # some combo was good
                            send_data = str(best_combo[0])

                        end = time.time()
                        #print "TOOK: " + str(end-start)

                        end_of_video = False
                        if ( post_data['lastRequest'] +1 >= opt.total_chunk ):
                            send_data = "REFRESH"
                            #stop when video download ends
                            end_of_video = True
                            Request_Handler.exp_finished=1
                            self.input_dict['last_total_rebuf'] = 0
                            self.input_dict['last_bit_rate'] = opt.default_quality#DEFAULT_QUALITY
                            self.input_dict['video_chunk_count'] = 0
                            self.log_file.write('\n')  # so that in the log we know where video ends
                            self.log_file.flush()
                        self.send_response(200)
                        self.send_header('Content-Type', 'text/plain')
                        self.send_header('Content-Length', len(send_data))
                        self.send_header('Access-Control-Allow-Origin', "*")
                        self.end_headers()
                        self.wfile.write(send_data.encode('utf-8'))

                        # record [state, action, reward]
                        # put it here after training, notice there is a shift in reward storage
                        
                        if end_of_video:
                            self.s_batch = [np.zeros((S_INFO, S_LEN))]
                        else:
                            self.s_batch.append(state)
                except Exception as e:
                    post_data_f.write('MPC SERVER ERROR : '+str(e))
                    post_data_f.flush()
                    exit(-1)
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

    # make chunk combination options
    for combo in itertools.product(list(range(0,len(opt.bitrate))), repeat=5):
        CHUNK_COMBO_OPTIONS.append(combo)

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

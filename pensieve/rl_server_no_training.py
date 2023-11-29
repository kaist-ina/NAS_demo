#!/usr/bin/env python
from http.server import BaseHTTPRequestHandler, HTTPServer
import socketserver
import base64
import urllib
import sys
import os
import json
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import time
import a3c
import template
from option import *
import ssl


DNN_TEST_NUM=float(np.sum(template.get_dnn_chunk_size(opt.quality))*1000)


A_DIM = len(opt.bitrate)+opt.dnn_mode
SUMMARY_DIR = opt.summary_dir
#LOG_FILE = opt.train_logfile
#180710 youngmok fixed the path
if opt.dnn_mode ==1:
    #LOG_FILE =os.path.join(opt.data_dir,opt.content, opt.quality,"realworld_"+opt.reward)
    LOG_FILE=opt.realworld_path
else:
    #LOG_FILE=os.path.join(opt.data_dir,opt.content,"realworld_"+opt.reward)
    LOG_FILE=opt.realworld_path


os.makedirs(LOG_FILE,exist_ok=True)
VIDEO_SIZE_FILE = './video_size_'


# in format of time_stamp bit_rate buffer_size rebuffer_time video_chunk_size download_time reward
# NN_MODEL = None
print(opt.model_dir)
NN_MODEL = os.path.join(opt.model_dir,opt.model)


#global time_stamp, dnn_chunk_remain
#time_stamp=0
#dnn_chunk_remain=opt.dnn_total_size
video_sizes={}

dummy = {0: 197337, 1: 394881, 2: 844676, 3: 972253, 4: 2034051, 5: 844676}



def get_chunk_size(quality, index):
    #if (index < 0 or index > opt.total_chunk - 1):
    # 	return 0
    # note that the quality and video labels are inverted (i.e., quality 8 is highest and this pertains to video1)
    global video_sizes
    #sizes = {5: opt.dnn_byte_size, 4: size_video2[index], 3: size_video3[index], 2: size_video4[index],
    #1: size_video5[index], 0: size_video6[index]}
    return dummy[quality] #jh TODO: fix this
    return video_sizes[quality][index]
    #return sizes[quality]


def make_request_handler(input_dict):
    class Request_Handler(BaseHTTPRequestHandler):
        dnn_chunk_remain=np.sum(template.get_dnn_chunk_size(opt.quality))*1000
        time_stamp=float(time.time())*1000
        last_end_of_dnn=0
        start_dnn_download=0
        send_dnn_baseline=0
        exp_finished=0
        def __init__(self, *args, **kwargs):
            self.input_dict = input_dict
            self.sess = input_dict['sess']
            
            self.log_file = input_dict['log_file']
            
            # file for hd and log
            # self.loglog_file = input_dict['loglog_file']
            # self.loghd_file = input_dict['loghd_file']
            self.actor = input_dict['actor']
            self.critic = input_dict['critic']
            self.saver = input_dict['saver']
            self.s_batch = input_dict['s_batch']
            self.a_batch = input_dict['a_batch']
            self.r_batch = input_dict['r_batch']
            #self.dnn_chunk_remain=DNN_TEST_NUM
            #self.time_stamp=0
            
            BaseHTTPRequestHandler.__init__(self, *args, **kwargs)
        def do_POST(self):
            post_data_f=open('./post_log/rl_post_data_'+opt.run_vid_trace,'a')
            content_length = int(self.headers['Content-Length'])
            post_data = json.loads(self.rfile.read(content_length))
            for i in post_data:
                post_data_f.write(i+': '+str(post_data[i])+'\t')
            post_data_f.write('\n')
            post_data_f.flush()
            #post_data_f.close()
            print(post_data)
            #if post_data['lastRequest']>opt.total_chunk:
            #    exit(0)
            #global time_stamp,dnn_chunk_remain
            #Request_Handler.time_stamp = Request_Handler.time_stamp + 1000
            time_stamp=float(time.time())*1000
            if ('pastThroughput' in post_data):
                # @Hongzi: this is just the summary of throughput/quality at the end of the load
                # so we don't want to use this information to send back a new quality
                post_data_f.write('pastThroughput in postdata \n')
                post_data_f.flush()
                print("Summary: ", post_data)
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

                    #try:
                        #print(post_data['finished'])
                    #except:
                        #post_data['finished']=0

                    try:
                        rebuf = float(post_data['RebufferTime'] - self.input_dict['last_total_rebuf'])/M_IN_K
                        delay = post_data['lastChunkFinishTime'] - post_data['lastChunkStartTime']
                        buffer_size=post_data['buffer']
                        end_of_dnn=post_data['finished'] #should be updated by postdata
                        bit_rate=post_data['lastquality']
                        lastq = bit_rate
                    except:
                        post_data_f.write('Error in processing post_data 1 \n')
                        post_data_f.flush()
                    if(Request_Handler.exp_finished!=1):
                        # added for dnn
                        if (post_data['lastquality'] < len(opt.bitrate)):

                            #reward = opt.bitrate[post_data['lastquality']] / M_IN_K \
                            #         - opt.linear_rebuf * rebuf \
                            #         - opt.smooth * np.abs(opt.bitrate[post_data['lastquality']] -
                            #                               opt.bitrate[self.input_dict['last_bit_rate']]) / M_IN_K
                            self.log_file.write(str(time_stamp / M_IN_K) + '\t' +
                                                str(opt.bitrate[post_data['lastquality']]) + '\t' +
                                                str(buffer_size) + '\t' +
                                                str(rebuf) + '\t' +
                                                str(post_data['lastChunkSize']) + '\t' +
                                                str(delay) + '\t' +
                                                str(end_of_dnn) + '\t' + 
                                                str(post_data['lastChunkFinishTime']) + '\n')

                        else:  # reward for dnn
                            if not Request_Handler.last_end_of_dnn:
                                #reward = -opt.linear_rebuf * rebuf
                                self.log_file.write(str(time_stamp / M_IN_K) + '\t' +
                                               'dnndownload' + '\t' +
                                               str(buffer_size) + '\t' +
                                               str(rebuf) + '\t' +
                                               str(post_data['lastChunkSize']) + '\t' +
                                               str(delay) + '\t' +
                                               #str(reward) + '\t' +
                                               'dnndownload' + '\t' +
                                               str(end_of_dnn) + '\t' +  
                                               str(post_data['lastChunkFinishTime']) + '\n')
                            else:
                                #print('DNN-pensieve: Wrong behavior')
                                #reward = -opt.dnn_penalty
                                self.log_file.write(str(time_stamp / M_IN_K) + '\t' +
                                                    'dnndownload' + '\t' +
                                               str(buffer_size) + '\t' +
                                               str(rebuf) + '\t' +
                                               str(post_data['lastChunkSize']) + '\t' +
                                               str(delay) + '\t' +
                                               #str(reward) + '\t' +
                                               'dnndownload' + '\t' +
                                               str(end_of_dnn) + '\t' + 
                                               str(post_data['lastChunkFinishTime']) + '\n')

                        self.log_file.flush()
                    # # --linear reward--
                    # reward = VIDEO_BIT_RATE[post_data['lastquality']] / M_IN_K \
                    #          - REBUF_PENALTY * rebuffer_time / M_IN_K \
                    #          - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[post_data['lastquality']] -
                    #                                    self.input_dict['last_bit_rate']) / M_IN_K
                    #
                    # # --log reward--
                    # log_bit_rate = np.log(VIDEO_BIT_RATE[post_data['lastquality']] / float(VIDEO_BIT_RATE[0]))
                    # log_last_bit_rate = np.log(self.input_dict['last_bit_rate'] / float(VIDEO_BIT_RATE[0]))
                    #
                    # rewardlog = log_bit_rate \
                    #             - 4.3 * rebuffer_time / M_IN_K \
                    #             - SMOOTH_PENALTY * np.abs(log_bit_rate - log_last_bit_rate)
                    #
                    # # --hd reward--
                    # rewardhd = BITRATE_REWARD[post_data['lastquality']] \
                    #            - 8 * rebuffer_time / M_IN_K - np.abs(
                    #     BITRATE_REWARD[post_data['lastquality']] - BITRATE_REWARD_MAP[self.input_dict['last_bit_rate']])

                    self.input_dict['last_total_rebuf'] = post_data['RebufferTime']

                    # retrieve previous state
                    if len(self.s_batch) == 0:
                        state = [np.zeros((opt.s_info + (1 if opt.dnn_mode == 1 else 0), opt.s_len))]
                    else:
                        state = np.array(self.s_batch[-1], copy=True)

                    # compute bandwidth measurement

                    video_chunk_size = post_data['lastChunkSize']

                    # compute number of video chunks left
                    video_chunk_remain =opt.total_chunk- post_data['lastRequest']-1
                    #print(int(post_data['lastquality']),len(opt.bitrate))
                    self.input_dict['video_chunk_count'] += (int(post_data['lastquality'])!=len(opt.bitrate))

                    # dequeue history record
                    state = np.roll(state, -1, axis=1)

                    next_video_chunk_sizes = []
                    for i in range(len(opt.bitrate)):
                        next_video_chunk_sizes.append(get_chunk_size(i, self.input_dict['video_chunk_count']))

                    # this should be S_INFO number of terms
                    Request_Handler.dnn_chunk_remain=(Request_Handler.dnn_chunk_remain-(int(post_data['lastquality'])==len(opt.bitrate))*video_chunk_size) if (Request_Handler.dnn_chunk_remain-(int(post_data['lastquality'])==len(opt.bitrate))*video_chunk_size)>=1000 else 0
                    if post_data['lastquality']<len(opt.bitrate):
                        self.input_dict['last_bit_rate'] = post_data['lastquality']
                    else:
                        post_data['lastquality']=self.input_dict['last_bit_rate']
                    try:
                        post_data_f.write('before state \n')
                        post_data_f.flush()
                        #print(time_stamp,post_data['lastquality'], buffer_size, float(video_chunk_size)/1000,delay,video_chunk_remain,Request_Handler.dnn_chunk_remain/DNN_TEST_NUM,len(self.s_batch))
                        state[0, -1] = opt.bitrate[post_data['lastquality']] / \
                                       float(np.max(opt.bitrate))  # last quality
                        state[1, -1] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
                        state[2, -1] = float(video_chunk_size) / \
                                       float(delay) / M_IN_K if delay else state[2, -2]  # kilo byte / ms
                        state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR if delay else state[3, -2]  # 10 sec
                        state[4, :len(opt.bitrate)] = np.array(
                            next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
                        state[5, -1] = np.minimum(video_chunk_remain,
                                                  opt.total_chunk) / float(opt.total_chunk)
                        if int(opt.dnn_mode)==1:
                            state[6, -1] = Request_Handler.dnn_chunk_remain/DNN_TEST_NUM
                        post_data_f.write('State_Info: quality: '+str(state[0,-1])+'\t buffer: '+str(state[1,-1])+'\t video_chunk_size: '+str(state[2,-1])+'\t delay: '+str(state[3,-1])+'\t next_vid_chunk: '+str(state[4,:len(opt.bitrate)])+'\t video_chunk_remain: '+str(state[5,-1])+'\t dnn_chunk_left:'+str(Request_Handler.dnn_chunk_remain/DNN_TEST_NUM)+'\n')
                        #print ()
                    except ZeroDivisionError:
                        # this should occur VERY rarely (1 out of 3000), should be a dash issue
                        # in this case we ignore the observation and roll back to an eariler one
                        if len(self.s_batch) == 0:
                            state = [np.zeros((opt.s_info + (1 if opt.dnn_mode == 1 else 0), opt.s_len))]
                        else:
                            state = np.array(self.s_batch[-1], copy=True)

                    # log wall_time, bit_rate, buffer_size, rebuffer_time, video_chunk_size, download_time, reward
                    # self.log_file.write(str(time.time()) + '\t' +
                    #                     str(VIDEO_BIT_RATE[post_data['lastquality']]) + '\t' +
                    #                     str(post_data['buffer']) + '\t' +
                    #                     str(rebuffer_time / M_IN_K) + '\t' +
                    #                     str(video_chunk_size) + '\t' +
                    #                     str(video_chunk_fetch_time) + '\t' +
                    #                     str(reward) + '\n')
                    # self.log_file.flush()
                    # self.loglog_file.write(str(time.time()) + '\t' +
                    #                        str(VIDEO_BIT_RATE[post_data['lastquality']]) + '\t' +
                    #                        str(post_data['buffer']) + '\t' +
                    #                        str(rebuffer_time / M_IN_K) + '\t' +
                    #                        str(video_chunk_size) + '\t' +
                    #                        str(video_chunk_fetch_time) + '\t' +
                    #                        str(rewardlog) + '\n')
                    # self.loglog_file.flush()
                    # self.loghd_file.write(str(time.time()) + '\t' +
                    #                       str(VIDEO_BIT_RATE[post_data['lastquality']]) + '\t' +
                    #                       str(post_data['buffer']) + '\t' +
                    #                       str(rebuffer_time / M_IN_K) + '\t' +
                    #                       str(video_chunk_size) + '\t' +
                    #                       str(video_chunk_fetch_time) + '\t' +
                    #                       str(rewardhd) + '\n')
                    # self.loghd_file.flush()

                    if(buffer_size>opt.buffer_threshold):
                        Request_Handler.start_dnn_download=1
                    #dnn_mode =0 dnn_method =0 do dnn byte based 

                    action_prob = self.actor.predict(np.reshape(state, (1, opt.s_info + (1 if opt.dnn_mode == 1 else 0), opt.s_len)))
                    #action_cumsum = np.cumsum(action_prob)


                    if not end_of_dnn or opt.dnn_mode!=1:
                        bit_rate = action_prob.argmax()
                    else:
                        action_prob[0][len(opt.bitrate)]=0
                        bit_rate = action_prob.argmax()
                    dnn_byte_size = 0 
                    if(opt.dnn_mode==2 and Request_Handler.start_dnn_download==1 and Request_Handler.send_dnn_baseline %2==0 and not end_of_dnn):
                        bit_rate=len(opt.bitrate)
                        dnn_byte_size=opt.dnn_byte_size
                        if opt.dnn_method==1 and opt.dnn_mode != 1:
                            dnn_byte_size=np.floor(video_chunk_size*opt.dnn_vid_rate)
                    Request_Handler.send_dnn_baseline +=1
                    #print(bit_rate)    
                    post_data_f.write(str(bit_rate))
                    post_data_f.flush()
                    post_data_f.close()
                        # Note: we need to discretize the probability into 1/RAND_RANGE steps,
                    # because there is an intrinsic discrepancy in passing single state and batch states

                    # send data to html side
                    send_data = str(bit_rate)
                    if opt.dnn_mode ==2:
                        send_data=str(bit_rate)+'\t'+str(dnn_byte_size)

                    Request_Handler.last_end_of_dnn=end_of_dnn
                    end_of_video = False
                    if (post_data['lastRequest'] + 1 >= opt.total_chunk and lastq < len(opt.bitrate)):
                        send_data = "REFRESH"
                        # stop when video download ends
                        end_of_video = True
                        Request_Handler.exp_finished=1
                        Request_Handler.start_dnn_download=0
                        Request_Handler.dnn_chunk_remain=DNN_TEST_NUM
                        self.input_dict['last_total_rebuf'] = 0
                        self.input_dict['last_bit_rate'] = opt.default_quality
                        self.input_dict['video_chunk_count'] = 0
                        self.log_file.write('\n')  # so that in the log we know where video ends
                        self.log_file.flush()
                    print(send_data)
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/plain')
                    self.send_header('Content-Length', len(send_data))
                    self.send_header('Access-Control-Allow-Origin', "*")
                    self.end_headers()
                    self.wfile.write(send_data.encode("utf-8"))

                    # record [state, action, reward]
                    # put it here after training, notice there is a shift in reward storage

                    if end_of_video:
                        self.s_batch.append(np.zeros((opt.s_info + (1 if opt.dnn_mode == 1 else 0), opt.s_len)))
                    else:
                        self.s_batch.append(state)
                except  Exception as e:
                    post_data_f.write('RL SERVER ERROR : '+str(e))
                    post_data_f.flush()
                    exit(-1)
        def do_GET(self):
            #print >> sys.stderr, 'GOT REQ'
            print('GOT REQ')
            self.send_response(200)
            # self.send_header('Cache-Control', 'Cache-Control: no-cache, no-store, must-revalidate max-age=0')
            self.send_header('Cache-Control', 'max-age=3000')
            self.send_header('Content-Length', 20)
            self.end_headers()
            self.wfile.write("console.log('here');".encode("utf-8"))

        def log_message(self, format, *args):
            return

    return Request_Handler


def run(server_class=HTTPServer, port=8333, log_file_path=LOG_FILE):
    np.random.seed(opt.random_seed)



    if not os.path.exists(SUMMARY_DIR):
        os.makedirs(SUMMARY_DIR)

    #with tf.Session() as sess, open(log_file_path +'/log_' +opt.reward+'_'+str(opt.dnn_mode)+'_'+opt.run_vid_trace+'_'+opt.content+'_'+opt.quality+'_', 'w') as log_file:
    with tf.Session() as sess, open(log_file_path +'/log_' +'{}_{}_{}_{}_{}_PARSEHERE{}'.format(str(opt.dnn_mode),opt.reward,opt.content, opt.quality, opt.model.split('/')[-1],opt.run_vid_trace), 'w') as log_file: 
        # actor = a3c.ActorNetwork(sess,
        #                          state_dim=[opt.s_info + opt.dnn_mode, opt.s_len],
        #                          action_dim=len(opt.bitrate) + opt.dnn_mode,
        #                          learning_rate=opt.actor_lr)
        #
        # critic = a3c.CriticNetwork(sess,
        #                            state_dim=[opt.s_info + opt.dnn_mode, opt.s_len],
        #                            learning_rate=opt.critic_lr)
        actor = a3c.ActorNetwork(sess,
                                 state_dim=[opt.s_info + (1 if opt.dnn_mode == 1 else 0), opt.s_len],
                                 action_dim=len(opt.bitrate) + (1 if opt.dnn_mode == 1 else 0),
                                 learning_rate=opt.actor_lr)
        critic = a3c.CriticNetwork(sess,
                                   state_dim=[opt.s_info + (1 if opt.dnn_mode == 1 else 0), opt.s_len],
                                   learning_rate=opt.critic_lr)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()  # save neural net parameters

        # restore neural net parameters
        nn_model = NN_MODEL
        if nn_model is not None:  # nn_model is the path to file
            saver.restore(sess, nn_model)
            print("Model restored.")
        else:
            print('no model is provided')
            sys.exit()

        action_vec = np.zeros(len(opt.bitrate) + (1 if opt.dnn_mode == 1 else 0))
        action_vec[opt.default_quality] = 1

        s_batch = [np.zeros((opt.s_info + (1 if opt.dnn_mode == 1 else 0), opt.s_len))]
        a_batch = [action_vec]
        r_batch = []

        train_counter = 0

        last_bit_rate = opt.default_quality
        last_total_rebuf = 0
        # need this storage, because observation only contains total rebuffering time
        # we compute the difference to get

        video_chunk_count = 0

        input_dict = {'sess': sess, 'log_file': log_file,
                      #'loglog_file': loglog_file, 'loghd_file': loghd_file,
                      'actor': actor, 'critic': critic,
                      'saver': saver, 'train_counter': train_counter,
                      'last_bit_rate': last_bit_rate,
                      'last_total_rebuf': last_total_rebuf,
                      'video_chunk_count': video_chunk_count,
                      's_batch': s_batch, 'a_batch': a_batch, 'r_batch': r_batch}

        # interface to abr_rl server
        handler_class = make_request_handler(input_dict=input_dict)
        
        server_address = ('0.0.0.0', port)
        httpd = server_class(server_address, handler_class)
        # httpd.socket = ssl.wrap_socket (httpd.socket, keyfile="./key.pem", certfile='./cert.pem', server_side=True)
        print('Listening on port ' + str(port))
        httpd.serve_forever()


def main():
    #if len(sys.argv) == 2:
    #    trace_file = sys.argv[1]
    #    run(log_file_path=LOG_FILE + '_RL_' + trace_file)
    #else:
    global video_sizes
    
    # for bitrate in range(len(opt.bitrate)):
    #         video_sizes[bitrate] = []
    #         with open(os.path.join(opt.video_size_dir, VIDEO_SIZE_FILE + str(bitrate))) as f:
    #             for line in f:
    #                 video_sizes[bitrate].append(int(line.split()[0]))
    #jh TODO: fix this
    # video_sizes = template.get_video_sizes(opt.video_dir, opt.content, opt.total_chunk)
    #assert self.total_video_chunk <= len(self.video_size[0])
    run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Keyboard interrupted.")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

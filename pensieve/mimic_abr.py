#!/usr/bin/env python
from http.server import BaseHTTPRequestHandler, HTTPServer
# import SocketServer
import base64
import urllib
import sys
import os
import logging
import json

from collections import deque
import numpy as np
import time


VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]  # Kbps
BITRATE_REWARD = [1, 2, 3, 12, 15, 20]
BITRATE_REWARD_MAP = {0: 0, 300: 1, 750: 2,
					  1200: 3, 1850: 12, 2850: 15, 4300: 20}
M_IN_K = 1000.0
DEFAULT_QUALITY = 0  # default video quality without agent
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> this number of Mbps
SMOOTH_PENALTY = 1
TOTAL_VIDEO_CHUNKS = 48
SUMMARY_DIR = './results'
LOG_FILE = './results/log'
# in format of time_stamp bit_rate buffer_size rebuffer_time video_chunk_size download_time reward


def make_request_handler(input_dict):

	class Request_Handler(BaseHTTPRequestHandler):
		var = 0
		#time_stamp=float(time.time())*1000
		last_end_of_dnn = 0
		def __init__(self, *args, **kwargs):
			self.input_dict = input_dict
			self.log_file = input_dict['log_file']
			self.loglog_file = input_dict['loglog_file']
			self.loghd_file = input_dict['loghd_file']
			self.bit_rate_all = input_dict['bit_rate']
			BaseHTTPRequestHandler.__init__(self, *args, **kwargs)

		def do_POST(self):
			content_length = int(self.headers['Content-Length'])
			post_data = json.loads(self.rfile.read(content_length))
			print(post_data)
		
			time_stamp=float(time.time())*1000
	 
			rebuf = float(post_data['RebufferTime'] - self.input_dict['last_total_rebuf'])/M_IN_K
			delay = post_data['lastChunkFinishTime'] - post_data['lastChunkStartTime']
			buffer_size=post_data['buffer']
			end_of_dnn=post_data['finished'] #should be updated by postdata
			bit_rate=post_data['lastquality']
			# print post_data['lastquality']
			x = int(10 * np.random.rand())
			send_data = self.bit_rate_all[Request_Handler.var]
			print (send_data +  str('\n'))
			Request_Handler.var += 1

	   			

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
			self.wfile.write("console.log('here');")

		def log_message(self, format, *args):
			return

	return Request_Handler


def run(server_class=HTTPServer, port=8333, log_file_path=LOG_FILE):

	bit_rate_all = []
	if not os.path.exists(SUMMARY_DIR):
		os.makedirs(SUMMARY_DIR)

	with open(log_file_path, 'r') as f:
		for line in f:
			parse =  line.split()
			if len(parse) == 0:
				break
			if parse[1] == 'dnndownload':
				bit_rate_all.append("5")
			elif parse[1] == '400.0':
				bit_rate_all.append("0")
			elif parse[1] == '800.0':
				bit_rate_all.append("1")
			elif parse[1] == '1200.0':
				bit_rate_all.append("2")
			elif parse[1] == '2400.0':
				bit_rate_all.append("3")
			else:
				bit_rate_all.append("4")
		print(bit_rate_all)

	with open(log_file_path + 'QOElin', 'wb') as log_file:
		with open(log_file_path + 'QOElog', 'wb') as loglog_file:
			with open(log_file_path + 'QOEhd', 'wb') as loghd_file:
				last_bit_rate = DEFAULT_QUALITY
				last_total_rebuf = 0
				input_dict = {'log_file': log_file,
							  'loglog_file': loglog_file,
							  'loghd_file': loghd_file,
							  'last_bit_rate': last_bit_rate,
							  'last_total_rebuf': last_total_rebuf,
							  'loglog_file': loglog_file,
							  'loghd_file': loghd_file,
				  'bit_rate': bit_rate_all}

				handler_class = make_request_handler(input_dict=input_dict)

				server_address = ('0.0.0.0', port)
				httpd = server_class(server_address, handler_class)
				print('Listening on port ' + str(port))
				httpd.serve_forever()


def main():
	if len(sys.argv) == 2:
		#abr_algo = sys.argv[1]
		trace_file = sys.argv[1]
		print(trace_file)
		run(log_file_path=trace_file)
	else:
		run()


if __name__ == "__main__":
	try:
		main()
	except KeyboardInterrupt:
		logging.debug("Keyboard interrupted.")
		try:
			sys.exit(0)
		except SystemExit:
			os._exit(0)

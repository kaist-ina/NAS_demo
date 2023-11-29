from pathlib import Path
import os
import sys
import option
import numpy as np
import read_sr_bitrates
def get_contents():
    return ['sport', 'technology', 'news', 'music', 'game', 'entertainment', 'cook', 'comedy', 'beauty']

#Note: KB / TODO: merge ultra0, ultra1 into ultra
def get_dnn_chunk_size(quality):

    return read_sr_bitrates.get_dnn_chunk_size(quality)


    ###Deprecated below###
    if quality == 'low':
        return [180]
    elif quality == 'medium':
        return [185, 232, 220]
    elif quality == 'high':
        return [352, 223, 225, 217, 217]
    elif quality == 'ultra0':
        return [353, 310, 222, 203, 222, 203, 222, 195, 222, 195]
    elif quality == 'ultra1':
        return [664, 425, 425, 417, 417]
    else:
        print('unsupported quality')
        sys.exit(-1)

#
def get_sr_exp_time(dnn_chunk,quality,qualityopt='ultra1',clock_fre=0):
    qualityopt='ultra1'
    if dnn_chunk ==0:
            return []
    if clock_fre==0:
        if quality==0:
            data=[21.66,
                  24.26,
                  26.92,
                  29.73,
                  32.54]
            data.reverse()
            return data[5-dnn_chunk:]

        elif quality==1:
            data=[16.1,
             19.95,
             23.76,
             27.72,
             31.71]
            data.reverse()
            return data[5-dnn_chunk:]
        elif quality==2:
            data=[13.02,
                  18.31,
                  23.42,
                  28.79,
                  34.16]
            data.reverse()
            return data[5-dnn_chunk:]
        elif quality==3:
            data=[13.88,
                  22.48,
                  31.39]
            data.reverse()
            return data[len(data)-dnn_chunk:]
        elif quality==4:
            data=[]
            return [0]


    elif clock_fre==1:
        if quality==0:
            data=[24.24,
                  27.15,
                  30.18,
                  33.55,
                  36.48]
            data.reverse()
            return data[5-dnn_chunk:]

        elif quality==1:
            data=[18.09,
                  22.34,
                  26.68,
                  31.14,
                  35.61]
            data.reverse()
            return data[5-dnn_chunk:]
        elif quality==2:
            data=[14.6,
                  20.35,
                  26.36,
                  32.44,
                  38.56]
            data.reverse()
            return data[5-dnn_chunk:]
        elif quality==3:
            data=[15.56,
                  25.18,
                  35.13]
            data.reverse()
            return data[3-dnn_chunk:]
        elif quality==4:
            data=[]
            return [0]


    elif clock_fre==2:
        if quality==0:
            data=[26.99,
                  30.21,
                  33.55,
                  37.04,
                  40.5]
            data.reverse()
            return data[5-dnn_chunk:]
        elif quality==1:
            data=[20.05,
                  24.74,
                  29.54,
                  34.49,
                  39.46]
            data.reverse()
            return data[5-dnn_chunk:]
        elif quality==2:
            data=[16.16,
                  22.42,
                  29.06,
                  35.71,
                  42.41]
            data.reverse()
            return data[5-dnn_chunk:]
        elif quality==3:
            data=[17.24,
                  27.91,
                  39.06]
            data.reverse()
            return data[3-dnn_chunk:]
        elif quality==4:
            data=[]
            return [0]

    elif clock_fre==3:

        if quality==0:
            data=[29.87,
                  33.03,
                  36.65,
                  40.31,
                  44.15]
            data.reverse()
            return data[5-dnn_chunk:]

        elif quality==1:
            data=[21.74,
                  26.83,
                  32.1,
                  37.45,
                  42.8]
            data.reverse()
            return data[5-dnn_chunk:]
        elif quality==2:
            data=[17.53,
                  23.91,
                  30.64,
                  37.41,
                  44.16]
            data.reverse()
            return data[5-dnn_chunk:]
        elif quality==3:
            data=[18.8,
                  30.46,
                  42.53]
            data.reverse()
            return data[3-dnn_chunk:]

        elif quality==4:
            data=[]
            return [0]
    elif clock_fre==4:

        if quality==0:
            data=[32.43,
                  35.93,
                  39.9,
                  43.9,
                  47.9]
            #print(data)
            #print(data.reverse())
            data.reverse()
            return data[5-dnn_chunk:]

        elif quality==1:
            data=[24.07,
                  29.63,
                  35.62,
                  41.45,
                  47.4]
            data.reverse()
            return data[5-dnn_chunk:]
        elif quality==2:
            data=[19.4,
                  25.98,
                  32.89,
                  39.83,
                  46.81]
            data.reverse()
            return data[5-dnn_chunk:]
        elif quality==3:
            data=[20.82,
                  33.92,
                  47.33]
            data.reverse()
            return data[3-dnn_chunk:]

        elif quality==4:
            data=[]
            return [0]

    elif clock_fre==5:
        if quality==0:
            data=[63.51,
                  70.67,
                  78.01,
                  85.22,
                  92.5]
            data.reverse()
            return data[5-dnn_chunk:]

        elif quality==1:
            data=[46.58,
                  57.65,
                  68.89,
                  80.19,
                  91.63]
            data.reverse()
            return data[5-dnn_chunk:]
        elif quality==2:
            data=[36.43,
                  46.95,
                  57.62,
                  68.3,
                  78.96]
            data.reverse()
            return data[5-dnn_chunk:]

        elif quality==3:
            data=[39.26,
                  64.67,
                  90.17]
            data.reverse()
            return data[3-dnn_chunk:]

        elif quality==4:
            data=[]
            return [0]



#TODO: fill-out device & quality
def get_sr_time(dnn_chunk,quality,qualityopt):
    if 1 or qualityopt == 'ultra0' or qualityopt == 'ultra1':
        #return [0]
        if quality==0:
            if dnn_chunk>=9:
                return [30,27,24,22,19]
            elif dnn_chunk>=7:
                return [27,24,22,19]
            elif dnn_chunk>=5:
                return [24,22,19]
            elif dnn_chunk>=3:
                return [22,19]
            elif dnn_chunk>=1:
                return [19]
        elif quality==1:
            if dnn_chunk >= 9:
                return [30,26,22,18,14]
            elif dnn_chunk >= 8:
                return [26,22,18,14]
            elif dnn_chunk >= 6:
                return [22,18,14]
            elif dnn_chunk >= 4:
                return [18,14]
            elif dnn_chunk >= 2:
                return [14]
        elif quality == 2:
            if dnn_chunk >= 9:
                return [25,21,17,13,10]
            elif dnn_chunk >= 8:
                return [21,17,13,10]
            elif dnn_chunk >= 6:
                return [17,13,10]
            elif dnn_chunk >= 4:
                return [13,10]
            elif dnn_chunk >= 2:
                return [10]
        elif quality==3:
            if dnn_chunk >= 6:
                return [27,18,10]
            elif dnn_chunk >= 4:
                return [18,10]
            elif dnn_chunk >= 2:
                return [10]
        elif quality==4:
            return [0]
        else:
            print("UNEXPECTED QUALITY ?: "+str(quality))
            return None

        return []
        return {'quality': 'ultra',
                '240p':[19, 22, 24, 27, 30],
                '360p':[14, 18, 22, 26, 30],
                '480p':[10, 13, 17, 21, 25],
                '720p':[10, 18, 27]
                }

    else:
        print('Unsupported devices')
        sys.exit()

def get_sr_quality(content, quality):
    return read_sr_bitrates.get_sr_quality(content,quality)

    #deprecation
    if quality == 'ultra0' or quality == 'ultra1':
        if content == 'game':

            return {'240p':[403, 490, 566, 653, 776],
                    '360p':[878, 1014, 1085, 1237, 1423],
                    '480p':[1277, 1471, 1663, 1801 , 2002],
                    '720p':[2400,2471,2512],
                    '1080p':[4800]
                   }
        elif content == "technology":
            return {'240p':[400,631,768,847,947],
                    '360p':[963,1252,1617,1783,1879],
                    '480p':[1255,1439,1837,1950,2136],
                    '720p':[2400,2400,2475],
                    '1080p':[4800]
                   }
        elif content == "beauty":
            return {'240p':[416,618,710,825,937],
                    '360p':[1016,1267,1555,1681,1784],
                    '480p':[1392,1594,1895,2070,2165],
                    '720p':[2400,2453,2594],
                    '1080p':[4800]
                   }
        elif content == "cook":
            return {'240p':[1242,1733,1873,2027,2064],
                    '360p':[3188,3896,4283,4382,4508],
                    '480p':[2622,3209,3613,3918,3929],
                    '720p':[2638,3153,3510],
                    '1080p':[4800]
                   }
        elif content == "game":
            return {'240p':[400, 490, 566, 653, 777],
                    '360p':[878,1014,1085,1237,1424],
                    '480p':[1277,1471,1663,1801,2002],
                    '720p':[2400,2471,2512],
                    '1080p':[4800]
                   }
        elif content == "news":
            return {'240p':[400,517,662,744,851],
                    '360p':[800,1033,1214,1540,1766],
                    '480p':[1200,1508,1809,2035,2175],
                    '720p':[2400,2400,2418],
                    '1080p':[4800]
                   }
        elif content == "comedy":
            return {'240p':[400,411,539,620,690],
                    '360p':[820,890,999,1112,1332],
                    '480p':[1210,1312,1545,1735,1881],
                    '720p':[2400,2400,2446],
                    '1080p':[4800]
                   }
        elif content == "entertainment":
            return {'240p':[429,576, 694 ,793, 899],
                    '360p':[1039,1203,1325,1424,1539],
                    '480p':[1377,1496,1690,1818,1920],
                    '720p':[2400,2471,2524],
                    '1080p':[4800]
                   }

        elif content == "music":
            return {'240p':[443,695,813,896,967],
                    '360p':[1124,1345,1488,1585,1679],
                    '480p':[1542,1715,1873,1938,2016],
                    '720p':[2400,2458,2554],
                    '1080p':[4800]
                   }

        elif content == "sport":
            return {'240p':[458,569,637,712,778],
                    '360p':[986,1086,1173,1240,1295],
                    '480p':[1373,1481,1579,1668,1726],
                    '720p':[2400,2423,2462],
                    '1080p':[4800]
                   }
        elif content == "pensieve":
            #copied retrun value of sport, in pensieve video we will not use SR
            return {'240p':[458,569,637,712,778],
                    '360p':[986,1086,1173,1240,1295],
                    '480p':[1373,1481,1579,1668,1726],
                    '720p':[2400,2423,2462],
                    '1080p':[4800]
            }
        
        elif content == "average":
            contents=['technology',"beauty","cook","game","news","comedy","entertainment","music","sport"]
            return_dict={}
            quality_dict=get_sr_quality('game',quality)
            for key in quality_dict:
                return_dict[key]=np.zeros(np.array(quality_dict[key]).shape)
            for content in contents:
                quality_dict=get_sr_quality(content,quality)
                for key in quality_dict:
                    return_dict[key]+=np.array(quality_dict[key])

            for key in return_dict:
                return_dict[key] /= len(contents)

            return return_dict
        
        else:
            print('Unsupported Contents')
            sys.exit()

    elif quality=='high':
        if content == 'game' or content== 'music':
             return {'240p':[400,486,539,608,718],
                     '360p':[880,984,1046,1158,1297],
                     '480p':[1273,1430,1542,1674,1834],
                     '720p':[2400,2451,2539],
                    '1080p':[4800]
                   }
        elif content == 'entertainment':
            return {'240p':[431,540,666,735,829],
                     '360p':[1024,1160,1232,1330,1457],
                     '480p':[1381,1486,1574,1659,1819],
                     '720p':[2400,2439,2523],
                    '1080p':[4800]
                   }


        else:
            print('Unsupported Contents')
            sys.exit()

    elif quality=='medium':
        if content == 'game' or content== 'music':
             return {'240p':[400, 469, 507, 570, 641],
                     '360p':[877, 982, 1006, 1082,1207],
                     '480p':[1270, 1368,1446, 1495, 1580],
                     '720p':[2400, 2405],
                    '1080p':[4800]
                   }
        elif content == 'entertainment':
            return {'240p':[422,516,593,687,763],
                     '360p':[1024,1138,1204,1304,1400],
                     '480p':[1374,1409,1461,1510,1594],
                     '720p':[2400,2400],
                    '1080p':[4800]
                   }

        else:
            print('Unsupported Contents')
            sys.exit()


    elif quality=='low':
        if content == 'game' or content== 'music':
             return {'240p':[405,424,470,481,499],
                     '360p':[877,931,968,995,1028],
                     '480p':[1248,1273,1301,1327,1339],
                     '720p':[2400,2400],
                     '1080p':[4800]
                   }
        elif content == 'entertainment':
            return {'240p':[420,407,436,519,567],
                     '360p':[1042,1066,1100,1146,1213],
                     '480p':[1379,1378,1409,1425,1423],
                     '720p':[2400,2400],
                    '1080p':[4800]
                   }

        else:
            print('Unsupported Contents')
            sys.exit()


    else:
        print('Unsupported quality')
        sys.exit()

#dnn_chunk : 1-10
def get_partial_sr_quality(dnn_chunk, content, quality):
    #quality_data = get_sr_quality(content, quality)
    quality_data=read_sr_bitrates.get_sr_quality(content,quality)
    #ultra1 is ultra 180912 DNN update
    quality_list = []
    if quality == 'ultra0':
        assert 0 #no more ultra0 deprecate
        assert dnn_chunk >=0 and dnn_chunk <=10
        
        #1080p
    #elif quality == 'ultra1':
    else:#Ultra high medium low
        assert dnn_chunk >= 0 and dnn_chunk <=5
        if dnn_chunk == 5:
            quality_list.append(quality_data['240p'][4])
            quality_list.append(quality_data['360p'][4])
            quality_list.append(quality_data['480p'][4])
            quality_list.append(quality_data['720p'][1])
            quality_list.append(quality_data['1080p'][0])
        elif dnn_chunk == 4:
            quality_list.append(quality_data['240p'][3])
            quality_list.append(quality_data['360p'][3])
            quality_list.append(quality_data['480p'][3])
            quality_list.append(quality_data['720p'][1])
            quality_list.append(quality_data['1080p'][0])
        elif dnn_chunk == 3:
            quality_list.append(quality_data['240p'][2])
            quality_list.append(quality_data['360p'][2])
            quality_list.append(quality_data['480p'][2])
            quality_list.append(quality_data['720p'][1])
            quality_list.append(quality_data['1080p'][0])
        elif dnn_chunk == 2: #360,480,720
            quality_list.append(quality_data['240p'][1])
            quality_list.append(quality_data['360p'][1])
            quality_list.append(quality_data['480p'][1])
            quality_list.append(quality_data['720p'][1])
            quality_list.append(quality_data['1080p'][0])
        elif dnn_chunk == 1: #240
            
            quality_list.append(quality_data['240p'][0])
            quality_list.append(quality_data['360p'][0])
            quality_list.append(quality_data['480p'][0])
            quality_list.append(quality_data['720p'][0])
            quality_list.append(quality_data['1080p'][0])
        elif dnn_chunk == 0:
            quality_list = option.opt.bitrate
    '''
    elif quality == 'high':
        assert dnn_chunk >= 0 and dnn_chunk <=5
        if dnn_chunk == 5:
            quality_list.append(quality_data['240p'][4])
            quality_list.append(quality_data['360p'][4])
            quality_list.append(quality_data['480p'][4])
            quality_list.append(quality_data['720p'][2])
            quality_list.append(quality_data['1080p'][0])
        elif dnn_chunk == 4:
            quality_list.append(quality_data['240p'][3])
            quality_list.append(quality_data['360p'][3])
            quality_list.append(quality_data['480p'][3])
            quality_list.append(quality_data['720p'][2])
            quality_list.append(quality_data['1080p'][0])
        elif dnn_chunk == 3:
            quality_list.append(quality_data['240p'][2])
            quality_list.append(quality_data['360p'][2])
            quality_list.append(quality_data['480p'][2])
            quality_list.append(quality_data['720p'][2])
            quality_list.append(quality_data['1080p'][0])
        elif dnn_chunk == 2: #360,480,720
            quality_list.append(quality_data['240p'][1])
            quality_list.append(quality_data['360p'][1])
            quality_list.append(quality_data['480p'][1])
            quality_list.append(quality_data['720p'][1])
            quality_list.append(quality_data['1080p'][0])
        elif dnn_chunk == 1: #240
            quality_list.append(quality_data['240p'][0])
            quality_list.append(quality_data['360p'][0])
            quality_list.append(quality_data['480p'][0])
            quality_list.append(quality_data['720p'][0])
            quality_list.append(quality_data['1080p'][0])
        elif dnn_chunk == 0:
            quality_list = option.opt.bitrate


    
    elif quality == 'medium':
        assert dnn_chunk >= 0 and dnn_chunk <=3
        if dnn_chunk == 3:
            quality_list.append(quality_data['240p'][4])
            quality_list.append(quality_data['360p'][4])
            quality_list.append(quality_data['480p'][4])
            quality_list.append(quality_data['720p'][1])
            quality_list.append(quality_data['1080p'][0])
        elif dnn_chunk == 2: #360,480,720
            quality_list.append(quality_data['240p'][2])
            quality_list.append(quality_data['360p'][2])
            quality_list.append(quality_data['480p'][2])
            quality_list.append(quality_data['720p'][1])
            quality_list.append(quality_data['1080p'][0])
        elif dnn_chunk == 1: #240
            quality_list.append(quality_data['240p'][0])
            quality_list.append(quality_data['360p'][0])
            quality_list.append(quality_data['480p'][0])
            quality_list.append(quality_data['720p'][0])
            quality_list.append(quality_data['1080p'][0])
        elif dnn_chunk == 0:
            quality_list = option.opt.bitrate
    elif quality == 'low':
        assert dnn_chunk >= 0 and dnn_chunk <=1
        if dnn_chunk == 1: #240
            quality_list.append(quality_data['240p'][4])
            quality_list.append(quality_data['360p'][4])
            quality_list.append(quality_data['480p'][4])
            quality_list.append(quality_data['720p'][1])
            quality_list.append(quality_data['1080p'][0])
        elif dnn_chunk == 0:
            quality_list = option.opt.bitrate
    
    '''

    #IF SR is smaller don't do it
    for i in range(len(quality_list)):
        if option.opt.bitrate[i]<quality_list[i]:
            pass
        else:
            quality_list[i]=option.opt.bitrate[i]

    return quality_list

def get_video_sizes(video_dir, content, total_chunk, resolution=[240,360,480,720,1080], bitrate=[400,800,1200,2400,4800]):


    #Video is Fixed to Game video
    
    #if content != 'game' and content != 'news':
    content='game'


    if content == 'average':
        video_size_dict = {}
        contents = get_contents()
        for content in contents:
            video_size_dict[content]= {}
            for idx in range(len(resolution)):
                video_size_dict[content][idx] = []
                for chunk_num in range(1, total_chunk+1):
                    chunk_num = (chunk_num-1) % total_chunk +1
                    video_chunk_path = os.path.join(video_dir, content, '{}p'.format(resolution[idx]), 'output_{}k_dash{}.m4s'.format(bitrate[idx], chunk_num))
                    chunk_size = os.path.getsize(video_chunk_path)
                    video_size_dict[content][idx].append(chunk_size)

        video_size = {}
        for idx in range(len(resolution)):
            video_size[idx] = []
            for chunk_num in range(0, total_chunk):
                total_video_size = 0
                for content in contents:
                    #print(video_size_dict[content][idx][chunk_num])
                    total_video_size += video_size_dict[content][idx][chunk_num]
                video_size[idx].append(total_video_size / len(contents))
        return video_size
    elif content =='pensieve':
        video_size={}
        for bitrate in range(6):
            video_size[bitrate]=[]
            for chunk_num in range(1, 49+1):#starting from 1 because pensieve does
                video_chunk_path = '../video_server/' + \
                               'video' + \
                               str(6 - bitrate) + \
                               '/' + \
                               str(chunk_num) + \
                               '.m4s'
                chunk_size = os.path.getsize(video_chunk_path)
                video_size[bitrate].append( chunk_size)
                #f.write(str(chunk_size) + '\n')
        return video_size
    else:
        video_size = {}
        for idx in range(len(resolution)):
            video_size[idx] = []
            with open('{}/video_size_'.format(video_dir) + str(idx), 'w') as f:
                for chunk_num in range(1, total_chunk+1):
                    chunk_num =  (chunk_num-1) % total_chunk +1
                    video_chunk_path = os.path.join(video_dir, content, '{}p'.format(resolution[idx]), 'output_{}k_dash{}.m4s'.format(bitrate[idx], chunk_num))
                    chunk_size = os.path.getsize(video_chunk_path)
                    video_size[idx].append(chunk_size)
        return video_size

#deprecated - 04/30
def get_sr_bitrates(content, device):
    print('use sr_quality')
    sys.exit()

    if device == '1080ti':
        if content == 'average':
            return [1061, 2025, 2458, 3007, 4800]
        elif content == 'sport':
            return [808, 1361, 1860, 2611, 4800]
        elif content == 'music':
            return [1070, 1782, 2256, 2825, 4800]
        elif content == 'entertainment':
            return [876, 1579, 2073, 2779, 4800]
        elif content == 'comedy':
            return [727, 1417, 2121, 2700, 4800]
        elif content == 'game':
            return [805, 1572, 2155, 2741, 4800]
        elif content == 'technology':
            return [963, 1991, 2528, 3078, 4800]
        elif content == 'news':
            return [863, 1872, 2368, 2710, 4800]
        elif content == 'cook':
            return [2479, 4762, 4292, 4438, 4800]
        elif content == 'beauty':
            return [960, 1886, 2469, 3185, 4800]
        else:
            print('Unsupported Contents')
            sys.exit()
    elif device == '1050ti':
        if content == 'game':
            return [559, 1054, 1534, 2502, 4800]
        else:
            print('Unsupported Contents')
            sys.exit()
    elif device == '1060':
        if content == 'game':
            return [671, 1241, 1765, 2499, 4800]
        else:
            print('Unsupported Contents')
            sys.exit()
    elif device == '1070':
        if content == 'game':
            return [724, 1368, 1939, 2639, 4800]
        else:
            print('Unsupported Contents')
            sys.exit()
    else:
        print('Unsupported devices')
        sys.exit()

#deprecated - 04/30
#used for pensieve baseline mode
def get_dnn_info(device):
    print('use get_dnn_chunk_size')
    sys.exit()

    dnn_info = {}
    if device == '1080ti':
        dnn_info['size'] = 3.218 * 1000 * 1000
        dnn_info['action'] = 400 * 1000
        return dnn_info
    elif device == '1070':
        dnn_info['size'] = 1.344 * 1000 * 1000
        dnn_info['action'] = 400 * 1000
        return dnn_info
    elif device == '1060':
        dnn_info['size'] = 0.661 * 1000 * 1000
        dnn_info['action'] = 400 * 1000
        return dnn_info
    elif device == '1050ti':
        dnn_info['size'] = 0.172 * 1000 * 1000
        dnn_info['action'] = 400 * 1000
        return dnn_info
    else:
        print('unsupported deice')
        sys.exit()

if __name__ == '__main__':
    #video_dir = os.path.join(str(Path.home()), 'OSDI18', 'video', '5min')
    #content = 'average'
    #video_sizes = get_video_sizes(video_dir, content, 75)
    #print(video_sizes)
    print(get_partial_sr_quality(5,'average','ultra1'))
    """
    print(get_partial_sr_quality(0, 'game', 'ultra0'))
    print(get_partial_sr_quality(1, 'game', 'ultra0'))
    print(get_partial_sr_quality(2, 'game', 'ultra0'))
    print(get_partial_sr_quality(3, 'game', 'ultra0'))
    print(get_partial_sr_quality(4, 'game', 'ultra0'))
    print(get_partial_sr_quality(5, 'game', 'ultra0'))
    print(get_partial_sr_quality(6, 'game', 'ultra0'))
    print(get_partial_sr_quality(7, 'game', 'ultra0'))
    print(get_partial_sr_quality(8, 'game', 'ultra0'))
    print(get_partial_sr_quality(9, 'game', 'ultra0'))
    print(get_partial_sr_quality(10, 'game', 'ultra0'))
    print(get_partial_sr_quality(0, 'game', 'ultra1'))
    print(get_partial_sr_quality(1, 'game', 'ultra1'))
    print(get_partial_sr_quality(2, 'game', 'ultra1'))
    print(get_partial_sr_quality(3, 'game', 'ultra1'))
    print(get_partial_sr_quality(4, 'game', 'ultra1'))
    print(get_partial_sr_quality(5, 'game', 'ultra1'))
    """

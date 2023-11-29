import sys, os, logging, argparse
from pathlib import Path

parser = argparse.ArgumentParser(description="PyTorch Super Resolution")

# Directory
# [dataRoot or resultRoot]/[dataType]/[videoType]/[contentType]
parser.add_argument('--dataRoot', default='./data', metavar='DIR')
parser.add_argument('--resultRoot', default='./result', metavar='DIR')
parser.add_argument('--dataType', default='video',  metavar='DIR', choices=('video, div2k'), help='dataset directory')
parser.add_argument('--videoType', default='final-full', metavar='DIR', help='dataset directory')
parser.add_argument('--contentType', default='', metavar='DIR', help='content directory')

# NASNet specifications
parser.add_argument('--quality', required=True, choices=('low', 'medium', 'high', 'ultra', '1080ti', 'titanxp'))

# SRDenseNet specification
parser.add_argument('--growthRate', type=int, default=8)
parser.add_argument('--bottleneckFeat', type=int, default=64)

# Training specifications
parser.add_argument('--patchSize', type=int, default=48, metavar='N', help='input patch size')
parser.add_argument('--vbatchSize', type=int, default=1, metavar='N', help='input batch size for validation')
parser.add_argument("--batchSize", type=int, default=64, help="Training batch size")
parser.add_argument("--nEpoch", type=int, default=50, help="Number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=0.1")
parser.add_argument("--threads", type=int, default=6, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="Weight decay, Default: 1e-4")
parser.add_argument('--decayEpoch', type=int, default=-1, metavar='N', help='learning rate decay per N epochs, -1 disables decaying')
parser.add_argument('--decayFactor', type=float, default=0.5, metavar='N', help='learning rate decay factor for step decay')
parser.add_argument('--lossFunc', type=str, default='l2', choices=('l2', 'l1', 'l1_c'))
parser.add_argument("--resume", action="store_true", help="resume training a model")
parser.add_argument('--pretrained', action="store_true", help='use a pretrained model')
parser.add_argument('--finetuned', action="store_true", help='use a generic model to finetune')
parser.add_argument("--startEpoch", default=1, type=int, help="Manual epoch number (useful on restarts)")

# Datset specification (related to videos)
parser.add_argument("--trainNum", type=int, default=-1, help="Validation dataset number") #not fully implemented
parser.add_argument("--trainFPS", type=float, default=1.0, help="Validation dataset number")
parser.add_argument("--validNum", type=int, default=-1, help="Validation dataset number") #not fully implemented
parser.add_argument("--validFPS", type=float, default=0.1, help="Validation dataset number")
parser.add_argument("--testNum", type=int, default=-1, help="Test dataset number") #not fully implemented
parser.add_argument("--testFPS", type=float, default=0.3, help="Validation dataset number")
parser.add_argument('--lazy_load', action="store_true", help='Load dataset in a lazy manner (not pre-load on memory)')
parser.add_argument('--pre_upscaled', action="store_true", help='Use pre-upscaled images for input like VDSR network') #TODO: automatically configure this value by verifyting neural network architectures

# Logging specification
parser.add_argument('--customName', default='', type=str, help='additional name for model')
parser.add_argument('--logEpoch', default=5, type=int, help='logging frequency')


#Test specification
parser.add_argument('--border', default=10, type=int, help='amount of pixel to remove from boarder when calculating PSNR')
parser.add_argument("--testEpoch", default=50, type=int, help="Manual epoch number to test(useful on restarts)")
parser.add_argument("--startBatchSize", default=1, type=int)

# DASH option
parser.add_argument('--dashHR', default=1080, type=int, help='High resolution of DASH')
parser.add_argument('--dashLR', default=[240, 360, 480, 720], nargs='+', type=int, help='Low resolution of DASH')

# DIV2K option
parser.add_argument('--div2kLR', default=[4, 3, 2, 1], nargs='+', type=float, help='Scale of DIV2K dataset')

# FFMPEG option
parser.add_argument('--resize', default='bicubic', type=str)

# process.py option
parser.add_argument('--processMode', default='runtime', choices=('multi', 'scalable', 'mock', 'runtime', 'figure16'), type=str)
parser.add_argument('--runtimeNum', default=1, type=int)

# Evaluation
parser.add_argument('--evalFrame', default=2880, type=int, help='Total amount of video length used for video')
parser.add_argument('--evalChunk', default=96, type=int, help='Total amount of video length used for video')
parser.add_argument('--evalMode', default='quality', choices=('quality', 'speed', 'all'), help='Evaluation mode')
parser.add_argument('--evalNode', default=-1, type=int, choices=(-1, 1, 2, 3, 4, 5))
parser.add_argument('--evalLast', action="store_true", help='Eval only last layer (used in evaluate.py')
parser.add_argument('--enable_debug', action="store_true", help='enable debugging mode (saving images)')

# Dummy job
parser.add_argument('--dummyType', default='DNN', type=str, choices=('DNN'), help='Type of dummy jobs')
parser.add_argument('--dummyLayer', default=20, type=int)
parser.add_argument('--dummyFeat', default=24, type=int) #TODO: modify to highest available feature

# (Deprecated) - maybe used for implementing new types of network
parser.add_argument('--network', type=str, default='edsr', choices=('vdsr', 'edsr', 'lapsrn', 'srdense', 'titanxp'))
parser.add_argument('--model', type=str, default='dynamic', choices=('static', 'dynamic'), help='choose a static DNN or a scalable (dynamic) DNN')
parser.add_argument('--act', type=str, default='relu', choices=('relu', 'lrelu', 'prelu'))
parser.add_argument('--nLayer', type=int, default=10, metavar='B',help='number of feature maps')
parser.add_argument('--nFeat', type=int, default=64, metavar='B', help='number of feature maps')
parser.add_argument('--nChannel', type=int, default=3, metavar='C',
                    help='number of color channels to use')
parser.add_argument('--minOutput', type=int, default=3, metavar='B',
                    help='minimun output layer')
parser.add_argument('--outputType', type=int, default=1, choices=(0,1,2,3), metavar='B',
                    help='output node type')
parser.add_argument('--outputFilter', type=int, default=1, metavar='B',
                    help='output node filter for scalable edsr')
parser.add_argument('--randType', type=int, default=3, choices=(0,1,2,3,4,5), metavar='B',
                    help='output node type')
parser.add_argument('--randWeight', type=int, default=1, help='weight for construct an array for ranaom sampling')
parser.add_argument('--randBias', type=int, default=5, help='weight for construct an array for random sampling')
parser.add_argument("--usePreprocess", action="store_true", help="use a preprocess block in mdsr network")

opt = parser.parse_args()

logging.basicConfig(level=logging.INFO)#Logging level
resolution_dict = {'HD': (720, 1280), 'Full_HD': (1080, 1920)}#Define key-value for model and resolution

#Model name
model_name = opt.quality
if opt.dataType == 'video':
    model_name += '_{}'.format(''.join(str(x) for x in opt.dashLR))
else:
    model_name += '_div2k'
    model_name += '_{}'.format(''.join(str(x) for x in opt.div2kLR))

if opt.customName != '':
    model_name += '_{}'.format(opt.customName)
opt.model_name = model_name

#Create a directory
if opt.dataType == 'video':
    opt.modelDir = os.path.join(opt.resultRoot, opt.dataType, opt.videoType, opt.contentType, opt.model_name, 'checkpoint')
    opt.resultDir= os.path.join(opt.resultRoot, opt.dataType, opt.videoType, opt.contentType, opt.model_name, 'result')
    opt.videoDir = os.path.join(opt.dataRoot, opt.dataType, opt.videoType, opt.contentType)
    opt.datasetDir = os.path.join(opt.dataRoot, opt.dataType, opt.videoType, opt.contentType, 'dataset')
else:
    opt.modelDir = os.path.join(opt.resultRoot, opt.dataType, opt.model_name, 'checkpoint')
    opt.resultDir= os.path.join(opt.resultRoot, opt.dataType, opt.model_name, 'result')
    opt.datasetDir = os.path.join(opt.dataRoot, opt.dataType, opt.model_name, 'dataset')

opt.imageDir = os.path.join(opt.resultDir, 'image')
os.makedirs(opt.modelDir, exist_ok=True)
os.makedirs(opt.resultDir, exist_ok=True)
os.makedirs(opt.imageDir, exist_ok=True)

"""
# Process specification
parser.add_argument('--header', default='Header.m4s', type=str, help='DASH header segment')
parser.add_argument('--videoIdx', default=1, type=int, help='DASH video segment')
parser.add_argument('--resolution', default='HD', choices=('HD', 'FULL_HD'), type=str, help='target resolution')
parser.add_argument('--modelPath', type=str, help='Pretrained model path')
parser.add_argument('--dashDir', default='./', type=str, help='DASH datapath')
parser.add_argument('--processDir', default='process', type=str, help='process directory')
parser.add_argument('--processNum', default=8, type=int, help='process number for multiprocess pool')

# DNN process
parser.add_argument('--codec', default='h264', type=str, help='encoding method')
parser.add_argument('--crf', default=23, type=int, help='crf for encoding')
parser.add_argument('--preset', default='ultrafast', type=str, help='preset for encoding')
parser.add_argument('--pix_fmt', default='yuv420p', type=str, help='pix_fmt for encoding')
"""

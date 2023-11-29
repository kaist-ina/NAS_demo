import os.path
import struct
import sys
import time
sys.path.insert(0, '../super_resolution')
import utility as util
#TIMESCALE = 12800import utility as util

index = int(sys.argv[2])

# print fsize

#print(f, index)
#print(f, fsize)
#print(sys.version)

prefix = "- "
ATOM = {"ftyp", "moov", "styp", "sidx", "moof", "mdat"}

start_time = time.time()
while True:
    total_frame = util.get_video_frame_count(sys.argv[1])

    if total_frame is None:
        continue

    if total_frame['frames'] == 96:
        break

f = open(sys.argv[1], "rb")
fsize = os.path.getsize(sys.argv[1])
n = 0
atom_offset = []
tfdt_offset = []
bmdt_values = []

while n < fsize:
    #Debug
    #fsize = os.path.getsize(sys.argv[1])
    #print(fsize)

    data = f.read(8)
    al, an = struct.unpack(">I4s", data)

    an = an.decode()
    if an == "moov":
        n += 8
    elif an == "trak":
        n += 8
    elif an == "mdia":
        n += 8
    elif an == "mdhd":
        n += 8
        f.read(4)
        n += 4
        # now reading actual data
        f.read(8)
        data = f.read(4)
        timescale = struct.unpack(">I", data)
        #print('timescale: {}'.format(timescale))
        TIMESCALE = timescale[0]
        f.read(8)
        n += 20

    elif an == "moof":
        atom_offset.append((an, al))
        n += 8
    elif an == "traf":
        atom_offset.append((an, al))
        n += 8
    elif an == "tfdt":
        f.read(4)
        tfhd = f.read(8)
        n += 8 + 4
        tfdt_offset.append(n)
        n += 8
        bmdt = struct.unpack(">Q", tfhd)
        bmdt_values.append(bmdt[0])
    else:
        atom_offset.append((an, al))
        f.read(al-8)
        n += al

#if len(bmdt_values) > 0:
#    break

f.close()
end_time = time.time()
print('bmdt_values: {} / elapsed_time: {}sec'.format(len(bmdt_values), end_time - start_time))

rf = open(sys.argv[1], "r+b")

for i in range(len(bmdt_values)):
    rf.seek(tfdt_offset[i])
    scale = float(bmdt_values[i]) / TIMESCALE
    new_bmdt = (scale + (index)*4) * TIMESCALE
    data = struct.pack(">Q", int(new_bmdt))
    rf.write(data)
rf.close()

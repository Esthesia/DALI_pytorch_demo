from time import perf_counter, sleep
import pynvml 
from datetime import datetime
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
util = pynvml.nvmlDeviceGetUtilizationRates(handle)
import sys

file_dir = sys.argv[1]

DELAY=0.001
DEVICE_NUM=pynvml.nvmlDeviceGetCount()

f = open(f"{file_dir}", 'w')

handles = []
f.write(f'timestamp,')
for i in range(DEVICE_NUM):
    handles.append(pynvml.nvmlDeviceGetHandleByIndex(i))
    f.write(f'GPU{i},')
f.write("\n")

len(handles)

while True:
    f.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')+",")
    for i in range(DEVICE_NUM):
        # start = perf_counter()
        util = pynvml.nvmlDeviceGetUtilizationRates(handles[i])
        f.write(f'{util.gpu},')
    f.write("\n")
    sleep(DELAY)
    
f.close()


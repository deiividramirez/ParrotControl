import signal
import time
 
def handler(signum, frame):
    print("Ctrl-c was pressed. Quitting...")
    exit()
 
 
signal.signal(signal.SIGINT, handler)
 
count = 0
while True:
    count += 1
    print(f"{count}", end="\r")
    time.sleep(1e-8)
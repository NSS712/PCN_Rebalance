import time
from tqdm import tqdm

for i in tqdm(range(3), leave=False):
    time.sleep(1)
print("\033[2K\033[G")



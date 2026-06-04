import sys
import time
from bgsl.utils.progress import _SingleLineStream
from tqdm import tqdm

stream = _SingleLineStream(sys.stderr)
for i in tqdm(range(10), file=stream):
    time.sleep(0.1)

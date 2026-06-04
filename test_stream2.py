import sys
import time
from tqdm import tqdm

class _SingleLineStream:
    def __init__(self, stream=None) -> None:
        self._stream = stream if stream is not None else sys.stderr

    def write(self, s: str) -> None:
        if not s:
            return

        if "\r" in s:
            parts = s.split("\r")
            for i, part in enumerate(parts):
                if i > 0:
                    self._stream.write("\r")
                    self._stream.write("\x1b[2K")   # ANSI: clear current line
                if part:
                    self._stream.write(part)
            self._stream.flush()
            return

        # Plain write (metric lines, newlines): pass through unchanged.
        self._stream.write(s)
        self._stream.flush()

    def flush(self) -> None:
        self._stream.flush()

    def isatty(self) -> bool:
        return getattr(self._stream, "isatty", lambda: False)()

    @property
    def encoding(self) -> str:
        return getattr(self._stream, "encoding", "utf-8")

stream = _SingleLineStream(sys.stdout)
for i in tqdm(range(50), file=stream):
    time.sleep(0.01)

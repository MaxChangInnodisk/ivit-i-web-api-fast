import contextlib
import logging as log
import threading
import time
from typing import Union

from ivit_i.utils import iDevice


class iDeviceAsync:
    def __init__(self, interval: int = 5):
        self.idev = iDevice()
        self.lock = threading.RLock()
        self.is_stop = False
        self.t = self.create_thread(start=True)
        self.info = {}

    def update_device_event(self) -> None:
        """The while loop to keep update device information."""
        try:
            log.warning("iDevice start")
            while not self.is_stop:
                self.lock.acquire()
                with contextlib.redirect_stdout(None):
                    self.info = self.idev.get_device_info()
                self.lock.release()
                time.sleep(1)
        except Exception as e:
            log.warning(f"iDevice Warning: {type(e).__name__} {e}")

    def create_thread(self, start: bool) -> threading.Thread:
        thr = threading.Thread(target=self.update_device_event, daemon=True)
        if start:
            thr.start()
        return thr

    def get_device_info(self, uid: Union[list, str, None] = None) -> dict:
        if isinstance(uid, list):
            uid = uid[0]
        device_info = self.info.get(uid)
        if device_info is None:
            return self.info
        else:
            return {uid: device_info}

    def stop(self):
        self.is_stop = True
        self.t.join()

    def __del__(self):
        self.stop()


def test_2():
    idev = iDeviceAsync()
    while True:
        print(idev.get_device_info())
        time.sleep(0.1)


if __name__ == "__main__":
    try:
        # test_1()
        test_2()
    except KeyboardInterrupt:
        pass
    finally:
        print("end")

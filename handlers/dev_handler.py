import logging as log
import threading
import time
from ivit_i.utils import iDevice 
from typing import Union ,get_args

class iDeviceAsync():
    

    def __init__(self, interval:int =5):

        self.idev = iDevice()
        self.lock = threading.RLock()
        self.is_stop = False 
        self.t = self.create_thread(start=True)
        self.info = {}

    def update_device_event(self) -> None:
        """ The while loop to keep update device information. """
        try:
            log.warning('iDevice start')
            while(not self.is_stop):
                self.lock.acquire()
                self.info = self.idev.get_device_info()
                self.lock.release()
                time.sleep(1)
        except Exception as e:
            log.exception(e)

        finally:
            log.warning(f'{self.__class__.__name__} is stopped !')

    def create_thread(self, start:bool) -> threading.Thread:
        thr = threading.Thread(
            target=self.update_device_event,
            daemon=True )
        if start: 
            thr.start()
        return thr

    def get_device_info(self, uid: Union[ str, None]=None) -> dict:
        return self.info

    def stop(self):
        self.is_stop = True
        self.t.join()

    def __del__(self):
        self.stop()

def test_2():
    idev = iDeviceAsync()
    while(True):
        
        print(idev.get_device_info())
        time.sleep(0.1)
    
if __name__ == "__main__":
    try:
        # test_1()
        test_2()
    except KeyboardInterrupt as e:
        pass
    finally:
        print('end')
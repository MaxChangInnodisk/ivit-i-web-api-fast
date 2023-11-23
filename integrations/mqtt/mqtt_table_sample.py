import argparse
import json
import time
from datetime import datetime
from typing import Union
from collections import defaultdict, UserDict
from rich.console import Console
from rich.table import Table

from mqtt_sample import MqttClient, get_args

"""MQTT Integration Sample

The data should be:

{
    "data": {
        "uid": "55d318be", 
        "title": "test", 
        "app_uid": "e668bac2", 
        "start_time": "1700647034186001749", 
        "end_time": "None", 
        "annotation": {
            "area": {
                "name": "Area 1", 
                "area_point": [
                    [0.244, 0.613], 
                    [0.614, 0.558], 
                    [0.795, 0.713], 
                    [0.311, 0.811]], 
                "output": {
                    "car": 1, "truck": 1
                }
            }
        }, 
        "task_uid": "e668bac2"
    },
    "message": "", 
    "type": "EVENT"
}

"""

class TFMT:
    """ Time Formatter """

    @staticmethod
    def verify(timestamp: Union[float, str, int]) -> float:
        """Verift and return the same format of time.time()

        Args:
            timestamp (Union[float, str, int]): the timestamp supports time.time() and time.time_ns()

        Returns:
            float: as same as time.time()
        
        """
        # string: time.time(), time.time_ns()
        bef = timestamp
        if isinstance(timestamp, str):
            if timestamp.isdigit():
                timestamp = int(timestamp)
            else:
                timestamp = float(timestamp)
        
        # time.time_ns()
        if isinstance(timestamp, int):
            timestamp = timestamp/1e9

        return timestamp

    @staticmethod
    def struct(timestamp: Union[float, str, int]) -> time.struct_time:
        """Structure use time.localtime

        Args:
            timestamp (Union[float, str, int]): the timestamp supports time.time() and time.time_ns()

        Returns:
            time.struct_time: time.localtime
        """
        return time.localtime(TFMT.verify(timestamp))

    @staticmethod
    def format(timestamp: time.struct_time, time_format: str="%Y-%m-%d %H:%M:%S") -> str:
        """Return a string as specified by the format argument

        Args:
            timestamp (time.struct_time): support gmtime() or localtime()
            time_format (_type_, optional): the date format. Defaults to "%Y-%m-%d %H:%M:%S".

        Returns:
            str: A string as specified by the format argument
        """
        return time.strftime(time_format, timestamp)

class Event(UserDict):
    """ Config Wrapper could help to build up config object with lower case """
    __t_pre = None
    __t_cur = None
    __t_pass = 0

    def update_time_pass(self, current_time:Union[float, str, int]) -> None:
        """Update modify time

        Args:
            current_time (Union[float, str, int]): the timestamp supports time.time() and time.time_ns()
        """
        current_time = TFMT.verify(current_time)
        if self.__t_cur is None:
            self.__t_cur = current_time
            return

        self.__t_pre = self.__t_cur
        self.__t_cur = current_time
        self.__t_pass = self.__t_cur - self.__t_pre

    def get_time_pass(self) -> float:
        """get the  modify time """
        return self.__t_pass

    def update(self, data):
        super().update(data)
        self.update_time_pass(self.__getitem__("start_time"))

class EventClient(MqttClient):

    def __init__(self, broker_address: str, broker_port: Union[int, str]):
        super().__init__(broker_address, broker_port)

        self.events = defaultdict(Event)

        self.console = Console()
        self.table = self.create_rich_table()
        self.console.print(self.table)

    def create_rich_table(self):
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Task")
        table.add_column("Title")
        table.add_column("Output")
        table.add_column("Start Time")
        table.add_column("Updated")
        
        return table

    def reset_table(self):
        self.console.clear()
        self.table = self.create_rich_table()

    def update_table(self):
        self.reset_table()
        for uid, event in self.events.items():
            self.table.add_row(
                event["task_uid"],
                event["title"],
                str(event["annotation"]["area"]["output"]),
                TFMT.format(TFMT.struct(event["start_time"])),
                f'{round(event.get_time_pass(), 3)}s ago'
            )
        self.console.print(self.table)

    def on_message(self, client, userdata, message):
        raw = message.payload.decode("utf-8")
        try:
            data = json.loads(raw)["data"]
            uid = data["task_uid"]
            self.events[uid].update(data)
            
            self.update_table()
        except Exception as e:
            print(e)


def main(args):
    
    IVIT_EVENT_TOPIC = "events" 

    mc = EventClient(
        broker_address=args.ip,
        broker_port=args.port
    )
    
    mc.connect()
    mc.subscribe(IVIT_EVENT_TOPIC)
    
    try:
        mc.loop_forever()
    except KeyboardInterrupt:
        mc.disconnect()
    finally:
        print('---- END ----')

def test_TFMT():
    tt = "1700717754.1669095"
    print(TFMT.struct(tt))
    print(TFMT.format(TFMT.struct(tt)))

def test_event():

    tn = time.time()
    data = {
        'uid': 'bbf86b10', 
        'title': 'test', 
        'app_uid': 'e668bac2', 
        'start_time': tn, 
        'end_time': 'None', 'annotation': {'area': {'name': 'Area 1', 'area_point': [[0.244, 0.613], [0.614, 0.558], [0.795, 0.713], [0.311, 0.811]], 'output': {'car': 2, 'truck': 0}}}, 'task_uid': 'e668bac2'}
    
    ti = Event(data)

    data2 = {
        'uid': 'bbf86b10', 
        'title': 'test', 
        'app_uid': 'e668bac2', 
        'start_time': tn+3,
        'end_time': 'None', 'annotation': {'area': {'name': 'Area 1', 'area_point': [[0.244, 0.613], [0.614, 0.558], [0.795, 0.713], [0.311, 0.811]], 'output': {'car': 2, 'truck': 0}}}, 'task_uid': 'e668bac2'}
    
    ti.update(data2)
    
    print(ti.get_time_pass())

if __name__ == "__main__":

    # test_TFMT()
    # test_event()

    main(get_args())

    
    
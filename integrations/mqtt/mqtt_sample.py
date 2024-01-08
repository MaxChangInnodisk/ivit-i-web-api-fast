import argparse
import paho.mqtt.client as mqtt
from typing import Union

""" MQTT Integration Sample

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

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--ip", type=str,
                        default="127.0.0.1",
                        required=True,
                        help="the IP Address of the MQTT broker.")
    parser.add_argument("-p", "--port", type=int,
                        default="6683",
                        required=True,
                        help="the port of the MQTT brocker.")
    return parser.parse_args()

class MqttClient:

    def __init__(self, broker_address: str, broker_port: Union[int, str]):
        self.broker_address = broker_address
        self.broker_port = int(broker_port)
        
        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect
        
        self.client.on_message = self.on_message

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            # print("Connected")
            pass
        else:
            mesg = f"Connected Failed: {rc}"
            # print(mesg)
            raise ConnectionError(mesg)
    
    def on_disconnect(self, client, userdata,  rc):
        print('Disconnected !')

    def on_message(self, client, userdata, message):
        print(message.payload.decode("utf-8"))

    def connect(self):
        return self.client.connect(self.broker_address, self.broker_port)

    def disconnect(self):
        return self.client.disconnect()

    def subscribe(self, topic: str):
        return self.client.subscribe(topic)

    def loop_forever(self):
        return self.client.loop_forever()


def main(args):

    IVIT_EVENT_TOPIC = "events" 

    mc = MqttClient(
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

if __name__ == "__main__":

    main(get_args())
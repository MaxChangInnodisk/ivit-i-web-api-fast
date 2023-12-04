# Intergrate with MQTT


1. Use mosquitto client
    1. Install mosquitto client
        ```
        sudo apt-get install mosquitto-clients 
        ```
    2. Subscribe the topic
        a. To get all events
            ```bash
            mosquitto_sub -h localhost -t "events" -p 6683
            ``` 
        b. To get target events
            ```bash
            mosquitto_sub -h localhost -t "events/<uid>" -p 6683
            ```

2. Use python `paho` to finished it.
    ```bash
    pip install paho-mqtt
    python3 mqtt_sample.py -i 127.0.0.1 -p 6683

    python3 mqtt_table_sample.py -i 127.0.0.1 -p 6683 
    ```
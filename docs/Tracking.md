# IVIT-I Application of basic Tracking
## Usage
You need to follow the step below to use application:  
Step 1. [Setting Config](#setting-app-config).  
Step 2. [Create Instance](#create-instance).  
Step 3. Follow the [format of input parameter](#format-of-input-parameter) to use application.

And the description of application output is [here](#application-output).   
## Setting app config 
* The description of key from config.(*) represent must be set.  

| Name | Type | Default | Description |
| --- | --- | --- | --- |
|application(*)|dict|{  }|Encapsulating all information of configuration.|
|areas(*)|list|[  ]|Seting the location of detection area. |
|name|str|default|Area name.|
| depend_on (*) | list | [ ] | The application depend on which label. |
| palette | dict | { } | Custom the color of each label. |
|area_point|list|[ ]|Area for detection.**Value need to normalization**|
|events|dict|{ }|Conditions for a trigger event ·|
|draw_result|bool|True|Display information of detection.|
|draw_bbox|bool|True|Display boundingbox.|

* Basic
    ```bash

    "application": {
                    "areas": [
                                {
                                    "name": "default",
                                    "depend_on": [ ],
                                    "area_point": [ ]
                                }
                            ]
                    }

    ```
* Set up application and event

   ```bash
   {
    "application": {
                    "areas": [
                                {
                                    "name": "Datong Rd",
                                    "depend_on": [ 'car', 'truck'],
                                    "palette":{
                                                    "car": [ 0, 255, 0 ],
                                                    "truck": [ 0, 255, 0 ]
                                                },
                                    "area_point": [ [0.156,0.203],[0.468, 0.203],[0.468, 0.592],[0.156, 0.592] ], 
                                    "events": {
                                                "title": "The daily traffic is over 2",
                                                "logic_operator": ">",
                                                "logic_value": 2,
                                              }
                                },
                            ],
                    "draw_result":False,
                    "draw_bbox":False
                    }
    }
   ``` 
## Create Instance
You need to use [app_config](#setting-app-config) and label path to create instance of application.
   ```bash
    from apps import Tracking

    app = Tracking( app_config, label_path )
   ``` 
## Format of input parameter
* Input parameters are the result of model predict, and the result must packed like below.

| Type | Description |
| --- | --- |
|object|Object's properties : xmin ,ymin ,xmax ,ymax ,score ,id ,label |
* Example:
    ```bash
        detection        # (type object)                   
        detection.label  # (type str)           value : person   
        detection.score  # (type numpy.float64) value : 0.960135 
        detection.xmin   # (type int)           value : 1        
        detection.ymin   # (type int)           value : 78       
        detection.xmax   # (type int)           value : 438      
    ```
## Application output 
* Application will return frame(already drawn) and two information(app_output、event_output).The format of organized information as below.
    ```bash
    #common output
    app_output = {
                    'areas': [
                                {
                                    'id': 0, 
                                    'name': 'The defalt area', 
                                    'data': [
                                                {'label': 'person', 'num': 1}, 
                                                {'label': 'cell phone', 'num': 0}
                                            ]
                                }
                             ]
                 }
    
    #triggering event
    event_output = {
                    'event': [
                                {
                                    'uuid': '288b0944-', 
                                    'title': '1111', 
                                    'areas': {
                                                'id': 0, 
                                                'name': 'The defalt area', 
                                                'data': [
                                                            {'label': 'person', 'num': 1}, 
                                                            {'label': 'cell phone', 'num': 0}
                                                        ]
                                              }, 
                                    'timesamp': datetime.datetime(2023, 4, 13, 10, 6, 11, 317019), 
                                    'screenshot': {
                                                    'overlay': './288b0944-/2023-04-13 10:06:11.317019.jpg', 
                                                    'original': './288b0944-/2023-04-13 10:06:11.317019_org.jpg'
                                                  }
                                }
                              ]
                   } 
    ```
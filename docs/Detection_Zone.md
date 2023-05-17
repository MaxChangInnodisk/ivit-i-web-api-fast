# IVIT-I Application of Detection Zone
## Usage
You need to follow the step below to use application:  
Step 1. [Setting Config](#setting-app-config).  
Step 2. [Create Instance](#create-instance).  
Step 3. Follow the [format of input parameter](#format-of-input-parameter)  to use application.

And the description of application output is [here](#application-output).

More function :  
1. User can control anythong about draw through function [set_draw()](#control-anything-about-draw).

## Setting app config 
Application Setting
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

* Basic Sample
    ```json
        {
            "application": {
                "areas": [
                    {
                        "name": "default",
                        "depend_on": [],
                    }
                ]
            }
        }
    ```
* Advanced Sample (Set up application and event)

   ```json
    {
        "application": {
						"palette": {
                        "car": [
                            0,
                            255,
                            0
                        ],
                        "truck": [
                            0,
                            255,
                            0
                        ]
                    },
            "areas": [
                {
                    "name": "The intersection of Datong Rd",
                    "depend_on": [
                        "car",
                        "truck"
                    ],
                    "area_point": [
                        [
                            0.156,
                            0.203
                        ],
                        [
                            0.468,
                            0.203
                        ],
                        [
                            0.468,
                            0.592
                        ],
                        [
                            0.156,
                            0.592
                        ]
                    ],
                    "events": {
                        "title": "Traffic is very heavy",
                        "logic_operator": ">",
                        "logic_value": 100,
                    }
                },
            ]
        }
    }
   ``` 
## Create Instance
You need to use [app_config](#setting-app-config) and label path to create instance of application.
   ```python
    
    from apps import Detection_Zone

    app = Detection_Zone( app_config , label_path )
    
   ``` 
## Format of input parameter
* Input parameters are the result of model predict, and the result must packed like below.

| Type | Description |
| --- | --- |
|list|[ detection1 ,detection2 ,detection3 ,...]|
* Example:
    ```bash
        detection        # (type object)                   
        detection.label  # (type str)           value : person   
        detection.score  # (type numpy.float64) value : 0.960135 
        detection.xmin   # (type int)           value : 1        
        detection.ymin   # (type int)           value : 78       
        detection.xmax   # (type int)           value : 438  
        detection.ymax   # (type int)           value : 50     
    ```
## Application output 
* Application will return frame(already drawn) and two information(app_output、event_output).The format of organized information as below.
    ```python
    #common output
    app_output = {
                    'areas': [
                            {
                    'id': 0, 
                    'name': 'The defalt area', 
                    'data': [
                                    {
                    'label': 'person', 
                    'num': 2
                                    }
                                ]
                            }
                        ]
                }
    #triggering event
    event_output =  {
                        'event': [
                                {
                        'uuid': 'bb4e7b1f-', 
                        'title': 'The daily traffic is over 1000', 
                        'areas': {
                        'id': 0, 
                        'name': 'The defalt area', 
                        'data': [
                                            {'label': 'person', 'num': 2
                                            },
                                            {'label': 'tvmonitor', 'num': 1
                                            }
                                        ]
                                    }, 
                        'timesamp': datetime.datetime(2023,
                                    4,
                                    13,
                                    9,
                                    52,
                                    4,
                                    703097), 
                        'screenshot': {
                        'overlay': './bb4e7b1f-/2023-04-13 09: 52: 04.703097.jpg', 'original': './bb4e7b1f-/2023-04-13 09: 52: 04.703097_org.jpg'
                                    }
                                }
                            ]
                    }
    
    ```
## Control anything about draw.
* In this application we can control :
1. show bounding box. 
2. show result box.  
3. show detection zone.
4. change color for specific object.   

* Before we use set_draw() , we need to set parameter like bellow.  
    ```bash

        data:dict = {  
                draw_bbox : bool ,  # Control bounding box whether draw or not draw.
                draw_result : bool , # Control result box whether draw or not draw.
                draw_area : bool , # Control detection zone whether draw or not draw.
                palette: list:[ turple:( label:str , color:turple ) ] # change color for specific object.
            }

    ```

* Usage
    ```python

        #step 1 : setting
        data = {  
            "draw_bbox" : True ,    
            "draw_result" : True , 
            "draw_area": True, 
            "palette": [ ( "car" , (255,0,166) ) ] 
        } 
        
        #step 2 : call set_draw()
        app.set_draw(data)
    ```
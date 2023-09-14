# IVIT-I Application of Movement Zone
## Usage
You need to follow the step below to use application:  
Step 1. [Setting Config](#setting-app-config).  
Step 2. [Create Instance](#create-instance).  
Step 3. Follow the [format of input parameter](#format-of-input-parameter) to use application.  
Other features : Different situation needs different value of "trancking distance".You can follow [here](#adjust-trancking-distance) to adjust trancking distance.  
And the description of application output is [here](#application-output).   

Note:If you want to use this application make sure you have filterpy==1.4.5 and lap==0.4.0 in your environment.

More function :  
1. User can control anythong about draw through function [set_draw()](#control-anything-about-draw).
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
|line_point(*)|dict|{ }|Seting the location of trigger line.|
|line_relation(*)|list|[ ]||
|events|dict|{ }|Conditions for a trigger event ·|
|draw_result|bool|True|Display information of detection.|
|draw_bbox|bool|True|Display boundingbox.|
* Basic Sample
    ```json
        {
            "application": {
                "areas": [
                    {
                        "name": "area",
                        "depend_on": [
                        ],
                        "line_point": {
                            "line_1": [
                                [
                                    0.16666666666,
                                    0.74074074074
                                ],
                                [
                                    0.57291666666,
                                    0.62962962963
                                ]
                            ],
                            "line_2": [
                                [
                                    0.26041666666,
                                    0.83333333333
                                ],
                                [
                                    0.72916666666,
                                    0.62962962963
                                ]
                            ],
                        },
                        "line_relation": [
                            {
                                "name": "Wrong Direction",
                                "start": "line_2",
                                "end": "line_1"
                            }
                        ],
                    }
                ]
            }
        }

    ```
* Advanced Sample (Traffic Flow)

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
                        "name": "area",
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
                        "line_point": {
                            "line_1": [
                                [
                                    0.14666666666,
                                    0.45074074074
                                ],
                                [
                                    0.40291666666,
                                    0.35962962963
                                ]
                            ],
                            "line_2": [
                                [
                                    0.14666666666,
                                    0.55074074074
                                ],
                                [
                                    0.50291666666,
                                    0.45962962963
                                ]
                            ],
                        },
                        "line_relation": [
                            {
                                "name": "To Taipei",
                                "start": "line_2",
                                "end": "line_1"
                            },
                            {
                                "name": "To Keelung",
                                "start": "line_1",
                                "end": "line_2"
                            }
                        ],
                        "events": {
                            "title": "Detect the traffic flow between Taipei and Keelung ",
                            "logic_operator": ">",
                            "logic_value": 2,
                        },
                    }
                ],
                "draw_result":False,
                "draw_bbox":False
            }
        }
   ``` 
## Create Instance
You need to use [app_config](#setting-app-config) and label path to create instance of application.
   ```python
    from apps import Movement_Zone

    app = Movement_Zone( app_config, label_path )

    #If you want to change the folder name of saving event image.  
    from apps import Movement_Zone 

    app = Movement_Zone( app_config , label_path ,event_save_folder="test")

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
## Adjust Trancking distance
*Trancking distance is a paremeter that we use to track object in different frame,but different situation we need to adjust differet value ,we can adjust tracking distance as below. 

        new_tracking_distance = 100 #defalt 60
        app.update_tracking_distance(new_tracking_distance)


## Application output 
* Application will return frame(already drawn) and organized information.The format of organized information as below.
    ```python
    #common output
    app_output = {
                    'areas': [
                            {
                    'id': 0, 
                    'name': 'The defalt area', 
                    'data': [
                                    {'label': 'To Taipei', 'num': 1
                                    },
                                    {'label': 'To Keelung', 'num': 2
                                    }
                                ]
                            }
                        ]
                }
    
    #triggering event 
    event output= {
                    'event': [
                            {
                    'uuid': 'd292a313-', 
                    'title': 'Detect the traffic flow between Taipei and Keelung ', 
                    'areas': {
                    'id': 0, 
                    'name': 'The defalt area', 
                    'data': [
                                        {'label': 'To Taipei', 'num': 1
                                        },
                                        {'label': 'To Keelung', 'num': 2
                                        }
                                    ]
                                }, 
                    'timesamp': datetime.datetime(2023,
                                4,
                                13,
                                10,
                                21,
                                59,
                                131903), 
                    'screenshot': {
                    'overlay': './d292a313-/2023-04-13 10: 21: 59.131903.jpg', 'original': './d292a313-/2023-04-13 10: 21: 59.131903_org.jpg'
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
4. show tracking tag.
5. show line that use to decide flow about object. 
6. change color for specific object.   

* Before we use set_draw() , we need to set parameter like bellow.  
    ```bash

        data:dict = {  
                draw_bbox : bool ,  # Control bounding box whether draw or not draw.
                draw_result : bool , # Control result box whether draw or not draw.
                draw_area : bool , # Control detection zone whether draw or not draw.
                draw_tracking : bool, # Control tracking tag whether draw or not draw.
                draw_line : bool , # Control line that use to decide flow about object whether draw or not draw.
                palette : (dict) { label(str) : color(Union[tuple, list]) }  # change color for specific object.
            }

    ```

* Usage
    ```python

        #step 1 : setting
        data = {  
            "draw_bbox" : True ,    
            "draw_result" : True , 
            "draw_area": True, 
            "draw_tracking" : True,
            "draw_line" : True,
            "palette":{
                "car":(100,25,60)
                } 
        } 
        
        #step 2 : call set_draw()
        app.set_draw(data)
    ```
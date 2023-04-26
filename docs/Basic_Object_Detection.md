# IVIT-I Application of basic Object detection
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
|areas(*)|list|[  ]|Seting the location of detection area. **(Basic Object detection only support 1 area(full screen).)**|
|name|str|default|Area name.**(Basic Object detection no need to set.)**|
| depend_on (*) | list | [ ] | The application depend on which label. |
| palette | dict | { } | Custom the color of each label. |
|draw_result|bool|True|Display information of detection.|
|draw_bbox|bool|True|Display boundingbox.|
* Basic
    ```bash
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
* Custom

   ```json
    {
        "application": {
            "areas": [
                {
                    "name": "default",
                    "depend_on": [
                        "car",
                        "truck"
                    ],
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
                    }
                }
            ]
        }
    }
   ``` 
## Create Instance
You need to use [app_config](#setting-app-config) and label path to create instance of application.
   ```python
    from apps import Basic_Object_Detection 

    app = Basic_Object_Detection( app_config , label_path )
   
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
        detection.ymax   # (type int)           value : 50   
    ```

## Application output 
* Application will return frame(already drawn) and organized information.The format of organized information as below.
    ```bash
        {
            'areas': [
                {
                    'id': 0, 
                    'name': 'default', 
                    'data': [
                        {
                            'xmin': 31, 
                            'ymin': 217, 
                            'xmax': 467, 
                            'ymax': 466, 
                            'label':'car', 
                            'score': 0.8984614964862956, 
                            'id': 0
                        }
                    ]
                }
            ]
        }
    
    ```
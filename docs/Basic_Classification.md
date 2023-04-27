# IVIT-I Application of basic classfication
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
|areas(*)|list|[  ]|Seting the location of detection area. **(Basic Classidication only support 1 area(full screen).)**|
|name|str|default|Area name.**(Basic Classidication no need to set.)**|
| depend_on (*) | list | [ ] | The application depend on which label. |
| palette | dict | { } | Custom the color of each label. |
* Sample

   ```json
    {
        "application": {
            "palette": {
                        "airplane": [
                            255,
                            255,
                            255
                        ],
                        "warpalne": [
                            0,
                            0,
                            0
                        ]
                    },
            "areas": [
                {
                    "name": "default",
                    "depend_on": [
                        "airplane",
                        "warplane"
                    ],
                    
                }
            ]
        }
    }
   ``` 
## Create Instance
You need to use [app_config](#setting-app-config) and label path to create instance of application.
   ```python
    from apps import Basic_Classification

    app = Basic_Classification( app_config, label_path )
   ``` 
## Format of input parameter
* Input parameters are the result of model predict, and the result must packed like below.

| Type | Description |
| --- | --- |
|tuple|( id, label, score )|
* Example:
    ```bash
        id      # (type int)           value : 0   
        label   # (type str)           value : cat   
        score   # (type numpy.float32) value : 0.5921569    
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
                    {'label': 'dog', 'score': 0.24705884
                    },
                    {'label': 'cat', 'score': 0.7568628
                    }
                ]
            }
        ]
    }
    
    ```
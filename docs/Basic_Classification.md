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
* Basic Sample
   ```bash
     {
        "application": {
            "areas": [
                {
                    "name": "default",
                    "depend_on": [ ]
                }
            ]
        }
    }

   ```
* Advanced Sample

   ```bash
    {
        "application": {
            "palette": {
                "Egyptian cat": [
                    0,
                    0,
                    0
                ],
                "tabby, tabby cat": [
                    255,
                    0,
                    0
                ]
            },
            "areas": [
                {
                    "name": "default",
                    "depend_on": ['Egyptian cat','tabby, tabby cat'
                    ]
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

| Name |Type | Description |
|--- |--- | --- |
| Input|list|[ turple ( id:int , label:str, score:float ) , ... ]|

* Item Description  
    |Name|Example|Description|
    |---|---| --- |
    |id|0|The sort index of the labels|
    |label|cat|The name of the predict label|
    |score|0.59221|The confidence of the prediction|

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
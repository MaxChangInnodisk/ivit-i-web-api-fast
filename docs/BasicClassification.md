# IVIT-I Application of basic classfication
## Usage
You need to follow the step below to use application:  
Step 1. [Init ivitAppHandler ](#init-ivitapphandler).  
Step 2. [Register Application](#register-application).  
Step 3. [Setting Config](#setting-app-config).  
Step 4. [Create Instance](#create-instance).  
Step 5. Follow the [format of input parameter](#format-of-input-parameter)  to use application.

And the description of application output is [here](#application-output). 


## Init ivitAppHandler 
Before starting , you must creat instance for iAPP_HANDLER.  
    

        # import register from ivit-i app
        from ivit_i.app import iAPP_HANDLER

        # creat instance for register
        app_handler = iAPP_HANDLER()

    
## Register Application
After you have created instance for iAPP_HANDLER , you can use register in iAPP_HANDLER to register your application.

        #BasicClassification -> Your application (instance) name during you use ivit-app.
        #CustomClsApp -> Your application (class).

        app_handler.register( BasicClassification , BasicClassification ) 


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

   ```bash
   {
    "application": {

                    "areas":[
                                {
                                    "name": "default",
                                    "depend_on":[ 
                                                    "airplane", 
                                                    "warplane" 
                                                ],
                                    "palette": {
                                                    "airplane": [255, 255, 255],
                                                    "warpalne": [0, 0, 0],
                                                }
                                }
                            ]
                   }
    }
   ``` 
## Create Instance
You need to use [app_config](#setting-app-config) and label path to create instance of application.
   ```bash

    app = app_handler.get_app("BasicClassification")( config , label_path )
    
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
        'areas':[
                    {
                        'id': 0, 
                        'name': 'default', 
                        'data': [
                                    {'label': 'dog', 'score': 0.24705884}, 
                                    {'label': 'cat', 'score': 0.7568628}
                                ]
                    }
                ]
    }
    
    ```
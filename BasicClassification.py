import sys, os, cv2
import numpy as np 
sys.path.append( os.getcwd() )
print(sys.path)
from ivit_i.app.common import ivitApp

class BasicClassification(ivitApp):
    """ __init__, __call__ """

    def __init__(self, params=None, label=None, palette=None, log=True):
        self.params = params
        self.app_type = 'cls'
        self.depend_on = []
        self.palette= {}
        self.depend_on = self.params['application']['areas'][0]['depend_on']
  
    def check_depend(self, label):
        ret = True
       
        if len(self.depend_on)>0:
            ret = (label in self.depend_on)
        return ret

    def __call__(self, frame, data, draw=True) -> tuple:
        
        app_output = { "areas":[{"id":0,"name":"default","data":[]}] }

        for id,det in enumerate(data["detections"]):

            label, score =[ det[key] for key in ["label", "score" ]  ]

            # Checking Depend
            if not self.check_depend(label): continue

            # Draw something                
            content     = '{} {:.1%}'.format(label, det['score'])
            ( text_width, text_height), text_base \
                = cv2.getTextSize(content, self.FONT, self.FONT_SCALE, self.FONT_THICKNESS)
            xmin        = 10
            ymin        = 10 + len(app_output['areas'][0]['data'])*(text_height+text_base)
            xmax        = xmin + text_width
            ymax        = ymin + text_height  
            app_output['areas'][0]['data'].append({"label":label,"score":score})


            if self.params['application']['areas'][0].__contains__('palette')==False or self.params['application']['areas'][0]['palette']=={}  :
                cur_color=[0,0,255]         
            else:   
                cur_color = self.params['application']['areas'][0]['palette'][label] if self.params['application']['areas'][0]['palette'].__contains__(label) else [0,0,0] 
            
            
            cv2.putText(
                frame, content, (xmin, ymax), self.FONT,
                self.FONT_SCALE, cur_color[0], self.FONT_THICKNESS, self.FONT_THICK
            )      

        return frame, app_output



if __name__ == "__main__":

    import cv2, time
    from ivit_i.common.model import get_ivit_model

    # Define iVIT Model
    model_type = 'cls'
    model_conf = {
        "tag": model_type,
        "openvino": {
            "model_path": "./model/resnet-v1/resnet_v1_50_inference.xml",
            "label_path": "./model/resnet-v1/imagenet.names",
            "device": "CPU",
            "thres": 0.98,
            "number_top": 3,
        }
    }

    ivit = get_ivit_model(model_type)
    ivit.load_model(model_conf)
    
    # Def Application
    app_config = {
        "application": {
		"name": "BasicClassification",
		"areas": [
				{
						"name": "default",
						"depend_on": [ "Egyptian cat","tabby, tabby cat"],
						"palettes": {
								"tabby, tabby cat":[255,0,26],
								"warpalne": [0, 0, 0],
						}
				}
		]
}
    }
    app = BasicClassification(params=app_config, label=model_conf['openvino']['label_path'])

    # Get Source
    data = './data/cat.jpg'
    cap = cv2.VideoCapture(data)
    while(cap.isOpened()):

        ret, frame = cap.read()
        if not ret: 
            cap = cv2.VideoCapture(data)
            time.sleep(1)
            continue

        output = ivit.inference(frame=frame)

        frame, info = app(frame, output)

        print(info)
        cv2.imshow('Test', frame)
        if cv2.waitKey(1) in [ ord('q'), 27 ]: break

    cap.release()
    ivit.release()

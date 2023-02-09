import sys, os, cv2
import numpy as np 
sys.path.append( os.getcwd() )
print(sys.path)
from ivit_i.app.common import ivitApp

class BasicClassification(ivitApp):
    
    """
    1. 繼承 ivitApp
    2. 必須要有 self.app_type, 跟 get_type() 讓 ivitAppHandler 取得到該 App ，支援的是 cls 還是 obj
    3. 必須要用 __call__(self, frame, data, draw=True)
    """

    def __init__(self, params=None, label=None, palette=None, log=True):
        """
        1. 設定 app_type
        2. 先設定所有 label 的顏色
        3. 

        """
        self.params = params
        self.app_type = 'cls'
        
        # Parse Parameters ( Config )
        # for idx, area in enumerate(self.params['areas']):
        #     print('Area: {}'.format(idx))

            # for key, val in area.items():
                # print('{}: {}'.format(key, val))

    def get_type(self):
        return self.app_type
                    
    def __call__(self, frame, data, draw=True) -> tuple:
        
        output_result={}
        temp_save_info={}
        for id,det in enumerate(data["detections"]):
            label, score =[ det[key] for key in ["label", "score" ]  ]
            temp_save_info.update({"label":label,"score":score})
            content     = '{} {:.1%}'.format(label, det['score'])
            ( text_width, text_height), text_base \
                = cv2.getTextSize(content, self.FONT, self.FONT_SCALE, self.FONT_THICKNESS)
            xmin        = 10
            ymin        = 10 + id*(text_height+text_base)
            xmax        = xmin + text_width
            ymax        = ymin + text_height
            depend_on = self.params['areas'][0]['depend_on']
            if len(depend_on)>0:
                if label not in depend_on: continue
            cur_color = self.params['areas'][0]['palette'].get(
                label, [0,0,0] )
            
            cv2.putText(
                frame, content, (xmin, ymax), self.FONT,
                self.FONT_SCALE, cur_color, self.FONT_THICKNESS, self.FONT_THICK
            )
        output_result.update({"results":[temp_save_info]})

            
        return frame, output_result



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
        "name": "BasicClassification",
        "areas": [
            {
                "name": "first_area",
                "depend_on": [ ],
                "area_point": [], 
                "palette":{ 'Egyptian cat':[15,255,255] ,'tabby, tabby cat':[0,0,255]},
            }
        ]
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

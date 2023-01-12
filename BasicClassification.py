import sys, os
import numpy as np 
sys.path.append( os.getcwd() )
from ivit_i.app.common import ivitApp

class BasicClassification(ivitApp):
    
    def __init__(self, params=None, label=None, palette=None, log=True):
        super().__init__(params, label, palette, log)
        self.set_type('cls')
        
    def init_params(self):
        self.def_param( name='depend_on', type='list', value=['car'] )
        self.def_param( name='color', type='list', value=[0,255,0] )
        self.def_param( name='opacity', type='float', value=0.3 )

    def run(self, frame, data, draw=True) -> tuple:

        info = []

        for idx, det in enumerate(data['detections']):

            # Parse output            
            label       = det['label'].split(',')[0]
            content     = '{} {:.1%}'.format(label, det['score'])
            ( text_width, text_height), text_base \
                = cv2.getTextSize(content, self.FONT, self.FONT_SCALE, self.FONT_THICKNESS)

            xmin        = 10
            ymin        = 10 + idx*(text_height+text_base)
            xmax        = xmin + text_width
            ymax        = ymin + text_height
            
            # Update Infor
            info.append(content)

            # Draw Top N label
            if not draw: continue
            
            # Half Opacity Black Background 
            text_area = frame[ymin:ymax, xmin:xmax]
            black_img = np.zeros(text_area.shape, dtype=np.uint8)
            frame[ymin:ymax, xmin:xmax] = cv2.addWeighted(text_area, 1-self.get_param('opacity'), black_img, self.get_param('opacity'), 1.0)
            
            # Draw Text
            cv2.putText(
                frame, content, (xmin, ymax), self.FONT,
                self.FONT_SCALE, self.get_param('color'), self.FONT_THICKNESS, self.FONT_THICK
            )

        return ( frame, info)

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
    app = BasicClassification(label=model_conf['openvino']['label_path'])

    # Get Source
    data = './data/image-slide.mp4'
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

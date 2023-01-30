import sys, os, cv2
import numpy as np
sys.path.append( os.getcwd() )
from ivit_i.app.common import ivitApp

class BasicObjectDetection(ivitApp):
    """ Basic Object Detection Application
    * Parameters
        1. depend_on
    * Function
        1. depend_label()
    """
    def __init__(self, params=None, label=None, palette=None, log=True):
        super().__init__(params, label, palette, log)
        self.set_type('obj')

    def init_params(self):
        self.def_param( name='depend_on', type='list', value=[], descr='add label into list if need to filter interest label' )
    
    @staticmethod
    def depend_label(label:str, interest_labels:list):
        """ Custom function for filter uninterest label """
        if interest_labels == []:
            return True
        # Not setup interest labels
        return (label in interest_labels)

    def custom_function(self, frame, label, color:tuple, left_top:tuple, right_down:tuple):
        """ The draw method customize by user 
        Do something you want to do
        """
        
        (xmin, ymin), (xmax, ymax) = left_top, right_down
        

        # Draw bounding box
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color , 2)

        
        # Draw Text
        text_base = cv2.getTextSize(label, self.FONT, self.FONT_SCALE, 2)[1]
        cv2.putText(
            frame, label, (xmin, ymin-(text_base)), self.FONT,
            self.FONT_SCALE, color, 2, self.FONT_THICK
        )

        return frame

    def run(self, frame, data, draw=True) -> tuple:

        for det in (data['detections']):

            # Check Label is what we want
            if not self.depend_label(det['label'], self.get_param('depend_on')):
                continue
            
            # Parsing output
            ( label, score, xmin, ymin, xmax, ymax ) \
                 = [ det[key] for key in [ 'label', 'score', 'xmin', 'ymin', 'xmax', 'ymax' ] ]                  
            
            # Draw Top N label
            if not draw: continue
            
            # Further Process
            frame = self.custom_function(
                frame = frame,
                label = '{} {:.1%}'.format(label, score),
                color = self.get_color(label),
                left_top = (xmin, ymin),
                right_down = (xmax, ymax)
            )
        
        return ( frame, data['detections'])

if __name__ == "__main__":

    import cv2
    from ivit_i.common.model import get_ivit_model

    # Define iVIT Model
    model_type = 'obj'
    model_anchor = [ 10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326 ]
    model_conf = { 
        "tag": model_type,
        "openvino": {
            "model_path": "./model/yolo-v3-tf/FP32/yolo-v3-tf.xml",
            "label_path": "./model/yolo-v3-tf/coco.names",
            "anchors": model_anchor,
            "architecture_type": "yolo",
            "device": "CPU",
            "thres": 0.9
        }
    }

    ivit = get_ivit_model(model_type)
    ivit.load_model(model_conf)
    
    # Def Application
    app = BasicObjectDetection(label=model_conf['openvino']['label_path'])

    # Get Source
    cap = cv2.VideoCapture('./data/4-corner-downtown.mp4')
    while(cap.isOpened()):

        ret, frame = cap.read()
        if not ret: break
        output = ivit.inference(frame=frame)

        frame, info = app(frame, output)
        
        print(info)
        cv2.imshow('Test', frame)
        if cv2.waitKey(1) in [ ord('q'), 27 ]: break

    cap.release()
    ivit.release()

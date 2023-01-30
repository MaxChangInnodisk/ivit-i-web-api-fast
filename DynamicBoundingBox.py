import sys, os, cv2, logging, time
import numpy as np
sys.path.append( os.getcwd() )
from ivit_i.app.common import ivitApp
try:
    from BasicObjectDetection import BasicObjectDetection 
except:
    from apps.BasicObjectDetection import BasicObjectDetection

class DynamicBoundingBox(BasicObjectDetection, ivitApp):
    """ A dynamic bounding box application.
    
    Provide helper function to update draw parameters (`self.thick`, `self.font_size`, `self.font_thick`)
        1. Add `init_draw_params()` in `__init__`.
        2. Add `update_draw_params()` in `run()` at first.
    """
    def __init__(self, params=None, label=None, palette=None, log=True):
        super().__init__(params, label, palette, log)
        self.set_type('obj')
        self.init_draw_params()

    def init_params(self):
        self.def_param( name='depend_on', type='list', value=[], descr='add label into list if need to filter interest label' )
        
    def init_draw_params(self):
        """ Initialize Draw Parameters """
        self.frame_idx = 0
        self.frame_size = None
        self.font_size  = None
        self.font_thick = None
        self.thick      = None

    def update_draw_param(self, frame):
        """ Update the parameters of the drawing tool, which only happend at first time. """
        
        # if frame_size not None means it was already init 
        if( self.frame_idx > 1): return None

        # Parameters
        FRAME_SCALE     = 0.0005    # Custom Value which Related with Resolution
        BASE_THICK      = 1         # Setup Basic Thick Value
        BASE_FONT_SIZE  = 0.5   # Setup Basic Font Size Value
        FONT_SCALE      = 0.2   # Custom Value which Related with the size of the font.

        # Get Frame Size
        self.frame_size = frame.shape[:2]
        
        # Calculate the common scale
        scale = FRAME_SCALE * sum(self.frame_size)
        
        # Get dynamic thick and dynamic size 
        self.thick  = BASE_THICK + round( scale )
        self.font_thick = self.thick//2
        self.font_size = BASE_FONT_SIZE + ( scale*FONT_SCALE )
        
        logging.info('Frame: {} ({}), Get Border Thick: {}, Font Scale: {}, Font Thick: {}'
            .format(self.frame_size, scale, self.thick, self.font_size, self.font_thick))
                    
    def custom_function(self, frame, label, score, color:tuple, left_top:tuple, right_down:tuple):
        
        (xmin, ymin), (xmax, ymax) = left_top, right_down
        info = '{} {:.1%}'.format(label, score)

        # Draw bounding box
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color , self.thick)

        # Draw Text
        (t_wid, t_hei), t_base = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, self.font_size, self.font_thick)
        t_xmin, t_ymin, t_xmax, t_ymax = xmin, ymin-(t_hei+(t_base*2)), xmin+t_wid, ymin
        cv2.rectangle(frame, (t_xmin, t_ymin), (t_xmax, t_ymax), color , -1)

        cv2.putText(
            frame, info, (xmin, ymin-(t_base)), cv2.FONT_HERSHEY_SIMPLEX,
            self.font_size, (255,255,255), self.font_thick, cv2.LINE_AA
        )

        return frame, info

    def run(self, frame, data, draw=True) -> tuple:
        
        self.frame_idx += 1

        self.update_draw_param(frame=frame)
        
        results = []
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
            frame, info = self.custom_function(
                frame = frame,
                label = label,
                score = score,
                color = self.get_color(label),
                left_top = (xmin, ymin),
                right_down = (xmax, ymax)
            )
            results.append(info)

        return ( frame, results)

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
    ivit.set_async_mode()
    
    # Def Application
    app = DynamicBoundingBox(label=model_conf['openvino']['label_path'])

    # Get Source
    src_path = '/dev/video0' # 640x480
    src_path = './data/car.mp4'   # 1280x720
    src_path = './data/4-corner-downtown.mp4' # 1920x1080
    cap = cv2.VideoCapture(src_path)

    output = None

    while(cap.isOpened()):

        ret, frame = cap.read()
        if not ret: break
        _output = ivit.inference(frame=frame)
        output = _output if _output is not None else output 
        if (output):
            frame, info = app(frame, output)
        cv2.imshow('Test', frame)
        if cv2.waitKey(1) in [ ord('q'), 27 ]: break

    cap.release()
    ivit.release()

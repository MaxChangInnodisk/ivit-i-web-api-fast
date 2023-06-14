import cv2
import logging
import numpy as np
from apps.palette import palette
from typing import Union, get_args
from ivit_i.common.app import iAPP_CLS

class Basic_Classification(iAPP_CLS):

    def __init__(self, params:dict, label:str, palette:dict=palette):
        """
        Basic_Classification .
        Args:
            params (dict, optional): _description_. Defaults to None.
            label (str, optional): _description_. Defaults to None.
            palette (dict, optional): _description_. Defaults to palette.
        """
        self.params = params
        self.app_type = 'cls'
        self.depend_on = self.params['application']['areas'][0]['depend_on']
        self.palette= {}
        self.label_path = label
        self.label_list =[]
        self._init_palette(palette)
        self.FONT            = cv2.FONT_HERSHEY_SIMPLEX
        self.FONT_SCALE      = 1
        self.FONT_THICK      = cv2.LINE_AA
        self.FONT_THICKNESS  = 1
        

    def _init_palette(self,palette:dict):
        """
        We will deal all color we need there.
        Step 1 : assign color for each label.
        Step 2 : if app_config have palette that user stting we will change the color follow user setting.
        Args:
            palette (dict): _description_
        """
        color = None
        with open(self.label_path,'r') as f:
            for idx, line in enumerate(f.readlines()):
                #idx in palette begin to 1.
                idx+=1
                if self.params['application'].__contains__('palette'):
                    
                    if self.params['application']['palette'].__contains__(line.strip()):
                        color = self.params['application']['palette'][line.strip()]
                    else:
                        color = palette[str(idx)]
                else :         
                    color = palette[str(idx)]
                
                self.palette.update({line.strip():color})
                self.label_list.append(line.strip())  
        
    def get_color(self, label:str):
        """
            Get color of label.
        Args:
            label (str): label of object.

        Returns:
            list: (B,G,R).
        """
        return self.palette[label]        
        
    def _check_depend(self, label:str):
        """
            Check label whether in the depend on or not.
        Args:
            label (str): label of model predict.

        Returns:
            bool : label whether in the depend on or not.
        """
        ret = True
       
        if len(self.depend_on)>0:
            ret = (label in self.depend_on)
        return ret

    def set_draw(self,params:dict):
        """
        Control anything about drawing.
        Which params you can contral :

        {  
            palette (dict) { label(str) : color(Union[tuple, list]) },
        }
        
        Args:
            params (dict): 
        """
        color_support_type = Union[tuple, list]
        palette = params.get('palette', None)
        if isinstance(palette, dict):
            if len(palette)==0:
                logging.warning("Not set palette!")
                pass
            else:
                for label,color in palette.items():

                    if isinstance(label, str) and isinstance(color, get_args(color_support_type)):
                        if self.palette.__contains__(label):
                           self.palette.update({label:color})
                        else:
                            logging.error("Model can't recognition the label {} , please checkout your label!.".format(label))
                        logging.info("Label: {} , change color to {}.".format(label,color))
                    else:
                        logging.error("Value in palette type must (label:str , color :Union[tuple , list] ),your type \
                                      label:{} , color:{} is error.".format(type(label),type(color)))
        else:
            logging.error("Not set palette or your type {} is error.".format(type(palette)))

    def __call__(self, frame:np.ndarray, detections:list):
        """

        Args:
            frame (np.ndarray): The img that we want to deal with.
            detections (list): output of model predict

        Returns:
            tuple:We will return the frame that finished painting and sort out infomation.
        """
        
        app_output = { "areas":[{"id":0,"name":"default","data":[]}] }

        for idx, label, score in detections:
           
            # Checking Depend
            if not self._check_depend(label): continue

            # Draw something                
            content     = '{} {:.1%}'.format(label, score)
            ( text_width, text_height), text_base \
                = cv2.getTextSize(content, self.FONT, self.FONT_SCALE, self.FONT_THICKNESS)
            xmin        = 10
            ymin        = 10 + len(app_output['areas'][0]['data'])*(text_height+text_base)
            xmax        = xmin + text_width
            ymax        = ymin + text_height  
            app_output['areas'][0]['data'].append({"label":label,"score":score})

            cur_color = self.get_color(label)
            

            cv2.putText(
                frame, content, (xmin, ymax), self.FONT,
                self.FONT_SCALE, cur_color, self.FONT_THICKNESS, self.FONT_THICK
            )      

        return ( frame, app_output, {} )



if __name__=='__main__':
    import logging as log
    import cv2, sys
    from argparse import ArgumentParser, SUPPRESS
    from ivit_i.io import Source, Displayer
    from ivit_i.core.models import iClassification
    from ivit_i.common import Metric

    def build_argparser():

        parser = ArgumentParser(add_help=False)

        basic_args = parser.add_argument_group('Basic options')
        basic_args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
        basic_args.add_argument('-m', '--model', required=True, help='the path to model')
        basic_args.add_argument('-i', '--input', required=True,
                        help='Required. An input to process. The input must be a single image, '
                            'a folder of images, video file or camera id.')
        basic_args.add_argument('-l', '--label', help='Optional. Labels mapping file.', default=None, type=str)
        basic_args.add_argument('-d', '--device', type=str,
                        help='Optional. `Intel` support [ `CPU`, `GPU` ] \
                                `Hailo` is support [ `HAILO` ]; \
                                `Xilinx` support [ `DPU` ]; \
                                `dGPU` support [ 0, ... ] which depends on the device index of your GPUs; \
                                `Jetson` support [ 0 ].' )
        
        model_args = parser.add_argument_group('Model options')
        model_args.add_argument('-t', '--confidence_threshold', default=0.1, type=float,
                                    help='Optional. Confidence threshold for detections.')
        model_args.add_argument('-topk', help='Optional. Number of top results. Default value is 5. Must be from 1 to 10.', default=5,
                                    type=int, choices=range(1, 11))

        io_args = parser.add_argument_group('Input/output options')
        io_args.add_argument('-n', '--name', default='ivit', 
                            help="Optional. The window name and rtsp namespace.")
        io_args.add_argument('-r', '--resolution', type=str, default=None, 
                            help="Optional. Only support usb camera. The resolution you want to get from source object.")
        io_args.add_argument('-f', '--fps', type=int, default=None,
                            help="Optional. Only support usb camera. The fps you want to setup.")
        io_args.add_argument('--no_show', action='store_true',
                            help="Optional. Don't display any stream.")

        args = parser.parse_args()
        # Parse Resoltion
        if args.resolution:
            args.resolution = tuple(map(int, args.resolution.split('x')))

        return args
    
    # 1. Argparse
    args = build_argparser()

    # 2. Basic Parameters
    infer_metrx = Metric()
    
    # 3. Init Model
    model = iClassification(
        model_path = args.model,
        label_path = args.label,
        confidence_threshold = args.confidence_threshold,
        device=args.device,
        topk = args.topk )
    
    # 4. Init Source
    src = Source(   
        input = args.input, 
        resolution = args.resolution, 
        fps = args.fps )
    
    # 5. Init Display
    if not args.no_show:
        dpr = Displayer( cv = True )
    
    # 6. Setting iApp
    app_config =   {
                        "application": {
                            "areas": [
                                {
                                    "name": "default",
                                    "depend_on": []
                                }
                            ]
                        }
                    }
    
    app = Basic_Classification(app_config,args.label)
    
    # 7. Start Inference
    try:
        while(True):
            # Get frame & Do infernece
            frame = src.read()       
            detections = model.inference( frame )

            frame , app_output , event_output =app(frame,detections)
    
            # Draw FPS: default is left-top                     
            infer_metrx.paint_metrics(frame)
            
            # Display
            dpr.show(frame=frame)                   
            if dpr.get_press_key()==ord('q'):
                break

            # Update Metrix
            infer_metrx.update()

    except KeyboardInterrupt: 
        log.info('Detected Key Interrupt !')

    finally:
        model.release()     # Release Model
        src.release()       # Release Source
        dpr.release()   # Release Display
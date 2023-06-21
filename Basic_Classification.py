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
        self.WIDTH_SPACE = 10
        self.HIGHT_SPACE = 10
        self.OPACITY = 0.4
    
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
    
    def _update_draw_param(self,frame:np.ndarray):
        """
            Accourding to the resolution of frame to modify draw size.
        Args:
            frame (np.ndarray): input image.

        """
        WIDTH_SCALE=(frame.shape[1]//640) if (frame.shape[1]//640)>=1 else 1
        HIGHT_SCALE=(frame.shape[0]//640) if (frame.shape[0]//640)>=1 else 1

        self.FONT_SCALE = 1 *((WIDTH_SCALE+HIGHT_SCALE)//2)
        self.FONT_THICKNESS = 2 * ((WIDTH_SCALE+HIGHT_SCALE)//2)
        self.WIDTH_SPACE = 10 * WIDTH_SCALE
        self.HIGHT_SPACE = 10 * HIGHT_SCALE

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
    
    def draw_app_result(self,frame,result:dict,id,color:tuple= (255,255,255)):
        outer_clor=color
        font_color = (255-color[0],255-color[1],255-color[2])
        overlay = frame.copy()

        (t_wid, t_hei), t_base = cv2.getTextSize(result, cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE, self.FONT_THICKNESS)
        
        t_xmin, t_ymin, t_xmax, t_ymax = int(self.WIDTH_SPACE), int(self.HIGHT_SPACE+self.HIGHT_SPACE*id+(id*(t_hei+t_base))), \
        int(self.WIDTH_SPACE+t_wid), int(self.HIGHT_SPACE+self.HIGHT_SPACE*id+((id+1)*(t_hei+t_base)))
        
        # cv2.rectangle(frame, (t_xmin, t_ymin), (t_xmax, t_ymax+t_base), outer_clor , -1)
        point = np.array([[t_xmin, t_ymin],[t_xmax,t_ymin],[t_xmax, t_ymax+t_base],[t_xmin,t_ymax+t_base]])

        
        
        cv2.fillPoly(overlay, pts=[point], color=outer_clor)
        frame = cv2.addWeighted( frame, 1-self.OPACITY, overlay, self.OPACITY, 0 )

       
        
        return cv2.putText(frame, result, (t_xmin, t_ymax), cv2.FONT_HERSHEY_SIMPLEX,
            self.FONT_SCALE, font_color, self.FONT_THICKNESS, cv2.LINE_AA)
        
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
        self._update_draw_param(frame)
        
        app_output = { "areas":[{"id":0,"name":"default","data":[]}] }
        
        object_count = 0
        for idx, label, score in detections:
            
            # Checking Depend
            if not self._check_depend(label): continue
            
            # Draw something                
            content     = ' {} {:.1%} '.format(label, score)
            cur_color = self.get_color(label)
            frame = self.draw_app_result(frame,content,object_count,cur_color)
            
            object_count+=1
            app_output['areas'][0]['data'].append({"label":label,"score":score})
        # cv2.putText(frame, "asdasdasdasdas", (int(frame.shape[1]/2), int(frame.shape[0]/2)), cv2.FONT_HERSHEY_SIMPLEX,
        #     self.FONT_SCALE, (255,255,255), self.FONT_THICKNESS, cv2.LINE_AA)    
        # print(app_output)
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
            # infer_metrx.paint_metrics(frame)
            
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
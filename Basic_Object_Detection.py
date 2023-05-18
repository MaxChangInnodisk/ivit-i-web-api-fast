import sys, os, cv2 ,logging
sys.path.append( os.getcwd() )
from apps.palette import palette
from ivit_i.common.app import iAPP_OBJ
from typing import Union, get_args
class Basic_Object_Detection(iAPP_OBJ):    
    """ Basic Object Detection Application
    * Parameters
        1. depend_onarea_opacity
    * Function
        1. depend_label()
    """
    def __init__(self, params=None, label=None, palette=palette,log=True):
        self.app_type = 'obj'
        self.params = params
        
        if self.params:
            self.depend_on =self.params['application']['areas'][0]['depend_on']

        self.depend_on = []
        self.palette={}
        self.model_label = label
        self.model_label_list =[]
        
        self.init_palette(palette)
        self.init_draw_params()

    def init_draw_params(self):
        """ Initialize Draw Parameters """
        #for draw result and boundingbox
        self.frame_idx = 0
        self.frame_size = None
        self.font_size  = None
        self.font_thick = None
        self.thick      = None
        self.draw_result =self.params['application']['draw_result'] if self.params['application'].__contains__('draw_result') else True
        
        #for draw area
        self.area_name={}
        self.draw_bbox =self.params['application']['draw_bbox'] if self.params['application'].__contains__('draw_bbox') else True
        self.draw_area=True
        self.area_opacity=None
        self.area_color=None
        self.area_pts = {}
        self.area_cnt = {}

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

        self.area_color=[0,0,255]
        self.area_opacity=0.4  
    
    
 
    def custom_function(self, frame, color:tuple, label,score, left_top:tuple, right_down:tuple,draw_bbox=True,draw_result=True):
        """ The draw method customize by user 
        """
        (xmin, ymin), (xmax, ymax) = map(int, left_top), map(int, right_down)
        info = '{} {:.1%}'.format(label, score)
        # Draw bounding box
        draw_bbox = self.draw_bbox if self.draw_bbox is not None else draw_bbox
        if draw_bbox:
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color , self.thick)

        draw_result = self.draw_result if self.draw_result is not None else draw_result
        # Draw Text
        if draw_result:
            (t_wid, t_hei), t_base = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, self.font_size, self.font_thick)
            t_xmin, t_ymin, t_xmax, t_ymax = xmin, ymin-(t_hei+(t_base*2)), xmin+t_wid, ymin
            cv2.rectangle(frame, (t_xmin, t_ymin), (t_xmax, t_ymax), color , -1)
            cv2.putText(
                frame, info, (xmin, ymin-(t_base)), cv2.FONT_HERSHEY_SIMPLEX,
                self.font_size, (255,255,255), self.font_thick, cv2.LINE_AA
            )
        return frame
        
    def init_palette(self,palette):

        color = None
        with open(self.model_label,'r') as f:
            # lines = f.read().splitlines()
            for idx, line in enumerate(f.readlines()):
                idx+=1
                if self.params['application'].__contains__('palette'):
                    
                    if self.params['application']['palette'].__contains__(line.strip()):
                        color = self.params['application']['palette'][line.strip()]
                    else:
                        color = palette[str(idx)]
                else :         
                    color = palette[str(idx)]
                
                self.palette.update({line.strip():color})
                self.model_label_list.append(line.strip())

    def get_color(self, label):

        return self.palette[label] 
       
    def set_draw(self,params:dict):
        """
        Control anything about drawing.
        Which params you can contral :

        {  
            draw_bbox : bool ,
            draw_result : bool ,
            palette:list[ tuple:( label:str , color:Union[tuple , list]  ) ]
        }
        
        Args:
            params (dict): 
        """
        color_support_type = Union[tuple, list]
        if not isinstance(params, dict):
            logging.error("Input type is dict! but your type is {} ,please correct it.".format(type(params.get('draw_area', None))))
            return

        if isinstance(params.get('draw_bbox', self.draw_bbox) , bool):
            self.draw_bbox= params.get('draw_bbox', self.draw_bbox)
            logging.info("Change draw_bbox mode , now draw_bbox mode is {} !".format(self.draw_bbox))
        else:
            logging.error("draw_bbox type is bool! but your type is {} ,please correct it.".format(type(params.get('draw_bbox', self.draw_bbox))))
        
        if isinstance(params.get('draw_result', self.draw_result) , bool):    
            self.draw_result= params.get('draw_result', self.draw_result)
            logging.info("Change draw_result mode , now draw_result mode is {} !".format(self.draw_result))
        else:
            logging.error("draw_result type is bool! but your type is {} ,please correct it.".format(type(params.get('draw_result', self.draw_result))))
        

        palette = params.get('palette', None)
        if isinstance(palette, list):
            if len(palette)==0:
                logging.warning("Not set palette!")
                pass
            else:
                for info in palette:
                    (label , color) = info
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


    def __call__(self, frame, detections, draw=True) -> tuple:
        #collect depend_on for each area from config
        app_output={"areas":[{"id":0,"name":"default","data":[]}]}

        self.update_draw_param(frame=frame)

        # for id,det in enumerate(data['detections']):
        for detection in detections:
            # Check Label is what we want
            ( label, score, xmin, ymin, xmax, ymax ) \
                 = detection.label, detection.score, detection.xmin, detection.ymin, detection.xmax, detection.ymax  
            # if user have set depend on
            if len(self.depend_on)>0:
                              
                if label in self.depend_on[0] :

                    app_output['areas'][0]['data'].append({'xmin':xmin,'ymin':ymin,'xmax':xmax,'ymax':ymax,'label':label,'score':score,'id':detection.id})
                    frame = self.custom_function(
                        frame = frame,
                        color = self.get_color(label) ,
                        label = label,
                        score=score,
                        left_top = (xmin, ymin),
                        right_down = (xmax, ymax)
                    )
            else:
                app_output['areas'][0]['data'].append({'xmin':xmin,'ymin':ymin,'xmax':xmax,'ymax':ymax,'label':label,'score':score,'id':detection.id})
                
                frame = self.custom_function(
                            frame = frame,
                            color = self.get_color(label) ,
                            label = label,
                            score=score,
                            left_top = (xmin, ymin),
                            right_down = (xmax, ymax)
                        ) 
                                        
        return ( frame, app_output, {})


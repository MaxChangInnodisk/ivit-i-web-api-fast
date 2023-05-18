import sys, os, cv2
import numpy as np 
import logging
sys.path.append( os.getcwd() )
from apps.palette import palette
from ivit_i.common.app import iAPP_CLS
class Basic_Classification(iAPP_CLS):
    """ __init__, __call__ """

    def __init__(self, params=None, label=None, palette=palette, log=True):
        self.params = params
        self.app_type = 'cls'
        self.depend_on = []
        self.palette= {}
        self.model_label = label
        self.model_label_list =[]
        self.init_palette(palette)
        self.depend_on = self.params['application']['areas'][0]['depend_on']
        self.FONT            = cv2.FONT_HERSHEY_SIMPLEX
        self.FONT_SCALE      = 1
        self.FONT_THICK      = cv2.LINE_AA
        self.FONT_THICKNESS  = 1
        

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
                
                

        
        
    def set_color(self,label:str,color:tuple):
        """
        set color :

        sample of paremeter : 
            label = "dog"
            color = (0,0,255)
        """
        self.palette.update({label:color})
        logging.info("Label: {} , change color to {}.".format(label,color))
        
    def get_color(self, label):
        return self.palette[label]        
        
    def check_depend(self, label):
        ret = True
       
        if len(self.depend_on)>0:
            ret = (label in self.depend_on)
        return ret

    def __call__(self, frame, detections, draw=True) -> tuple:
        
        app_output = { "areas":[{"id":0,"name":"default","data":[]}] }

        for idx, label, score in detections:
           
            # Checking Depend
            if not self.check_depend(label): continue

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

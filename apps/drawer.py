import cv2
import numpy as np

# Parameters
FRAME_SCALE     = 0.0005    # Custom Value which Related with Resolution
BASE_THICK      = 1         # Setup Basic Thick Value
BASE_FONT_SIZE  = 0.5   # Setup Basic Font Size Value
FONT_SCALE      = 0.2   # Custom Value which Related with the size of the font.
WIDTH_SPACE = 10
HEIGHT_SPACE = 10
FONT_TYPE = cv2.FONT_HERSHEY_SIMPLEX
LINE_TYPE = cv2.LINE_AA

class DrawTool:
    """ Draw Tool for Label, Bounding Box, Area ... etc. """

    def __init__(self, labels:list, palette:dict) -> None:
        # palette[cat] = (0,0,255), labels = [ cat, dog, ... ]
        self.palette, self.labels = palette, labels
        
        # Initialize draw params
        self.font_size = 1
        self.font_thick = 1
        
        # bbox
        self.bbox_line = 1
        
        # circle
        self.circle_radius = 3

        # area
        self.area_thick = 3
        self.area_color = [ 0,0,255 ]
        self.area_opacity = 0.4          
        self.draw_params_is_ready = False

        # output
        self.out_color = [0, 255, 255]
        self.out_font_color = [0, 0, 0]

        # line
        self.line_color = [0, 255, 255]

    def update_draw_params(self, frame: np.ndarray) -> None:
        
        if self.draw_params_is_ready: return

        # Get Frame Size
        self.frame_size = frame.shape[:2]
        
        # Calculate the common scale
        scale = FRAME_SCALE * sum(self.frame_size)
        
        # Get dynamic thick and dynamic size 
        self.thick  = BASE_THICK + round( scale )
        self.font_thick = self.thick//2
        self.font_size = BASE_FONT_SIZE + ( scale*FONT_SCALE )
        self.width_space = int(scale*WIDTH_SPACE) 
        self.height_space = int(scale*HEIGHT_SPACE) 

        # Change Flag
        self.draw_params_is_ready = True
        print('Updated draw parameters !!!')

    def draw_areas(  self, 
                    frame: np.ndarray, 
                    areas: list, 
                    draw_point: bool= True,
                    draw_name: bool= True,
                    draw_line: bool= True,
                    name: str = None,
                    radius: int = None,
                    color: list = None, 
                    thick: int = None,
                    opacity: float = None) -> np.ndarray:

        radius = radius if radius else self.circle_radius
        color = color if color else self.area_color
        opacity = opacity if opacity else self.area_opacity
        thick = thick if thick else self.area_thick
        
        overlay = frame.copy()

        for area in areas:
            
            area_pts = area["area_point"]
            
            # draw poly
            cv2.fillPoly(overlay, pts=[ np.array(area_pts) ], color=color)

            # draw point and line if need
            prev_point_for_line = area_pts[-1]       # for line
            for point in area_pts:

                # draw point
                if draw_point:
                    cv2.circle(frame, tuple(point), radius, color, -1)
                
                # draw line
                if draw_line:
                    cv2.line(frame, point, prev_point_for_line, color, thick)
                    prev_point_for_line = point

                if draw_name and not name:
                    pass

        return cv2.addWeighted( frame, 1-opacity, overlay, opacity, 0 ) 

    def draw_bbox(self, frame: np.ndarray, left_top: list, right_bottom: list, color: list, thick: int= None) -> None:
        # Draw bbox
        thick = thick if thick else self.thick
        (xmin, ymin), (xmax, ymax) = map(int, left_top), map(int, right_bottom)
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color , thick)

    def draw_label(self, frame: np.ndarray, label: str, left_bottom: list, color: list, thick: int=None) -> None:
        # Draw label
        xmin, ymin = left_bottom
        thick = thick if thick else self.thick

        (t_wid, t_hei), t_base = cv2.getTextSize(label, FONT_TYPE, self.font_size, self.font_thick)
        t_xmin, t_ymin, t_xmax, t_ymax = xmin, ymin-(t_hei+(t_base*2)), xmin+t_wid, ymin
        cv2.rectangle(frame, (t_xmin, t_ymin), (t_xmax, t_ymax), color , -1)
        cv2.putText(
            frame, label, (xmin, ymin-(t_base)), FONT_TYPE,
            self.font_size, (255,255,255), self.font_thick, LINE_TYPE
        )

    def draw_area_results(self, frame: np.ndarray, areas: dict, color: list= None, font_color: list= None) -> None:
        color = color if color else self.out_color
        font_color = font_color if font_color else self.out_font_color
        
        cur_idx = 0
        for area in areas:
            area_name = area["name"]
            sorted_area_output = sorted(area["output"].items(), key=lambda x:x[1], reverse=True)
            
            for (cur_label, cur_nums) in sorted_area_output:
                
                result = f"{area_name} : {cur_nums} {cur_label}"
                
                (t_wid, t_hei), t_base = cv2.getTextSize(result, FONT_TYPE, self.font_size, self.font_thick)
                
                t_xmin = WIDTH_SPACE
                t_ymin = HEIGHT_SPACE + ( HEIGHT_SPACE*cur_idx) + (cur_idx*(t_hei+t_base))
                t_xmax = t_wid + WIDTH_SPACE
                t_ymax = HEIGHT_SPACE + ( HEIGHT_SPACE*cur_idx) + ((cur_idx+1)*(t_hei+t_base))
                
                cv2.rectangle(frame, (t_xmin, t_ymin), (t_xmax, t_ymax+t_base), color , -1)
                cv2.rectangle(frame, (t_xmin, t_ymin), (t_xmax, t_ymax+t_base), (0,0,0) , 1)
                cv2.putText(
                    frame, result, (t_xmin, t_ymax), FONT_TYPE,
                    self.font_size, font_color, self.font_thick, LINE_TYPE
                )
                cur_idx += 1

    def draw_line(self, frame:np.ndarray, line_point: tuple, line_name:str=None, color: tuple=None):

        color = color if color else self.line_color

        if line_name:
            cv2.putText( frame, line_name , line_point[1], FONT_TYPE,
                    self.font_size, color, self.font_thick, LINE_TYPE
                )
        cv2.line(frame, line_point[0], line_point[1], color, 3)

                
    def get_color(self, label:str) -> list:
        return self.palette[label]

    def get_size(self, text: str) -> tuple:
        (t_wid, t_hei), t_base = cv2.getTextSize(text, FONT_TYPE, self.font_size, self.font_thick)
        return (t_wid, t_hei), t_base




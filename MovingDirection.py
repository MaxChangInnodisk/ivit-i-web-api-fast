import os, sys, time, logging, math, cv2
import numpy as np 

sys.path.append( os.getcwd() )
from ivit_i.app.common import ivitApp
try:
    from DynamicBoundingBox import DynamicBoundingBox
    from Tracking import Tracking

except:
    from apps.DynamicBoundingBox import DynamicBoundingBox
    from apps.Tracking import Tracking

# Parameters
K_DETS          = "detections"
K_LABEL         = "label"
K_DEPEND        = "depend_on"
K_DRAW_BB       = "draw_bbox"
K_DRAW_INFO     = "draw_info"
K_DIS           = "distance"
K_BUFFER        = "buffer"
K_THRES         = "thres"
K_ARROW_LEN     = "arrow_length"

class MovingDirection( Tracking, DynamicBoundingBox, ivitApp):

    def __init__(self, params=None, label=None, palette=None, log=True):
        super().__init__(params, label, palette, log)
        self.set_type('obj')
        self.init_draw_params()

        self.init_vector_params()

    def init_params(self):
        self.def_param( name=K_DEPEND, type='list', value=['car'] )
        self.def_param( name=K_DRAW_BB, type='bool', value=True )
        self.def_param( name=K_DIS, type='int', value=20, descr='The limit of tracking distance')
        self.def_param( name=K_DRAW_INFO, type='bool', value=True, descr="Display the app information on the top left corner.")
        self.def_param( name=K_BUFFER, type='int', value=20, descr='The buffer of the average vector')
        self.def_param( name=K_ARROW_LEN, type='float', value=20, descr='The length of the arrow')

    def init_vector_params(self):
        """ Initialize Vecotr Parameters """
        self.track_obj_buf  = dict()
        self.track_obj_vec  = dict()

    def get_coord_distance(self, p1 , p2):
        return math.sqrt( ((int(p1[0])-int(p2[0]))**2)+((int(p1[1])-int(p2[1]))**2) )

    def get_angle_for_cv(self, pos1, pos2):
        """
        Calculate Vector's Angle for OpenCV Pixel
        Because the pixel position is reversed, Angle will be reversed, too
        """
        dx, dy = pos2[0] - pos1[0], pos2[1] - pos1[1]
        return int(math.atan2(dy, dx)*180/math.pi) *(-1)
        
    def update_vector(self, label, track_idx):
        """ Get average vector, return None if buffer is not enough.

        - args
            - label: which object
            - track_idx: the track index of the object
        - return
            - arrow_point
                - type: tuple
                - desc: return two point to draw arrow on frame
                - exam: (cur_x, cur_y), (prev_x, prev_y)
        - algorithm
            1. Record each vector
            2. Average vector when the buffer is enough, then will get two coordinate ( arrow_1, arrow_2 )
            3. Normalize the coordinate depend on the arrow_length defined on the top
            3. Return ( arrow_1, arrow_2 ) 
        """

        def helper(obj, label, track_idx):
            if obj.get(label) is None: obj.update( {label: {}} )
            if obj[label].get(track_idx) is None: obj[label].update( {track_idx: []} )

        # Update label and track_idx in self.track_obj_buf: { label: { 0: [ vec1, vec2, ..., vecN ] } }
        helper( self.track_obj_buf, label, track_idx )
        helper( self.track_obj_vec, label, track_idx )

        # Add buffer into track_obj_buf
        # Check the vector is too different
        self.track_obj_buf[label][track_idx].append( 
            self.track_obj[label][track_idx] )

        # Get All Buffer and calculate mounts
        buffers = self.track_obj_buf[label][track_idx]
        
        # If buffer not enough
        if len(buffers) <= int(self.get_param(K_BUFFER)): 
            return False

        # Get all vector and clear
        vectors = [ np.array(buffers[i+1]) - np.array(buffers[0]) for i in range(len(buffers)-1) ]
        
        # Clear data
        pos_vectors_1 = [ vec[0] for vec in vectors if vec[0] >= 0 ]
        neg_vectors_1 = [ vec[0] for vec in vectors if vec[0] < 0 ]
        pos_vectors_2 = [ vec[1] for vec in vectors if vec[1] >= 0 ]
        neg_vectors_2 = [ vec[1] for vec in vectors if vec[1] < 0 ]
        vectors_1 = pos_vectors_1 if len(pos_vectors_1)>len(neg_vectors_1) else neg_vectors_1
        vectors_2 = pos_vectors_2 if len(pos_vectors_2)>len(neg_vectors_2) else neg_vectors_2
        new_vectors = np.array([ [v1, v2] for v1, v2 in zip(vectors_1, vectors_2) ])

        # Average Vector
        avg_vector = sum(new_vectors)/len(new_vectors)
        
        # Get new Head and Tail
        tail_pt, head_pt = np.array(buffers[-1]-(avg_vector/2)), np.array(buffers[-1] + (avg_vector/2))
    
        # Get Arrow Pixel and Re-scale
        distance = max(self.get_coord_distance( head_pt, tail_pt ), 0.1)
        bias = (tail_pt - head_pt)*self.get_param(K_ARROW_LEN)/distance
        head_pt = (head_pt - bias).astype(int)
        tail_pt = (tail_pt + bias).astype(int)
        self.track_obj_vec[label][track_idx] = (head_pt, tail_pt)
        
        # Clear Old directions to keep the information is newest
        # self.track_obj_buf[label][track_idx].clear()
        self.track_obj_buf[label][track_idx].pop(0)
        return True

    # Re-define the method for vector detection
    def track_new_object_and_draw(self, frame, draw=True):
        """
        Track new object after `track_prev_object` function

        1. The remain point in `self.cur_pts` can identify to the new object
        2. Calculate the vector.
            a. Update direction in `self.track_obj_vec`
        """
        
        # traval each label
        for label in self.detected_labels:

            # update track object
            for pt in self.cur_pts[label]:
                self.track_obj[label][ self.track_idx[label] ] = pt
                self.track_idx[label] +=1
            
            # draw the track number on object
            if (not draw) or (frame is None): continue
            for track_idx, cur_pt in self.track_obj[label].items():

                # Get Vector ( Arrow )
                if not self.update_vector( label, track_idx ): 
                    continue
                
                vector = self.track_obj_vec[label].get(track_idx)
                
                # Draw Angle
                cv2.putText( 
                    frame, str(self.get_angle_for_cv(vector[1], vector[0])), 
                    (cur_pt[0]+10, cur_pt[1]+10), cv2.FONT_HERSHEY_SIMPLEX,
                    self.font_size, self.get_color(label), self.font_thick, cv2.LINE_AA )

                # Draw Arrow
                cv2.arrowedLine(
                    frame, vector[1], vector[0],
                    self.get_color(label), self.thick+1, tipLength = 0.5   )
                
        return frame

    def run(self, frame, data, draw=True) -> tuple:
        """
        Define the worflow when execute the application

        1. Tracking Object
        2. Record the vector between previous and current object
        3. Define the mount of the buffer to stablizing the final vector
        """
        if data is None:
            return frame, data
            
        # Init
        self.frame_idx += 1
        self.clear_current_point()
        self.update_draw_param( frame )

        # Capture all center point in current frame and draw the bounding box
        self.update_point_and_draw( data )
        
        # if not first frame: start to calculate distance to check the object is the same one
        self.track_prev_object()

        # adding the remaining point to track_obj
        self.track_new_object_and_draw( frame )

        # get tracking information which will be stored in `self.app_info`
        info = self.draw_app_info( frame )

        return ( frame, info)

if __name__ == "__main__":

    import cv2, time
    from ivit_i.common.model import get_ivit_model

    FONT, FONT_SCALE, FONT_THICKNESS = cv2.FONT_HERSHEY_SIMPLEX, 1, 1

    # Define iVIT Model
    model_type = 'obj'
    model_conf = { 
        "tag": model_type,
        "openvino": {
            "model_path": "./model/yolo-v3-tf/FP32/yolo-v3-tf.xml",
            "label_path": "./model/yolo-v3-tf/coco.names",
            "anchors": [ 10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326 ],
            "architecture_type": "yolo",
            "device": "CPU",
            "thres": 0.96
        }
    }

    app_conf = {
        'depend_on': [ 'car' ],
        'draw_bbox': True,
        'draw_info': False,
        "buffer": 40
    }

    ivit = get_ivit_model(model_type)
    ivit.set_async_mode()
    ivit.load_model(model_conf)
    
    # Def Application
    app = MovingDirection(params=app_conf, label=model_conf['openvino']['label_path'])

    # Get Source
    data, fps, fps_pool = None, -1, []

    cap = cv2.VideoCapture('./data/car.mp4')
    while(cap.isOpened()):

        t_start = time.time()

        ret, frame = cap.read()
        draw = frame.copy()
        if not ret: break

        _data = ivit.inference(frame=frame)
        data = _data if _data else data

        draw, info = app(draw, data)
        
        cv2.putText( draw, f"FPS: {fps}", (draw.shape[1]-200, 40), cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE, (0,0,255), FONT_THICKNESS, FONT )
        
        cv2.imshow('Tracking Sample', draw)

        press_key = cv2.waitKey(1) 
        if press_key in [ ord('q'), 27 ]: 
            break
        elif press_key == ord('b'): 
            app.set_param('draw_bbox', not app.get_param('draw_bbox')) 
        elif press_key == ord('i'):
            app.set_param('draw_info', not app.get_param('draw_info'))
        elif press_key in [ord('+'), ord('=')]:
            app.set_param('arrow_length', app.get_param('arrow_length')+1 )
        elif press_key in [ord('-'), ord('_')]:
            app.set_param('arrow_length', app.get_param('arrow_length')-1 )
        else:
            continue

        # Calculate FPS
        if _data:
            fps_pool.append( int(1/(time.time()-t_start)) )
            if len(fps_pool)>10:
                fps = sum(fps_pool)//len(fps_pool)
                fps_pool.clear()

    cap.release()
    ivit.release()

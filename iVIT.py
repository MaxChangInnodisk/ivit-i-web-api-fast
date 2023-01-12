# import sys, os, cv2
# import numpy as np
# sys.path.append( os.getcwd() )
# from ivit_i.app.common import ivitApp

# class DisplayImage(ivitApp):

#     def __init__(self, config=None, label=None, palette=None, log=True):
#         super().__init__(label, palette, log)
#         self.set_type('obj')
#         self.opacity = 0.3
#         self.border = 1
#         self.image = None
#         self.im_hei, self.im_wid = 0, 0

#     def init_params(self):
#         super().init_params()

#     def submit_image(self, image):
#         self.image = image
#         self.im_hei, self.im_wid = self.image.shape[:2] # h,w -> y, x
#         print("Get Image: {}x{}".format(self.im_hei, self.im_wid))

#     def run(self, frame, data, draw=True) -> tuple:
#         info = []
#         for det in (data['detections']):

#             # Parse output            
#             label       = det['label']
#             content     = '{} {:.1%}'.format(label, det['score'])
#             ( xmin, ymin, xmax, ymax ) \
#                  = [ det[key] for key in [ 'xmin', 'ymin', 'xmax', 'ymax' ] ] 
            
#             # Update Infor
#             info.append(content)

#             # Draw Top N label
#             if not draw: continue
            
#             # Draw bounding box
#             cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), self.palette[label] , self.border)

#             # Prepare Color Background
#             ( text_width, text_height), text_base \
#                 = cv2.getTextSize(content, self.FONT, self.FONT_SCALE, self.FONT_THICKNESS)

#             t_xmin, t_ymin, t_xmax, t_ymax = xmin, ymin, xmin+text_width, ymin+text_height+text_base 
#             text_area = frame[t_ymin:t_ymax, t_xmin:t_xmax]
            
#             color_img = np.zeros(text_area.shape, dtype=np.uint8)
#             for c in range(3):
#                 color_img[:,:,c] = np.ones(text_area.shape[:2], dtype=np.uint8) * self.palette[label][c]

#             frame[t_ymin:t_ymax, t_xmin:t_xmax] \
#                 = cv2.addWeighted(text_area, 1-self.opacity, color_img, self.opacity, 1.0)
            
#             # Draw Text
#             cv2.putText(
#                 frame, content, (t_xmin, t_ymax), self.FONT,
#                 self.FONT_SCALE, self.palette[label], self.FONT_THICKNESS, self.FONT_THICK
#             )

#             if self.image is None:
#                 continue
            
#         cv2.putText(
#             frame, 'hihihihihihihihih', (10, 50), self.FONT,
#             self.FONT_SCALE, (0,0,255), self.FONT_THICKNESS, self.FONT_THICK
#         )

#         return ( frame, info)

# if __name__ == "__main__":

#     import cv2
#     from ivit_i.common.model import get_ivit_model

#     # Define iVIT Model
#     model_type = 'obj'
#     model_anchor = [ 10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326 ]
#     model_conf = { 
#         "tag": model_type,
#         "openvino": {
#             "model_path": "./model/yolo-v3-tf/FP32/yolo-v3-tf.xml",
#             "label_path": "./model/yolo-v3-tf/coco.names",
#             "anchors": model_anchor,
#             "architecture_type": "yolo",
#             "device": "CPU",
#             "thres": 0.9
#         }
#     }

#     ivit = get_ivit_model(model_type)
#     ivit.load_model(model_conf)
    
#     # Def Application
#     app = DisplayImage(label=model_conf['openvino']['label_path'])

#     # Submit Image
#     im_path = os.path.join( os.path.dirname(os.path.realpath(__file__)),'logo.png')
#     im = cv2.imread(im_path)
#     im.resize([50, 100, 3])
#     if im is None:
#         raise FileNotFoundError('Could not find {}'.format(im_path))
#     app.submit_image(image=im)

#     # Get Source
#     cap = cv2.VideoCapture('/dev/video0')
#     while(cap.isOpened()):

#         ret, frame = cap.read()
#         if not ret: break
#         output = ivit.inference(frame=frame)

#         frame, info = app(frame, output)
#         cv2.imshow('Test', frame)
#         if cv2.waitKey(1) in [ ord('q'), 27 ]: break

#     cap.release()
#     ivit.release()

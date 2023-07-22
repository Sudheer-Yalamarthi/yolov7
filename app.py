import torch
import gradio as gr
import cv2
import numpy as np
import random
import numpy as np
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, \
    scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import time_synchronized
import time
from ultralytics import YOLO
from track import MOT

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)

names = ["animal",
"autorickshaw",
"bicycle",
"bus",
"car",
"motorcycle",
"person",
"rider",
"traffic light",
"traffic sign",
"truck"
]


#colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
colors  = {
    "animal": [246,198, 145],
    "autorickshaw": [255,204, 54],
    "bicycle": [119,11, 32],
    "bus": [ 0,60,100],
    "car": [ 0,0,142],
    "motorcycle": [ 0,0,230],
    "person": [220,20, 60],
    "rider": [255,0, 0],
    "traffic light": [250,170, 30],
    "traffic sign": [220,220, 0],
    "truck": [ 0,0, 70]
}


def detectv7(img,model,device,iou_threshold=0.45,confidence_threshold=0.25):   
    imgsz = 640
    img = np.array(img)
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    imgs = img.copy()  # for NMS
    
    image, ratio, dwdh = letterbox(img, auto=False)
    image = image.transpose((2, 0, 1))
    img = torch.from_numpy(image).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)


    # Inference
    t1 = time_synchronized()
    start = time.time()
    with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
        pred = model(img,augment=True)[0]
    fps_inference = 1/(time.time()-start)
    t2 = time_synchronized()

    # Apply NMS
    pred = non_max_suppression(pred, confidence_threshold, iou_threshold, classes=None, agnostic=True)
    t3 = time_synchronized()

    for i, det in enumerate(pred):  # detections per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], imgs.shape).round()


            # Write results
            for *xyxy, conf, cls in reversed(det):
                label = f'{names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, imgs, label=label, color=colors[names[int(cls)]], line_thickness=1)

    return imgs,fps_inference

def detectv8(img,model,device,iou_threshold=0.45,confidence_threshold=0.25):   
    img = np.array(img)
    # Inference
    t1 = time_synchronized()
    start = time.time()
    results= model.predict(img,conf=confidence_threshold, iou=iou_threshold)
    fps_inference = 1/(time.time()-start)
    
    if torch.cuda.is_available():
        boxes= results[0].boxes.cpu().numpy()
    else:
        boxes=results[0].boxes.numpy()
    for bbox in boxes:
        #print(f'{colors[names[int(bbox.cls[0])]]}')
        label = f'{names[int(bbox.cls[0])]} {bbox.conf[0]:.2f}'
        plot_one_box(bbox.xyxy[0],img,colors[names[int(bbox.cls[0])]],label, line_thickness=1)

    return img,fps_inference

def inference(img,model_link,iou_threshold,confidence_threshold):
    print(model_link)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Load model
    model_path = 'weights/'+str(model_link)+'.pt'
    if model_link== 'yolov8m':
        model = YOLO(model_path)
        return detectv8(img,model,device,iou_threshold,confidence_threshold)
    else:
        model = attempt_load(model_path, map_location=device) 
        return detectv7(img,model,device,iou_threshold,confidence_threshold)


def inference2(video,model_link,iou_threshold,confidence_threshold):
    print(model_link)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Load model
    model_path = 'weights/'+str(model_link)+'.pt'
    if model_link== 'yolov8m':
        model = YOLO(model_path)
    else:
        model = attempt_load(model_path, map_location=device) 
    frames = cv2.VideoCapture(video)
    fps = frames.get(cv2.CAP_PROP_FPS)
    image_size = (int(frames.get(cv2.CAP_PROP_FRAME_WIDTH)),int(frames.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    finalVideo = cv2.VideoWriter('output.mp4',cv2.VideoWriter_fourcc(*'VP90'), fps, image_size)
    fps_video = []
    while frames.isOpened():
        ret,frame = frames.read()
        if not ret:
            break
        if model_link== 'yolov8m':
            frame,fps = detectv8(frame,model,device,iou_threshold,confidence_threshold)
        else:
            frame,fps = detectv7(frame,model,device,iou_threshold,confidence_threshold)
        fps_video.append(fps)
        finalVideo.write(frame)
    frames.release()
    finalVideo.release()
    return 'output.mp4',np.mean(fps_video)

def inference_comp(image,iou_threshold,confidence_threshold):
    v8_out, v8_fps = inference(image, "yolov8m",iou_threshold,confidence_threshold)
    v7_out, v7_fps = inference(image, "yolov7",iou_threshold,confidence_threshold)
    return v7_out,v8_out,v7_fps,v8_fps

def MODT(sourceVideo, trackingmethod):    
    #model_path = 'weights/'+str(model_link)+'.pt'
    model_path = 'weights/yolov8m.pt'
    return MOT(model_path, trackingmethod, sourceVideo), 30

examples_images = ['data/images/1.jpg',
            'data/images/2.jpg',
            'data/images/bus.jpg',
            'data/images/3.jpg']
examples_videos = ['data/video/1.mp4','data/video/2.mp4'] 

models = ['yolov8m','yolov7','yolov7t']
trackers = ['strongsort', 'bytetrack', 'ocsort']

with gr.Blocks() as demo:
    gr.Markdown("## IDD Inference on Yolo V7 and V8 ")
    with gr.Tab("Image"):
        gr.Markdown("## Yolo V7 and V8 Inference on Image")
        with gr.Row():
            image_input = gr.Image(type='pil', label="Input Image", source="upload")
            image_output = gr.Image(type='pil', label="Output Image", source="upload")
        fps_image = gr.Number(0,label='FPS')
        image_drop = gr.Dropdown(choices=models,value=models[0])
        image_iou_threshold = gr.Slider(label="IOU Threshold",interactive=True, minimum=0.0, maximum=1.0, value=0.5)
        image_conf_threshold = gr.Slider(label="Confidence Threshold",interactive=True, minimum=0.0, maximum=1.0, value=0.6)
        gr.Examples(examples=examples_images,inputs=image_input,outputs=image_output)
        text_button = gr.Button("Detect")
    with gr.Tab("Video"):
        gr.Markdown("## Yolo V7 and V8 Inference on Video")
        with gr.Row():
            video_input = gr.Video(type='pil', label="Input Video", source="upload")
            video_output = gr.Video(type="pil", label="Output Video",format="mp4")
        fps_video = gr.Number(0,label='FPS')
        video_drop = gr.Dropdown(label="Model", choices=models,value=models[0])
        video_iou_threshold = gr.Slider(label="IOU Threshold",interactive=True, minimum=0.0, maximum=1.0, value=0.5)
        video_conf_threshold = gr.Slider(label="Confidence Threshold",interactive=True, minimum=0.0, maximum=1.0, value=0.6)
        gr.Examples(examples=examples_videos,inputs=video_input,outputs=video_output)
        with gr.Row():
            video_button_detect = gr.Button("Detect")
    
    with gr.Tab("Compare Models"):
        gr.Markdown("## YOLOv7 vs YOLOv8 Object detection comparision")
        with gr.Row():
            image_comp_input = gr.Image(type='pil', label="Input Image", source="upload")
        with gr.Row():
            image_comp_iou_threshold = gr.Slider(label="IOU Threshold",interactive=True, minimum=0.0, maximum=1.0, value=0.5)
            image_comp_conf_threshold = gr.Slider(label="Confidence Threshold",interactive=True, minimum=0.0, maximum=1.0, value=0.6)
        text_comp_button = gr.Button("Detect")
        with gr.Row():
            image_comp_output_v7 = gr.Image(type='pil', label="YOLOv7 Output Image", source="upload")
            image_comp_output_v8 = gr.Image(type='pil', label="YOLOv8 Output Image", source="upload")
        with gr.Row():
            v7_fps_image = gr.Number(0,label='v7 FPS')        
            v8_fps_image = gr.Number(0,label='v8 FPS')
        gr.Examples(examples=examples_images,inputs=image_comp_input,outputs=[image_comp_output_v7,image_comp_output_v8])
    
    with gr.Tab("Video Tacking"):
        gr.Markdown("## MOT using YoloV8 detection with tracking")
        with gr.Row():
            videotr_input = gr.Video(type='pil', label="Input Video", source="upload")
            videotr_output = gr.Video(type="pil", label="Output Video",format="mp4")
        fpstr_video = gr.Number(0,label='FPS')
        tracking_drop = gr.Dropdown(choices=trackers,value=trackers[0], label="Select the tracking method")
        videotr_iou_threshold = gr.Slider(label="IOU Threshold",interactive=True, minimum=0.0, maximum=1.0, value=0.5)
        videotr_conf_threshold = gr.Slider(label="Confidence Threshold",interactive=True, minimum=0.0, maximum=1.0, value=0.6)
        gr.Examples(examples=examples_videos,inputs=videotr_input,outputs=videotr_output)
        video_button_track = gr.Button("Track")
        

    text_button.click(inference, inputs=[image_input,image_drop,
                                         image_iou_threshold,image_conf_threshold],
                                        outputs=[image_output,fps_image])
    video_button_detect.click(inference2, inputs=[video_input,video_drop,
                                           video_iou_threshold,video_conf_threshold],            
                                        outputs=[video_output,fps_video])
    text_comp_button.click(inference_comp,inputs=[image_comp_input,
                                            image_comp_iou_threshold,
                                            image_comp_conf_threshold],
                                            outputs=[image_comp_output_v7,image_comp_output_v8,v7_fps_image,v8_fps_image])
    video_button_track.click(MODT,inputs=[videotr_input, tracking_drop],          
                             outputs=[videotr_output, fpstr_video])

demo.launch(debug=True,enable_queue=True)
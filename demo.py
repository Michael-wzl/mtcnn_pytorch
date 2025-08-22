import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
from typing import Optional
from datetime import datetime

import cv2
import torch
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
import tqdm

from mtcnn import MTCNN

def detect_vid(scr_vid_path: str, dst_vid_path: str, device: Optional[torch.device] = None) -> None:
    """
    Detect faces in a video and save the output video with bounding boxes.

    Args:
        scr_vid_path: Path to the input video file.
        dst_vid_path: Path to the output video file.
        device: Device to run the model on
    """
    detector = MTCNN(device=device)
    scr = cv2.VideoCapture(scr_vid_path)
    f_rate = scr.get(cv2.CAP_PROP_FPS)
    n_frame = int(scr.get(cv2.CAP_PROP_FRAME_COUNT))
    H, W = int(scr.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(scr.get(cv2.CAP_PROP_FRAME_WIDTH))
    dst = cv2.VideoWriter(dst_vid_path, cv2.VideoWriter_fourcc(*'mp4v'), f_rate, (H, W))
    pbar = tqdm.tqdm(total=n_frame, desc="Processing Video Frames")
    time_cnt = 0
    for frame_idx in range(int(scr.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = scr.read()
        if not ret:
            break
        start_time = datetime.now()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float().to(device)
        boxes, landmarks = detector(img_tensor)
        time_cnt += (datetime.now() - start_time).total_seconds()
        if boxes is not None and len(boxes) > 0:
            for box in boxes.cpu().numpy():
                x1, y1, x2, y2, score = box[:5]
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"{score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            for landmark in landmarks.cpu().numpy():
                if len(landmark) >= 10:  # Make sure we have at least 10 values
                    for i in range(5):  # 5 landmark points
                        x, y = int(landmark[i]), int(landmark[i+5])
                        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        dst.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        pbar.update(1)
    pbar.close()
    print(f"Processed {n_frame} frames in {time_cnt:.2f} seconds. Average FPS: {n_frame / time_cnt:.2f}" if time_cnt > 0 else "N/A")
    scr.release()
    dst.release()

def detect_imgs(scr_img_path: str, dst_img_path: str, device: Optional[torch.device] = None):
    """
    Detect faces in an image and save the output image with bounding boxes.

    Args:
        scr_img_path: Path to the input image file.
        dst_img_path: Path to the output image file.
        device: Device to run the model on
    """
    detector = MTCNN(device=device)
    img = cv2.imread(scr_img_path)
    start = datetime.now()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device)
    boxes, landmarks = detector(img_tensor)
    print(f"Detection time: {datetime.now() - start}")
    if boxes is not None and len(boxes) > 0:
        for box in boxes.cpu().numpy():
            x1, y1, x2, y2, score = box[:5]
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img, f"{score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        for landmark in landmarks.cpu().numpy():
            if len(landmark) >= 10:  # Make sure we have at least 10 values
                for i in range(5):  # 5 landmark points
                    x, y = int(landmark[i]), int(landmark[i+5])
                    cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
        cv2.imwrite(dst_img_path, img)

if __name__ == "__main__":
    mode = "vid" # vid / imgs
    device = torch.device("mps")
    if mode == 'vid':
        scr_vid_path = "assets/bc.mp4"
        dst_vid_path = "results/vids/bc_detected.mp4"
        detect_vid(scr_vid_path, dst_vid_path, device=device)
    elif mode == 'imgs':
        scr_imgs_folder = "assets/bbt.png"
        dst_imgs_folder = "results/imgs/bb_detected.png"
        detect_imgs(scr_imgs_folder, dst_imgs_folder, device=device)

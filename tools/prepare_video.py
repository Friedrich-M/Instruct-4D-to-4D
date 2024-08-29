import os
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    """
    --data_dir/
    -----video_1.mp4
    -----video_2.mp4
    -----frames/
    ---------video_1/
    ---------video_2/
    """
    parser.add_argument('data_dir', type=str, help='folder for processing')
    args = parser.parse_args()
    all_cams = sorted([i.replace('.mp4', '') for i in os.listdir(f'{args.data_dir}/') if 'mp4' in i])
    
    video_dirs = [os.path.join(args.data_dir, i) for i in all_cams]
    frames_dir = os.path.join(args.data_dir, 'frames')
    os.makedirs(frames_dir, exist_ok=True)
    
    for idx, cam in enumerate(all_cams): 
        video_path = os.path.join(args.data_dir, cam + '.mp4')
        cam_dir = os.path.join(frames_dir, cam)
        os.makedirs(cam_dir, exist_ok=True)
        
        cmd = 'ffmpeg -i {} {}'.format(video_path, os.path.join(cam_dir, "%06d.jpg"))
        os.system(cmd)
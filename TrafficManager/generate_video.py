import os
import cv2
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def resize_image(image, target_height):
    aspect_ratio = image.width / image.height
    new_width = int(target_height * aspect_ratio)
    return image.resize((new_width, target_height), Image.LANCZOS)

def create_frame(paths, target_height=500):
    images = [Image.open(path) for path in paths]
    resized_images = [resize_image(img, target_height) for img in images]
    new_width = sum(img.width for img in resized_images)
    new_image = Image.new('RGB', (new_width, target_height))
    x_offset = 0
    for img in resized_images:
        new_image.paste(img, (x_offset, 0))
        x_offset += img.width
    return np.array(new_image)

def process_images(paths):
    return create_frame(paths)

def get_sorted_files(input_dir, prefix, extension):
    """普通文件（diffusion_, agent_）"""
    return sorted([
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.startswith(prefix) and f.endswith(extension)
    ])

def get_bev_files(input_dir):
    flat_frames = get_sorted_files(input_dir, 'bev_', '.png')
    if flat_frames:
        return flat_frames

    bev_dirs = sorted([
        os.path.join(input_dir, d)
        for d in os.listdir(input_dir)
        if d.startswith('bev_') and os.path.isdir(os.path.join(input_dir, d))
    ])
    frames = []
    for bev_dir in bev_dirs:
        imgs = sorted([
            os.path.join(bev_dir, f)
            for f in os.listdir(bev_dir)
            if f.endswith('.png')
        ])
        if imgs:
            frames.append(imgs[-1])   # ← 只取最后一张
    return frames


def main(args):
    input_dir = os.path.join(args.output_dir, 'imgs')

    bev_files       = get_bev_files(input_dir)
    diffusion_files = get_sorted_files(input_dir, 'diffusion_', '.jpg')
    agent_files     = get_sorted_files(input_dir, 'agent_', '.jpg')

    print(f"BEV 帧数:       {len(bev_files)}")
    print(f"Diffusion 帧数: {len(diffusion_files)}")
    print(f"Agent 帧数:     {len(agent_files)}")

    if not (len(bev_files) == len(diffusion_files) == len(agent_files)):
        print("Error: Number of files in each category doesn't match.")
        print(f"  bev={len(bev_files)}, diffusion={len(diffusion_files)}, agent={len(agent_files)}")
        return

    all_frame_paths = list(zip(bev_files, diffusion_files, agent_files))
    collision_bev = os.path.join(input_dir, 'collision_bev.png')
    collision_diffusion = os.path.join(input_dir, 'collision_diffusion.jpg')
    if (
        os.path.isfile(collision_bev)
        and os.path.isfile(collision_diffusion)
        and agent_files
    ):
        all_frame_paths.append(
            (collision_bev, collision_diffusion, agent_files[-1])
        )
        print("Collision frame appended with the last available Agent image.")

    first_frame = create_frame(list(all_frame_paths[0]))
    height, width, _ = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        os.path.join(args.output_dir, args.output_video),
        fourcc, args.fps, (width, height)
    )

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_images, list(paths)) for paths in all_frame_paths]
        for future in tqdm(futures, total=len(futures), desc="Processing images"):
            frame = future.result()
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    out.release()

    input_video_path = os.path.join(args.output_dir, args.output_video)
    temp_compressed_path = os.path.join(args.output_dir, 'temp_compressed.mp4')
    os.system(f'ffmpeg -i {input_video_path} -vcodec libx265 -crf 28 {temp_compressed_path}')
    os.remove(input_video_path)
    os.rename(temp_compressed_path, input_video_path)
    print(f"Video saved as {args.output_video}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a video from images.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_video", type=str, default="output_video.mp4")
    parser.add_argument("--fps", type=int, default=2)
    args = parser.parse_args()
    main(args)

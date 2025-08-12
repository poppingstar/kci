import cv2 as cv
import numpy as np
from pathlib import Path
import concurrent.futures as futures
from typing import Iterable, Generator
from torchmetrics.image import StructuralSimilarityIndexMeasure
from itertools import repeat
import torch


def aggregate(func):
    def wrapper(paths:Iterable[Path]) -> Generator[Path]:
        for path in paths:
            frames = func(path)
            yield from frames

    return wrapper


def get_leaf_files(dir_path:Path|str) -> Generator[Path]:
    dir_path = Path(dir_path)

    if not dir_path.is_dir():
        yield dir_path
    else:
        for child in dir_path.iterdir():
            if child.is_dir():
                yield from get_leaf_files(child)
            else:
                yield child


def read_frame_at(cap:cv.VideoCapture, frame_idx:int) -> np.ndarray:
    cap.set(cv.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()

    if ret:
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2,0,1).unsqueeze(0).float() / 255.0
        return tensor
    else:
        return None
    


def get_parents_by_files(files:Iterable[Path], *suffixes:str)->dict[Path]:
    suffixes = set(suffixes)
    matches_suffix = (lambda x: x.suffix in suffixes) if suffixes else (lambda x: True)
    
    grouped_files = {}
    for f in files:
        if not matches_suffix(f):
            continue
        parent = f.parent
        name = f.name
        if parent not in grouped_files:
            grouped_files[parent] = [name]
        else:
            grouped_files[parent].append(name)
    
    return grouped_files


def get_frames_at_interval(video_path:str|Path, interval_sec:int) -> list[np.ndarray]:
    get_total_frames = lambda cap: cap.get(cv.CAP_PROP_FRAME_COUNT)
    get_fps = lambda cap: cap.get(cv.CAP_PROP_FPS)
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        return

    fps = get_fps(cap)
    total_frame = int(get_total_frames(cap))
    step = int(fps*interval_sec)

    try:
        frames = []
        past_frame_idx = 0
        for frame_idx in range(1, total_frame, step):
            target_frame = np.random.randint(past_frame_idx, frame_idx)
            frame = read_frame_at(cap, target_frame)
            if frame is None:
                break
            frames.append(frame)
            past_frame_idx = frame_idx
    finally:
        cap.release()
    return frames


def filter_similar_imgs(imgs: list[np.ndarray], threshold: float = 0.9) -> list[np.ndarray]:
    """
    imgs: BGR uint8 이미지를 담은 리스트
    returns: 유사도(threshold) 미만인 프레임만 남긴 리스트
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    # 2) numpy→Tensor, [N,C,H,W], float, [0,1] 정규화
    tensors = []
    for img in imgs:
        rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2,0,1).unsqueeze(0).float() / 255.0
        tensors.append(t)
    batch = torch.cat(tensors, dim=0).to(device)  # shape: [N,3,H,W]

    kept_indices = []
    refs = []  # 기준 프레임 텐서들
    for i in range(batch.size(0)):
        current = batch[i:i+1]  # [1,3,H,W]
        if not refs:
            refs.append(current); kept_indices.append(i)
        else:
            ref_batch = torch.cat(refs, dim=0)  # [M,3,H,W]
            # 3) batch별 SSIM 계산: ref 마다 current와 비교
            #    output shape: [M] (자동으로 reduction='none')
            similarity = ssim_metric(ref_batch, current.expand_as(ref_batch))  
            # all scores < threshold 이면 보관
            if torch.all(similarity < threshold):
                refs.append(current)
                kept_indices.append(i)

    # 4) 인덱스 기반으로 원본 BGR 이미지 반환
    return [imgs[i] for i in kept_indices]


def save_imgs(imgs:np.ndarray, output_dir:Path) -> None:
    output_dir = Path(output_dir)
    for i, img in enumerate(imgs):
        cv.imwrite(output_dir/f'{i}.png', img)


def process_videos(paretnt:Path, children:Iterable[Path], output_dir:Path, interval_sec:int):
    output_dir.mkdir(exist_ok=True, parents=True)
    
    total_frames = []
    for child in children:
        frame = get_frames_at_interval(paretnt/child, interval_sec)
        total_frames.extend(frame)
    
    non_similar_frames = filter_similar_imgs(total_frames, 0.9)

    output_dir_sub = output_dir/paretnt.name
    save_imgs(non_similar_frames, output_dir_sub)


if __name__ == '__main__':
    def main():
        root_dir = Path(fr"E:\Datasets\딥페이크 변조 영상\1.Train\dffs")
        output_dir = repeat(Path(fr"E:\Datasets\outputs"))
        interval = repeat(1000)
        leafs = get_leaf_files(root_dir)
        grouped_files = get_parents_by_files(leafs, '.mp4')


        parents = [k for k in grouped_files.keys()]
        children = [v for v in grouped_files.values()]
        with futures.ProcessPoolExecutor(max_workers=3) as executor:
            executor.map(process_videos, parents, children, output_dir, interval)


    main()
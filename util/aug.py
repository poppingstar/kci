import torchvision.transforms as transforms
import torch, torchvision, threading, random
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from os import cpu_count
from pathlib import Path
# from utils import get_leaf_dirs

def get_leaf_dirs(p: Path)->list[Path]:
    p = Path(p)
    assert p.is_dir(), 'Input path is not dir'
    
    if not p.is_dir():
        return []
    
    subs = list(p.iterdir())
    if not subs:
        return [p]
    
    leaf_dirs = []
    for sub in subs:
        if sub.is_dir():
            leaf_dirs.extend(get_leaf_dirs(sub))
    
    if not leaf_dirs:
        return [p]
    
    return leaf_dirs

def get_imgs(files)->list[Image.Image]:
    imgs = []
    for f in files:
        with Image.open(f) as img:
            imgs.append(img.copy())
    return imgs

def aug_img(img:Image.Image, 
            augmentation_transforms:transforms.Compose=None)->Image.Image:
    assert isinstance(img, Image.Image), 'Input image type is must be PIL.Image.Imge'

    if not augmentation_transforms:
        augmentation_transforms = transforms.Compose([
            transforms.ToTensor(), transforms.RandomHorizontalFlip(), transforms.RandomRotation(5),
            transforms.ColorJitter(0.2, hue=0.05), transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))])
    to_pil = transforms.ToPILImage()
        
    augmented_img = augmentation_transforms(img)
    augmented_img = to_pil(augmented_img)
    return augmented_img

def list_split(data, parts, shuffle)->list[list]:
    chunk_size = len(data) // parts
    reminder = len(data) % parts

    if shuffle:
        data = random.shuffle(data)
    
    result = []
    start = 0
    for i in range(1, parts+1):
        extra = 1 if i <= reminder else 0
        end = start + chunk_size + extra
        splited_list = data[start:end]
        result.append(splited_list)
        start = end
    return result

def aug_dir(target_dir, after_dir, factor:int):
    target_dir = Path(target_dir)
    after_dir = Path(after_dir)
    after_dir.mkdir(exist_ok=True, parents=True)
    leafs = get_leaf_dirs(target_dir)
    thread_num = cpu_count()

    def img_processing(files, after_dir, start):
        imgs = get_imgs(files)
        temp = []
        file_name = start + 0
        for img in imgs:
            for _ in range(factor):
                clone_img = img.copy()
                augmented_img = aug_img(clone_img)
                augmented_img.save(after_dir/f'{file_name}.png', format='PNG')
                temp.append(augmented_img)
                file_name+=1

    for d in leafs:
        files = list(d.iterdir())
        files_per_thread = list_split(files, thread_num)
        start = 0
        threads = []
        dst = after_dir/d.name
        dst.mkdir(exist_ok=True,parents=True)
        for imgs in files_per_thread:
            thread = threading.Thread(target = img_processing, args=(imgs, dst, start))
            thread.start()
            threads.append(thread)
            start += len(imgs)*factor
        
        for t in threads:
            t.join()


if __name__ == '__main__':
    # aug_dir(r"E:\Datasets\deepfake\before", r"E:\Datasets\deepfake\after_aug", 3)
    print('Done') 
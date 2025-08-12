import PIL.Image as Image
import PIL.ImageFile as ImageFile
from pathlib import Path
import numpy as np
import threading, os, piexif, random

def is_gray_scale(img_path:Path):
    with Image.open(img_path) as img:
        return img.mode == 'L'

def separate_non_rgb(directory_list:list, separation_dir:Path):
    for directory in directory_list:
        directory = Path(directory)
        for file in directory.iterdir():
            if is_gray_scale(file):
                separation_dir.mkdir(exist_ok=True)
                name = file.stem
                extension = file.suffix
                file_path = separation_dir/directory/file.name

                i=1
                while file_path.exists():
                    file_path = separation_dir/directory/f'{name}({i}){extension}'
                    i+=1
                file.rename(file_path)

def chk_corrupt(separtion_dir:Path, dirlist):
    for sub_dir in dirlist:
        sub_dir = Path(sub_dir)
        for file in sub_dir.iterdir():
            try:
                with Image.open(file) as img:
                    img.verify()
                with Image.open(file) as img:
                    img.load()
                    exif_data = img.info.get('exif')
                    if exif_data:
                        piexif.load(exif_data)
            except:
                sub_sepration_dir = separtion_dir/sub_dir.name
                sub_sepration_dir.mkdir(exist_ok=True, parents=True)
                new_path = sub_sepration_dir/file.name
                file.rename(new_path)

def seperate_corrupt_non_rgb(root):
    ImageFile.LOAD_TRUNCATED_IMAGES=False
    cpu_count = os.cpu_count()
    root = Path(root)
    corrupt_dir = root/'corrupt_img'
    non_rgb_dir = root/'non_rgb'

    sub_dirs = list(root.iterdir())
    thread_num = min(len(sub_dirs), cpu_count)
    chunk = len(sub_dirs) // thread_num
    residual = len(sub_dirs) % thread_num

    threads = []
    start=0
    for i in range(thread_num):
        end=start+chunk
        if residual>i:
            end+=1
        dir_per_thread = sub_dirs[start:end]
        start=end

        thread = threading.Thread(target=separate_non_rgb, args=(dir_per_thread, non_rgb_dir))
        thread = threading.Thread(target=chk_corrupt, args=(corrupt_dir, dir_per_thread))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

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

def split_dir(dataset_path:Path|str, **split_dict:float) -> None:
    if not np.isclose(sum(split_dict.values()), 1.0, atol=1e-9):
        raise ValueError("The sum of the split ratios must be approximately 1.0")

    dataset_path = Path(dataset_path)
    parent_dir = dataset_path.parent
    class_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]

    for class_dir in class_dirs:
        files = list(class_dir.iterdir())
        random.shuffle(files)
        num_files = len(files)
        split_sizes = [int(num_files * rate) for rate in split_dict.values()]
        split_sizes[-1] += num_files - sum(split_sizes)  

        for dst, split_size in zip(split_dict.keys(), split_sizes):
            sliced_list = files[:split_size]
            del files[:split_size]

            dst_dir = parent_dir/dst/class_dir.name
            dst_dir.mkdir(exist_ok=True, parents=True)
            for f in sliced_list:
                f.rename(dst_dir/f.name)

def get_leaf_dirs(p:Path):
    assert isinstance(p, Path), '인수는 Path의 인스턴스여야 합니다!'
    
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

def split_integer0(root_dir, **split_dict)->list[Path]:
    root_dir = Path(root_dir)
    leaf_dirs = get_leaf_dirs(root_dir)
    
    for leaf in leaf_dirs:
        all_files = list(leaf.iterdir())

        is_split_size_exceeding = sum(split_dict.values()) > len(all_files)
        if is_split_size_exceeding:
            raise ValueError('The sum of split_size exceeds the total number of files')\
        
        random.shuffle(all_files)
        for target_dir_name, chunk_size in split_dict.items():
            parent = root_dir/target_dir_name/leaf.name
            parent.mkdir(exist_ok=True, parents=True)

            files_to_move = all_files[:chunk_size]
            for file in files_to_move:
                file.rename(parent/file.name)
            del all_files[:chunk_size]

def is_color_gray(img: np.ndarray, tolerance=5, brightness_range=(30, 220)):
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    
    if img.ndim == 2:
        return True

    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("Expected RGB image with shape (H, W, 3)")

    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    r, g, b = r.astype(np.int16), g.astype(np.int16), b.astype(np.int16)

    diff_rg = np.abs(r - g)
    diff_gb = np.abs(g - b)
    diff_br = np.abs(b - r)

    avg = (r + g + b) / 3
    in_gray_range = (diff_rg <= tolerance) & (diff_gb <= tolerance) & (diff_br <= tolerance)
    in_brightness = (avg >= brightness_range[0]) & (avg <= brightness_range[1])

    gray_mask = in_gray_range & in_brightness
    gray_ratio = np.sum(gray_mask) / (img.shape[0] * img.shape[1])

    return gray_ratio >= 0.7


def is_color_gray(img:np.ndarray, tolerance=5):
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    is_one_channel = img.ndim == 2
    if is_one_channel:
        return True
    
    gray = 128
    r,g,b = img[:,:,0], img[:,:,1], img[:,:,2]

    total = r/3 + g/3 + b/3
    c = total[(total<=gray+tolerance) & (total>=gray-tolerance)]
    c = sum(c)

    img_size = img.shape[0] * img.shape[1]
    is_gray = c >= (img_size*0.70)
    return True if is_gray else False

def temp():
    root_dir = input('디렉토리 경로를 입력하세요: ')
    root_dir = root_dir.strip('"')
    root_dir = Path(root_dir)
    leaf_dirs = get_leaf_dirs(root_dir)

    gray_dir = root_dir/'gray'
    gray_dir.mkdir(exist_ok=True)

    for leaf_dir in leaf_dirs:
        gray_leaf_dir = gray_dir/leaf_dir.name
        gray_leaf_dir.mkdir(exist_ok=True)
        for file in leaf_dir.iterdir():
            with Image.open(file) as img:
                img = np.array(img)

            if is_color_gray(img):
                file.rename(gray_leaf_dir/file.name)

if __name__ == '__main__':
    temp()
    print('Done')
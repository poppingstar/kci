from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import PIL.Image as Image
import pandas as pd
from torchvision import transforms
import shutil, os, re


def labeling(dataset_path, label_file):
    dataset_path=Path(dataset_path)
    outputdir=dataset_path/'output'
    outputdir.mkdir(exist_ok=True)

    df=pd.read_csv(label_file)
    df=df[['file_name', 'label']]
    
    for row in df.itertuples():
        file=Path(row.file_name)
        original_path=dataset_path/file
        label_dir=outputdir/str(row.label)
        label_dir.mkdir(exist_ok=True)

        original_path.rename(label_dir/file.name)


def split(dataset_path:Path, val_rate=0.2, test_rate=0.1):
    train_dir=dataset_path/'train'
    for d in train_dir.iterdir():
        if not d.is_dir():
            continue

        flist=[f for f in d.iterdir()]
        num_files=len(flist)

        val_size=int(num_files*val_rate)
        test_size=int(num_files*test_rate)
        
        val_files=flist[:val_size]
        test_files=flist[val_size:val_size+test_size]

        val_dir=dataset_path/'valid'/d.name
        val_dir.mkdir(exist_ok=True, parents=True)
        for file in val_files:
            file.rename(val_dir/file.name)

        test_dir=dataset_path/'test'/d.name
        test_dir.mkdir(exist_ok=True, parents=True)
        for file in test_files:
            file.rename(test_dir/file.name)


def find_low_dirs(path:Path):
    while path.is_dir():
        path = path.iterdir()[0]
    low_dirs = path.parent.iterdir()
    return low_dirs


def get_imgs(paths:list[str])->list:
    max_threads_num = os.cpu_count()
    subset_size = len(paths)//max_threads_num

    if len(paths) <= max_threads_num:
        subsets = [paths]
    else: 
        subsets = [paths[(i)*subset_size:(i+1)*subset_size] for i in range(max_threads_num)]
        subsets[-1].extend(paths[max_threads_num*subset_size:])
        
    def load_image_subset(path_subset):
        subset_imgs = []
        for p in path_subset:
            with Image.open(p) as img:
                subset_imgs.append(img.copy())
        return subset_imgs

    
    with ThreadPoolExecutor(max_threads_num) as executor:
        futures = [executor.submit(load_image_subset, subset) for subset in subsets]
    
    imgs = []
    for future in futures:
        imgs.extend(future.result())
    return imgs


def save_imgs(imgs, output_dir:str):        
    max_threads_num = os.cpu_count() or 6
    subset_size = len(imgs)//max_threads_num

    if len(imgs) <= max_threads_num:
        subsets = [imgs]
    else: 
        subsets = [imgs[(i)*subset_size:(i+1)*subset_size] for i in range(max_threads_num)]
        subsets[-1].extend(imgs[max_threads_num*subset_size:])
        
    def save_image_subset(imgs:list, output_dir:str, start_name):
        for i, img in enumerate(imgs):
            img.save(output_dir/f'{start_name + i}.png')

    #디렉토리에 기존 파일 여부 확인
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
        start_num = 0
    else:
        exist_fies = output_dir.iterdir()
        exist_fies_num = sum(1 for file in exist_fies if file.is_file())
        start_num = exist_fies_num + 1

    #스레드 실행
    with ThreadPoolExecutor(max_threads_num) as executor:
        for subset in subsets:
            executor.submit(save_image_subset, subset, output_dir, start_num)
            start_num += len(subset)


def augment(times, imgs, transformer, output_dir):
    augemted_imgs = []
    for img in imgs:
        for _ in range(times):
            augemted_imgs.append(transformer(img))

    save_imgs(augemted_imgs, output_dir)

    # for augmentd_img in augemted_imgs:
    #     augmentd_img.save(output_dir/f'{file_name}.png')
    #     file_name += 1


def m2(dir_path:Path):
    sub_files = list(dir_path.iterdir())
    imgs = get_imgs(sub_files)
    transformer = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomRotation(5),
                    transforms.ColorJitter(0.1,0.1,0.1,0.1),
                    transforms.RandomAffine(degrees=0, translate=(0.1,0.1))
                    ])
    output_dir = dir_path.parent/f'{dir_path.name}_augmented'
    augment(3, imgs, transformer, output_dir)


def organize_files_by_regex(path:str|Path, expression:str|Path, file_only = True) -> None:
    """
    root_dir 내의 파일을 검색해,
    정규식 `expression`에 매칭된 문자열을 폴더명으로 사용해 이동시킵니다.

    Args:
        root_dir (str | Path): 탐색할 최상위 디렉토리 경로
        expression (str): 매칭할 정규식 패턴
    """
    d = Path(path)
    pattern = re.compile(expression)
    for child in d.iterdir():
        if file_only and child.is_dir():
            continue

        match_result = pattern.search(child.name)

        if not match_result:
            continue

        matched_string = match_result.group(0)
        group_dir = child.parent/matched_string
        group_dir.mkdir(exist_ok=True)

        dst = group_dir/child.name

        if child.resolve() == dst.resolve():
            continue

        shutil.move(child, dst)


if __name__ == '__main__':
    p = Path(r"E:\Datasets\deep_fake\valid")
    for sub_dir in p.iterdir():
        m2(sub_dir)


""" 
#멀티 스레딩 포함해서 재설계 ㄱㄱ
def main(dataset:Path, destination:Path, num_per_class:int):
    class_dirs = dataset.iterdir()

    for class_dir in class_dirs:
        sub_dirs = list(class_dir.iterdir())        
        file_num_maps={}
        for sub_dir in sub_dirs:
            low_dirs = list(sub_dir.iterdir())
            for low_dir in low_dirs:
                imgs = list(low_dir.iterdir())
                file_num_maps[low_dir]=len(imgs)

        num = num_per_class
        a = []
        while num > 0:
            m = min(file_num_maps.values())
            
            for k, v in file_num_maps.items():
                num -= m
                if num < 0:
                    m = -1*num
                    break

                for img in k.iterdir():
                    shutil.copy(img, destination / class_dir.name / img.name)

            file_num_maps = {k:v-m for k,v in file_num_maps.items() if v > 0}
"""
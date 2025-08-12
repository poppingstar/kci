from pathlib import Path
import shutil, os


def dataset_split(input_dir:Path|str, val_rate:float = 0.2 , test_rate:float = 0.1):
    input_dir = Path(input_dir)
    assert val_rate + test_rate <= 1, '합계 비율이 1 이하여야 합니다'
    assert input_dir.is_dir(), '입력및 출력 디렉토리는 반드시 디렉토리여야합니다'

    for sub_dir in input_dir.iterdir():
        files = list(sub_dir.iterdir())
        file_num = len(files)
        val_num = int(file_num * val_rate)
        test_num = int(file_num * test_rate)
        
        val_files = files[:val_num]
        test_files = files[val_num:val_num+test_num]
        train_files = files[val_num+test_num:]

        for file_subset, group in ((val_files, 'valid'), (test_files, 'test'), (train_files, 'train')):
            current_dir = input_dir/group/sub_dir.name  
            current_dir.mkdir(exist_ok=True, parents=True)
            for file in file_subset:
                shutil.move(file, current_dir/file.name)
        os.rmdir(sub_dir)


if __name__ == '__main__':
    root = Path(r"C:\Users\user\Desktop\dataset\deepfake-vs-real-60k")

    dataset_split(root, 0.2, 0.1)
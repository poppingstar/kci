from pathlib import Path


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


if __name__ == '__main__':
    root = Path(r"C:\Users\user\Desktop\dataset\deepfake-vs-real-60k")

    split(root)
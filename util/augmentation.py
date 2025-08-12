from torchvision import transforms
from pathlib import Path
from PIL import Image


def augmentation(augs):
    input_dir=Path(r"E:\Datasets\mypaper\after_aug\original")
    output_dir=Path(r"E:\Datasets\mypaper\after_aug\augmented")
    target_num=1000

    augs=transforms.Compose([
                                    transforms.RandomAdjustSharpness(4),
                                    transforms.RandomVerticalFlip(),
                                    transforms.ColorJitter(0.5,0.5,0.5,0.1),
                                    transforms.RandomRotation(90),
                                ])

    for sub in input_dir.iterdir():
        length=sum([1 for _ in sub.iterdir()])
        target_by_class=target_num-length
        num_augment=int(target_by_class/length)
        residual=target_by_class-num_augment*length
        output_subdir=output_dir/sub.name
        output_subdir.mkdir(exist_ok=True,parents=True)

        fname=0
        for file in sub.iterdir():
            num=num_augment
            if residual:
                num=num+1
                residual-=1
            for _ in range(num):
                img=Image.open(file)
                img=augs(img)
                img.save(output_subdir/f'{fname}{file.suffix}')
                fname+=1


def one():
    file=Path(r"E:\Datasets\mypaper\bfore_aug\train\Brahmaea certhia\1.jpg")
    base_img=Image.open(file)
    output_dir=Path(r"E:\Datasets\mypaper\example")

    base_img.save(output_dir/'original.png')
    tl=[
        transforms.RandomAdjustSharpness(1),
        transforms.RandomVerticalFlip(1),
        transforms.ColorJitter(0.5,0.5,0.5,0.1),
        transforms.RandomRotation(90),
    ]

    augs=[]; i=0
    for transform in tl:
        augmented_img=transform(base_img)
        augmented_img.save(output_dir/f'{i}.png')
        i+=1
        augs.append(transform)
        c=transforms.Compose(augs)
        c(base_img).save(output_dir/f'{100-i}.png')


if __name__ == '__main__':
    one()
    print('Done')
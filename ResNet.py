import torch
from torchvision import transforms
import torch.nn as nn
from torch.utils.data import DataLoader
import util.trainer as trainer
from pathlib import Path
import torch
from torchvision import transforms
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
import util.trainer as trainer
from pathlib import Path


if __name__ == '__main__':
    def main():
        model_name = 'ResNet18'
        input_size = 224, 224
        hyper = trainer.TrainConfig(batch_size=256, patience=5, save_point=5, inplace=input_size, workers=16)

        dataset_path = Path(r"E:\Datasets\딥페이크 관련\deepfake-vs-real-60k")
        save_dir = dataset_path/'weights'/model_name
        save_dir = trainer.no_overwrite(save_dir)
        
        transformer = {  #케이스 별 transform 정의
                    'train':transforms.Compose([transforms.Resize(hyper.inplace), transforms.RandomRotation(45) ,transforms.ToTensor()]),
                    'valid':transforms.Compose([transforms.Resize(hyper.inplace), transforms.ToTensor()]),
                    'test':transforms.Compose([transforms.Resize(hyper.inplace),  transforms.ToTensor()])
                    }

        train_dataset = trainer.ImageDir(dataset_path/'train', transformer['train'])
        validation_dataset = trainer.ImageDir(dataset_path/'valid',transformer['valid'])
        test_dataset = trainer.ImageDir(dataset_path/'test', transformer['test'])

        train_loader = DataLoader(train_dataset,hyper.batch_size,True,num_workers=hyper.workers,pin_memory=True,drop_last=False)
        validation_loader = DataLoader(validation_dataset, hyper.batch_size,True,num_workers=hyper.workers,pin_memory=True,drop_last=False)
        test_loader = DataLoader(test_dataset, hyper.batch_size,True,num_workers=hyper.workers,pin_memory=True,drop_last=False)
        class_num = len(train_dataset.classes)

        model_maker = getattr(models, model_name.lower())
        pre_trained_weight = getattr(models, f'{model_name}_Weights').DEFAULT

        model = model_maker(weights = pre_trained_weight)
        model._get_name()

        # trainer.print_named_params(model)

        trainer.layer_freeze(model, 'fc')
        # fc_in_features = model.classifier[1].in_features
        model.fc = nn.Linear(512, out_features=class_num)

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), hyper.lr)
        hyper.set_optimizer(optimizer)
        hyper.save_log(save_dir/'log.txt')

        trainer.train_test(model, train_loader, validation_loader, 
                            test_loader, hyper, save_dir)


    main()
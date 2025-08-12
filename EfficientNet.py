import torch
from torchvision import transforms
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b7, EfficientNet_B7_Weights
import util.trainer as trainer
from pathlib import Path
import torch
from torchvision import transforms
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
import util.trainer as trainer
from pathlib import Path


def get_input_size(model):
    match model:
        case 'EfficientNet_B0':
            input_size = 224,224

        case 'EfficientNet_B1':
            input_size = 240,240

        case 'EfficientNet_B2':
            input_size = 260,260

        case 'EfficientNet_B3':
            input_size = 300,300

        case 'EfficientNet_B4':
            input_size = 380,380

        case 'EfficientNet_B5':
            input_size = 456,456

        case 'EfficientNet_B6':
            input_size = 528,528

        case 'EfficientNet_B7':
            input_size = 600,600

    return input_size


if __name__ == '__main__':
    def main():
        model_name = 'EfficientNet_B3'
        input_size = get_input_size(model_name)
        hyper = trainer.TrainConfig(batch_size=64, patience=5, save_point=5, inplace=input_size, workers=16)

        dataset_path = Path(r"C:\Users\user\Desktop\dataset\deepfake and real")
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

        # trainer.layer_freeze(model, 'features.8')
        fc_in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(fc_in_features, out_features=class_num)

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), hyper.lr)
        hyper.set_optimizer(optimizer)
        hyper.save_log(save_dir/'log.txt')

        trainer.train_test(model, train_loader, validation_loader, 
                            test_loader, hyper, save_dir)

    main()
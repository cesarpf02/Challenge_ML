import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 6]
plt.rcParams['figure.dpi'] = 400
plt.rcParams['font.size'] = 16

import os
from utils import util_general

def plot_loss(history, model_name, plot_training_dir):
    model_plot_dir = os.path.join(plot_training_dir, model_name)
    util_general.create_dir(model_plot_dir)

    # Training results Loss function
    plt.figure(figsize=(8, 6))
    for c in ['train_loss', 'val_loss']:
        plt.plot(history[c], label=c)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.savefig(os.path.join(model_plot_dir, "Loss"), dpi=400)
    plt.show()

def get_pretrained(model_name, num_classes=2, finetune=True):
    if model_name == "alexnet":
        model = models.alexnet(pretrained=True) # optimized weights on ImageNet
        if not finetune:
            for param in model.parameters():
                param.requires_grad=False
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features=num_ftrs, out_features=num_classes)
    elif model_name == "vgg16_bn":
        model = models.vgg16_bn(pretrained=True)
        if not finetune:
            for param in model.parameters():
                param.requires_grad = False
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features=num_ftrs, out_features=num_classes)
    elif model_name == "resnet18":
        model = models.resnet18(pretrained=True)
        if not finetune:
            for param in model.parameters():
                param.requires_grad = False
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    else:
        raise NotImplementedError
    return model
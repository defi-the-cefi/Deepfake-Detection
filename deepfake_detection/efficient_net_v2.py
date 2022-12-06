#%% options for pre-trained arch and weights:
#     "EfficientNet",
#     "EfficientNet_B0_Weights",
#     "EfficientNet_B1_Weights",
#     "EfficientNet_B2_Weights",
#     "EfficientNet_B3_Weights",
#     "EfficientNet_B4_Weights",
#     "EfficientNet_B5_Weights",
#     "EfficientNet_B6_Weights",
#     "EfficientNet_B7_Weights",
#     "EfficientNet_V2_S_Weights",
#     "EfficientNet_V2_M_Weights",
#     "EfficientNet_V2_L_Weights",
#     "efficientnet_b0",
#     "efficientnet_b1",
#     "efficientnet_b2",
#     "efficientnet_b3",
#     "efficientnet_b4",
#     "efficientnet_b5",
#     "efficientnet_b6",
#     "efficientnet_b7",
#     "efficientnet_v2_s",
#     "efficientnet_v2_m",
#     "efficientnet_v2_l"

#%%
import torch
from dataset_object import images_dataset, train_frame, val_frame, test_frame
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_v2_s as architecture
from torchvision.models import EfficientNet_V2_S_Weights as pretrained_weights
# for v2_s might ave to use weights.IMAGENET1K_V1
from training import Optimization

min_image_size = 384
dataloader_workers = 32

# %% Model Hyperparameter
# Device configuration
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
model_name = 'Effnet'
# Hyper-parameters
num_epochs = 50
batch_size = 50
shuffle_batches = False
# dropout = 0.3
learning_rate = 1e-5
weight_decay = 1e-8
print('defined all hyperparams')

# changed num classes to 2 (real [1,0] or fake [0,1])

#%% Print Summary of Arch
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print (name, param.data)

#%%
# no_grad = []
# tensor_device = []
# for name, param in model.named_parameters():
#     if not param.requires_grad:
#         no_grad.append(name)
#     tensor_device.append(param.data.device)
# print(tensor_device)
# print('tensors with no grad: ', sum(no_grad))



#%% Model training specs


def main():
    # %%
    # Dataset Generation
    train_images = images_dataset(dataset_spec_frame=train_frame, min_image_size=min_image_size)
    train_loader = DataLoader(train_images, batch_size=batch_size, shuffle=True, num_workers=dataloader_workers)
    print(train_images[0])

    val_images = images_dataset(dataset_spec_frame=val_frame, min_image_size=min_image_size)
    val_loader = DataLoader(val_images, batch_size=batch_size, shuffle=True, num_workers=dataloader_workers)

    test_images = images_dataset(dataset_spec_frame=test_frame, min_image_size=min_image_size)
    test_loader = DataLoader(test_images, batch_size=batch_size, shuffle=True, num_workers=dataloader_workers)
    print('generated required datasets')

    # %% Model Specification
    model = architecture(weights=pretrained_weights.DEFAULT, progress=True)
    print(model)

    # for efficientnet_v2_s, last_channel = None, which means lastconv_output_channels = 4*last_conv)input_channels

    print(model.classifier)
    print('above is our original classifier layer')
    model.classifier[1] = torch.nn.Linear(in_features=1280, out_features=2, bias=True)
    print(model.classifier)
    print('above is our modded classifier layer')
    model = model.to(device)
    print('sent to gpu')
    print('finished defining model specs')
    # Cross Entropy Loss Function
    criterion = torch.nn.CrossEntropyLoss()
    # Adam
    # https://arxiv.org/abs/1412.6980
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # optimal ADAM parameters: Good default settings for the tested machine learning problems are stepsize "lr" = 0.001,
    #  Exponential decay rates for the moment estimates "betas" β1 = 0.9, β2 = 0.999 and
    #  epsilon decay rate "eps" = 10−8
    # AdamW decouple weight decay regularization improving upon standard Adam
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # https://arxiv.org/abs/1711.05101

    # Learning Rate Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=True)
    # Cosine Annealing LR
    # torch.optim.lr_scheduler.CosineAnnealingLR
    opt = Optimization(model=model, model_name=model_name, loss_fn=criterion, optimizer=optimizer, device=device, scheduler=scheduler)
    opt.train(train_loader, val_loader, batch_size=batch_size, n_epochs=num_epochs)

if __name__ == '__main__':
    main()

import segmentation_models_pytorch as smp
import torch
import albumentations as albu
from dataset import GetDatasetSgm, annotation_files


def train():
    class_id, batch_size, epochs, lr = 100, 1, 50, 1e-4

    train_transforms = get_train_transforms()
    val_transforms = get_val_transforms()
    train_files = annotation_files("../data/train", "images", "labels")
    val_files = annotation_files("../data/val", "images", "labels")
    train_dataset = GetDatasetSgm(train_files, [class_id], train_transforms)
    val_dataset = GetDatasetSgm(val_files, [class_id], val_transforms)
    train_loader = getLoader(train_dataset, batch_size)
    val_loader = getLoader(val_dataset, 1)

    model = getModel()
    loss_func = smp.losses.DiceLoss("multilabel", from_logits=False)
    optimizer = torch.optim.RAdam([dict(params=model.parameters(), lr=lr)])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=scheduler_func
    )

    for epoch in range(epochs):
        print(f"train step, epoch: {epoch + 1}/{epochs}")
        model.train()
        for i, (images, masks) in enumerate(train_loader):
            optimizer.zero_grad()
            images = images.cuda()
            masks = masks.cuda()
            pred = model(images)
            loss = loss_func(pred, masks)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch}, iteration {i}, loss: {loss.item()}")

        model.eval()
        val_loss = 0
        for images, masks in val_loader:
            images = images.cuda()
            masks = masks.cuda()
            pred = model(images)
            loss = loss_func(pred, masks)
            val_loss += loss.item()
        print(f"Epoch {epoch}, validation loss: {val_loss / len(val_loader)}")

        scheduler.step()

    torch.save(model.state_dict(), "model.pth")


def getModel():
    model = smp.FPN(
        encoder_name="efficientnet-b7",
        encoder_weights="imagenet",
        activation="sigmoid",
        in_channels=3,
        classes=1,
    )
    model = model.to(torch.device("cuda"))
    return model


def getLoader(dataset, batch_size):
    return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            pin_memory=True,
            drop_last=True,
            num_workers=2,
            shuffle=True,
    )


def get_train_transforms():
    return albu.Compose(
        [
            albu.Resize(256, 512),
            albu.HorizontalFlip(),
            albu.ShiftScaleRotate(
                border_mode=0,
                rotate_limit=90,
                scale_limit=0.5,
                shift_limit=0.2
            ),
            albu.RandomBrightnessContrast(),
            albu.MotionBlur(),
            albu.GaussianBlur(),
        ],
        is_check_shapes=False,
    )


def get_val_transforms():
    return albu.Compose([albu.Resize(256, 512)], is_check_shapes=False)


def scheduler_func(epoch):
    return 0.95 ** (epoch - 10) if epoch > 10 else 1


if __name__ == '__main__':
    train()

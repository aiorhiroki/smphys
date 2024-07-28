import segmentation_models_pytorch as smp
import torch
import albumentations as albu
from dataset import GetDatasetSgm, annotation_files


def eval():
    class_id = 100
    test_transforms = get_val_transforms()
    test_files = annotation_files("../data/test", "images", "labels")
    test_dataset = GetDatasetSgm(test_files, [class_id], test_transforms)
    test_loader = getLoader(test_dataset, 1)
    model = getModel()
    model.eval()
    loss_func = smp.losses.DiceLoss("multilabel", from_logits=False)

    test_loss = 0
    for images, masks in test_loader:
        images = images.cuda()
        masks = masks.cuda()
        pred = model(images)
        loss = loss_func(pred, masks)
        test_loss += loss.item()

    print(f"test loss: {test_loss / len(test_loader)}")


def getModel():
    model = smp.FPN(
        encoder_name="efficientnet-b7",
        encoder_weights="imagenet",
        activation="sigmoid",
        in_channels=3,
        classes=1,
    )
    model.load_state_dict(torch.load('model.pth'))
    model = model.to(torch.device("cuda"))
    return model


def get_val_transforms():
    return albu.Compose([albu.Resize(256, 512)], is_check_shapes=False)


def getLoader(dataset, batch_size):
    return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            pin_memory=True,
            drop_last=True,
            num_workers=2,
            shuffle=False,
    )


if __name__ == "__main__":
    eval()

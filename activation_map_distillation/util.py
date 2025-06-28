import cv2
import torch
import numpy as np
import math
import os
import sys
import shutil
import random
import albumentations as A
import torch.nn.functional as F
from constants import OCT_DEFAULT_MEAN, OCT_DEFAULT_STD
from albumentations.pytorch.transforms import ToTensorV2

np.random.seed(42)


# Class DataUpdater, used to update the statistic data such as loss, accuracy.
class DataUpdater(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# Class Matrics, used to calculate top_k accuracy, sensitivity, specificity, etc.
class Matrics(object):
    def accuracy(self, output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)  # Pytorch 1.7
                res.append(correct_k.mul_(100.0 / batch_size))

            return res


# define the dataAugmentation for supervised_train
class DAForSupervisedTrain(object):
    def __init__(self, args):
        self.transform = A.Compose([
            A.RandomResizedCrop(height=args.frame_height, width=args.frame_width,
                                scale=(0.8, 1.0), ratio=(1.8, 2.2),
                                interpolation=cv2.INTER_CUBIC),
            A.HorizontalFlip(p=0.5),
        ])

        self.img_transform = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
            A.GaussianBlur(p=0.1),
            A.Normalize(mean=OCT_DEFAULT_MEAN,
                        std=OCT_DEFAULT_STD),
            ToTensorV2()
        ])

    def __call__(self, image, activation_map, mask):
        # 将热力图和掩码堆叠成一个多通道数组
        stacked_masks = np.stack((activation_map, mask), axis=-1)  # [H, W, 2]
        augmented = self.transform(image=image, mask=stacked_masks)
        image = augmented['image']
        two_mask = augmented['mask']  # [H, W, 2] 这个地方先对mask不做处理，就是float32的二维数组
        two_mask = torch.from_numpy(two_mask)  # [H, W, 2]
        image = self.img_transform(image=image)['image']  # [3, H, W] torch.Tensor
        activation_map = two_mask[..., 0]
        mask = two_mask[..., 1]
        return image, activation_map, mask


class DAForFinetuningTrain(object):
    def __init__(self, args):
        self.transform = A.Compose([
            A.RandomResizedCrop(height=args.frame_height, width=args.frame_width,
                                scale=(0.8, 1.0), ratio=(1.8, 2.2),
                                interpolation=cv2.INTER_CUBIC),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
            A.GaussianBlur(p=0.1),
            A.HorizontalFlip(),
            A.Normalize(mean=OCT_DEFAULT_MEAN,
                        std=OCT_DEFAULT_STD),
            ToTensorV2()
        ])

    def __call__(self, image):
        image = self.transform(image=image)['image']
        return image


# define the dataAugmentation for val_or_test
class DAForValOrTest(object):
    def __init__(self, args):
        self.transform = A.Compose([
            A.Resize(height=args.frame_height, width=args.frame_width,
                     interpolation=cv2.INTER_CUBIC),
            A.Normalize(mean=OCT_DEFAULT_MEAN,
                        std=OCT_DEFAULT_STD),
            ToTensorV2()
        ])

    def __call__(self, image, activation_map, mask):
        image = self.transform(image=image)['image']
        return image, activation_map, mask  # shape is (3, H, W)


class ImageProcessorLabel(object):
    def __init__(self, args, augmentation='supervised_train'):
        self.augmentation = augmentation
        self.args = args

    def __call__(self, img, activation_map, mask):
        if self.augmentation in ['supervised_train', 'distilling_train']:
            data_augmentation = DAForSupervisedTrain(self.args)
        elif self.augmentation == 'valid_or_test':
            data_augmentation = DAForValOrTest(self.args)
        else:
            print('not support data augmentation type !')
            sys.exit(-1)
        # 不同数据增强方式返回的图片数量不同，后面需要做适当的调整
        image, activation_map, mask = data_augmentation(img, activation_map, mask)
        return image, activation_map, mask


class ImageProcessorUnLabel(object):
    def __init__(self, args, augmentation='distilling_train_nolabel'):
        self.augmentation = augmentation
        self.args = args

    def __call__(self, img):
        if self.augmentation == 'distilling_train_nolabel':
            data_augmentation = DAForFinetuningTrain(self.args)
        else:
            print('not support data augmentation type !')
            sys.exit(-1)
        # 不同数据增强方式返回的图片数量不同，后面需要做适当的调整
        image = data_augmentation(img)
        return image


def seeding(seed):
    """
    Set the seed for randomness.
    :param seed: int; the seed for randomness.
    :return: None.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.enabled = False


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    print("Set warmup steps = %d" % warmup_iters)

    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule


def create_directory(directory):
    """
    Create the directory.
    :param directory: str; the directory path we want to create.
    :return: None
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_checkpoint(state, checkpoint_dir, is_best=False, filename='checkpoint.pth'):
    """
    :param state:
    :param checkpoint_dir: 存储的检查点目录路径
    :param is_best: 自监督训练时is_best为False, 监督训练和微调时动态传入
    :param filename:
    :return:
    """
    filename = os.path.join(checkpoint_dir, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(checkpoint_dir, 'model_best.pth'))


if __name__ == '__main__':
    # color_transform = A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=1)
    # sol_transform = A.Solarize(p=1.0)
    # img = cv2.imread('2.png', cv2.COLOR_BGR2GRAY)
    # img = cv2.imread('2.png')
    # img = A.GaussianBlur(blur_limit=(9, 11))(image=img)['image']
    # print(img.shape)

    # col_img = color_transform(image=img)['image']
    # sol_img = sol_transform(image=img)['image']
    # print(img.shape)
    # cv2.imshow('img', img)
    # # cv2.imshow('col_img', col_img)
    # cv2.imshow('sol_img', sol_img)
    # cv2.waitKey(0)
    # print(col_img)

    # a = np.random.rand(224, 448, 3)
    #
    # a = np.clip(a, 0, 1)  # 将numpy数组约束在[0, 1]范围内
    # a = (a * 255).astype(np.uint8)
    #
    # im = Image.fromarray(a)
    # print(im.size)
    pass

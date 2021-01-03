from .avazu import AvazuDataset
from .cifar10 import Cifar10
from .cifar100 import Cifar100
import zeus
if zeus.is_gpu_device():
    from .cityscapes import Cityscapes
    from .div2k import DIV2K
    from .div2k_unpair import Div2kUnpair
    from .fmnist import FashionMnist
    from .imagenet import Imagenet
    from .mnist import Mnist
    from .sr_datasets import Set5, Set14, BSDS100
#   from .auto_lane_datasets import AutoLaneDataset
    from .cls_ds import ClassificationDataset
    from .coco import CocoDataset
    from .mrpc import MrpcDataset
#   from .nasbench101 import Nasbench101
#   from .nasbench201 import Nasbench201

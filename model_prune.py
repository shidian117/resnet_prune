# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import logging
import paddle
import argparse
import functools
import math
import time
import numpy as np
import paddleslim
from paddleslim.common import get_logger
from paddleslim.analysis import dygraph_flops as flops
from paddle.distributed import ParallelEnv
from paddle.static import load_program_state
from paddle.vision.models import mobilenet_v1
from paddle.vision.models import resnet18
import paddle.vision.transforms as T
from paddleslim.dygraph.quant import QAT

sys.path.append(os.path.join(os.path.dirname("__file__")))
#from mobilenet_v3 import MobileNetV3_large_x1_0
from optimizer import create_optimizer
sys.path.append(
    os.path.join(os.path.dirname("__file__"), os.path.pardir, os.path.pardir))
from utility import add_arguments, print_arguments

_logger = get_logger(__name__, level=logging.INFO)

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
#lr_strategy cosine_decay; piecewise_decay;mobilenet_v3
add_arg('batch_size',               int,    256,                                         "Single Card Minibatch size.")
add_arg('use_gpu',                  bool,   True,                                        "Whether to use GPU or not.")
add_arg('pretrained_model',         str,    "./pre_trained_model/best_model_train",    "Whether to use pretrained model.")
add_arg('model',                    str,    "resnet18",                              "The target model.")
add_arg('lr',                       float,  0.01,                                      "The learning rate used to fine-tune pruned model.")
add_arg('lr_strategy',              str,    "piecewise_decay",                           "The learning rate decay strategy.")
add_arg('l2_decay',                 float,  4e-5,                                        "The l2_decay parameter.")
add_arg('ls_epsilon',               float,  0.1,                                         "Label smooth epsilon.")
add_arg('momentum_rate',            float,  0.9,                                         "The value of momentum_rate.")
add_arg('num_epochs',               int,    40,                                           "The number of total epochs.")
add_arg('total_images',             int,    50000,                                     "The number of total training images.")
add_arg('data',                     str,    "cifar10",                                  "Which data to use. 'cifar10' or 'imagenet'")
add_arg('log_period',               int,    40,                                          "Log period in batches.")
add_arg('model_save_dir',           str,    "./pre_trained_pruned_model",                           "model save directory.")
add_arg('pruned_ratio',             float,    0.25,                                           "The ratios to be pruned.")
add_arg('criterion',                str, "fpgm",         "The prune criterion to be used, support l1_norm and batch_norm_scale.")
#add_arg('--lr-decay', type=str, default='step', help='learning rate decay method, step, cos or sgdr')
#add_arg('--step-size', type=int, default=3, help='step size in stepLR()')
#add_arg('--warmup-epochs', type=int, default=5, help='warmup epochs using in CosineWarmupLR')
parser.add_argument('--step_epochs', nargs='+', type=int, default=[10, 20, 30], help="piecewise decay step")
# yapf: enable

#T.Transpose()
def get_pruned_params(args, model):
    params = []
    if args.model == "mobilenet_v1":
        skip_vars = ['linear_0.b_0',
                     'conv2d_0.w_0']  # skip the first conv2d and last linear
        for sublayer in model.sublayers():
            for param in sublayer.parameters(include_sublayers=False):
                if isinstance(
                        sublayer, paddle.nn.Conv2D
                ) and sublayer._groups == 1 and param.name not in skip_vars:
                    params.append(param.name)
    elif args.model == "mobilenet_v2":
        for sublayer in model.sublayers():
            for param in sublayer.parameters(include_sublayers=False):
                if isinstance(sublayer, paddle.nn.Conv2D):
                    params.append(param.name)
        return params
    elif args.model == "resnet18":
        for sublayer in model.sublayers():
            for param in sublayer.parameters(include_sublayers=False):
                if isinstance(sublayer, paddle.nn.Conv2D):
                    params.append(param.name)
        return params
    else:
        raise NotImplementedError(
            "Current demo only support for mobilenet_v1, mobilenet_v2, resnet18")
    return params

def load_dygraph_pretrain(model, path=None, load_static_weights=False):
    if not (os.path.isdir(path) or os.path.exists(path + '.pdparams')):
        raise ValueError("Model pretrain path {} does not "
                         "exists.".format(path))
    if load_static_weights:
        
        pre_state_dict = load_program_state(path)
        param_state_dict = {}
        model_dict = model.state_dict()
        for key in model_dict.keys():
            #print("\033[1;35m %s \033[0m!" %(key))
            weight_name = model_dict[key].name
            #print(model_dict[key])
            if key in pre_state_dict.keys():
                #print("\033[1;35m %s \033[0m!" %(weight_name))
                #print('Load weight: {}, shape: {}'.format(weight_name, pre_state_dict[weight_name].shape))
                param_state_dict[key] = pre_state_dict[key]
                #param_state_dict[key].name = pre_state_dict[key].name

            else:
                param_state_dict[key] = model_dict[key]
        model.set_dict(param_state_dict)
        return

    param_state_dict = paddle.load(path + ".pdparams")
    model.set_dict(param_state_dict)
    return


def compress(args):
    if args.data == "cifar10":
        transform = T.Compose([ T.RandomCrop(32, padding=4),T.RandomHorizontalFlip(), T.Transpose(),  T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        transform1 = T.Compose([ T.Transpose(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) ])
        train_dataset = paddle.vision.datasets.Cifar10(
            mode="train", backend="cv2", transform=transform)
        val_dataset = paddle.vision.datasets.Cifar10(
            mode="test", backend="cv2", transform=transform1)
        class_dim = 10
        image_shape = [3, 32, 32]
        pretrain = False
        args.total_images = 50000
    elif args.data == "imagenet":
        import imagenet_reader as reader
        train_dataset = reader.ImageNetDataset(mode='train')
        val_dataset = reader.ImageNetDataset(mode='val')
        class_dim = 1000
        image_shape = "3,224,224"
    else:
        raise ValueError("{} is not supported.".format(args.data))

    trainer_num = paddle.distributed.get_world_size()
    use_data_parallel = trainer_num != 1

    place = paddle.set_device('gpu' if args.use_gpu else 'cpu')
    # model definition
    if use_data_parallel:
        paddle.distributed.init_parallel_env()

    pretrain = True if args.data == "cifar10" else False
    if args.model == "mobilenet_v1":
        net = mobilenet_v1(pretrained=pretrain, num_classes=class_dim)
    elif args.model == "mobilenet_v3":
        net = MobileNetV3_large_x1_0(class_dim=class_dim)
        if pretrain:
            load_dygraph_pretrain(net, args.pretrained_model, True)
    elif args.model == "resnet18":
        net = resnet18(num_classes=class_dim)
        if pretrain:
            load_dygraph_pretrain(net, args.pretrained_model, True)
    else:
        raise ValueError("{} is not supported.".format(args.model))
    _logger.info("Origin model summary:")
    paddle.summary(net, (1, 3, 32, 32))

    net.eval()
    if args.criterion == 'fpgm':
        pruner = paddleslim.dygraph.FPGMFilterPruner(net, [1] + image_shape)
    elif args.criterion == 'l1_norm':
        pruner = paddleslim.dygraph.L1NormFilterPruner(net, [1] + image_shape)

    params = get_pruned_params(args, net)
    ratios = {}
    for param in params:
        ratios[param] = args.pruned_ratio
    plan = pruner.prune_vars(ratios, [0])

    _logger.info("FLOPs after pruning: {}GFLOPs; pruned ratio: {}".format(
        flops(net, [1] + image_shape) / 1000, plan.pruned_flops))

    for param in net.parameters():
        if "conv2d" in param.name:
            print("{}\t{}".format(param.name, param.shape))
    '''
     ############################################################################################################
    # 0. quantization configs
    ############################################################################################################
    quant_config = {
        # weight preprocess type, default is None and no preprocessing is performed. 
        'weight_preprocess_type': None,
        # activation preprocess type, default is None and no preprocessing is performed.
        'activation_preprocess_type': None,
        # weight quantize type, default is 'channel_wise_abs_max'
        'weight_quantize_type': 'channel_wise_abs_max',
        # activation quantize type, default is 'moving_average_abs_max'
        'activation_quantize_type': 'moving_average_abs_max',
        # weight quantize bit num, default is 8
        'weight_bits': 8,
        # activation quantize bit num, default is 8
        'activation_bits': 8,
        # data type after quantization, such as 'uint8', 'int8', etc. default is 'int8'
        'dtype': 'int8',
        # window size for 'range_abs_max' quantization. default is 10000
        'window_size': 10000,
        # The decay coefficient of moving average, default is 0.9
        'moving_rate': 0.9,
        # for dygraph quantization, layers of type in quantizable_layer_type will be quantized
        'quantizable_layer_type': ['Conv2D', 'Linear'],
    }

    #if args.use_pact:
    #    quant_config['activation_preprocess_type'] = 'PACT'

    ############################################################################################################
    # 2. Quantize the model with QAT (quant aware training)
    ############################################################################################################

    quanter = QAT(config=quant_config)
    quanter.quantize(net)
    '''
    ############################################################################################################
    # 1. training
    ############################################################################################################

    _logger.info("QAT model summary:")
    paddle.summary(net, (1, 3, 32, 32))

    opt, lr = create_optimizer(net, trainer_num, args)

    if use_data_parallel:
        net = paddle.DataParallel(net)

    train_batch_sampler = paddle.io.DistributedBatchSampler(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    train_loader = paddle.io.DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        places=place,
        return_list=True,
        num_workers=1,
        use_shared_memory=False)

    valid_loader = paddle.io.DataLoader(
        val_dataset,
        places=place,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        return_list=True,
        num_workers=1,
        use_shared_memory=False)

    @paddle.no_grad()
    def test(epoch, net):
        net.eval()
        batch_id = 0
        acc_top1_ns = []
        acc_top5_ns = []

        eval_reader_cost = 0.0
        eval_run_cost = 0.0
        total_samples = 0
        reader_start = time.time()
        for data in valid_loader():
            eval_reader_cost += time.time() - reader_start
            image = data[0]
            label = data[1]
            if args.data == "cifar10":
                label = paddle.reshape(label, [-1, 1])

            eval_start = time.time()

            out = net(image)
            acc_top1 = paddle.metric.accuracy(input=out, label=label, k=1)
            acc_top5 = paddle.metric.accuracy(input=out, label=label, k=5)

            eval_run_cost += time.time() - eval_start
            batch_size = image.shape[0]
            total_samples += batch_size
            '''
            if batch_id % args.log_period == 0:
                log_period = 1 if batch_id == 0 else args.log_period
                _logger.info(
                    "Eval epoch[{}] batch[{}] - top1: {:.6f}; top5: {:.6f}; avg_reader_cost: {:.6f} s, avg_batch_cost: {:.6f} s, avg_samples: {}, avg_ips: {:.3f} images/s".
                    format(epoch, batch_id,
                           np.mean(acc_top1.numpy()),
                           np.mean(acc_top5.numpy()), eval_reader_cost /
                           log_period, (eval_reader_cost + eval_run_cost) /
                           log_period, total_samples / log_period, total_samples
                           / (eval_reader_cost + eval_run_cost)))
                eval_reader_cost = 0.0
                eval_run_cost = 0.0
                total_samples = 0
            '''
            acc_top1_ns.append(np.mean(acc_top1.numpy()))
            acc_top5_ns.append(np.mean(acc_top5.numpy()))
            batch_id += 1
            reader_start = time.time()

        _logger.info(
            "Final eval epoch[{}] - \033[1;35m acc_top1: {:.6f} \033[0m! ; acc_top5: {:.6f}".format(
                epoch,
                np.mean(np.array(acc_top1_ns)), np.mean(np.array(acc_top5_ns))))
        return np.mean(np.array(acc_top1_ns))

    def cross_entropy(input, target, ls_epsilon):
        if ls_epsilon > 0:
            if target.shape[-1] != class_dim:
                target = paddle.nn.functional.one_hot(target, class_dim)
            target = paddle.nn.functional.label_smooth(
                target, epsilon=ls_epsilon)
            target = paddle.reshape(target, shape=[-1, class_dim])
            input = -paddle.nn.functional.log_softmax(input, axis=-1)
            cost = paddle.sum(target * input, axis=-1)
        else:
            cost = paddle.nn.functional.cross_entropy(input=input, label=target)
        avg_cost = paddle.mean(cost)
        return avg_cost

    def train(epoch, net):

        net.train()
        batch_id = 0

        train_reader_cost = 0.0
        train_run_cost = 0.0
        total_samples = 0
        reader_start = time.time()
        for data in train_loader():
            train_reader_cost += time.time() - reader_start

            image = data[0]
            label = data[1]
            if args.data == "cifar10":
                label = paddle.reshape(label, [-1, 1])

            train_start = time.time()
            out = net(image)
            avg_cost = cross_entropy(out, label, args.ls_epsilon)

            acc_top1 = paddle.metric.accuracy(input=out, label=label, k=1)
            acc_top5 = paddle.metric.accuracy(input=out, label=label, k=5)
            avg_cost.backward()
            opt.step()
            opt.clear_grad()
            lr.step()

            loss_n = np.mean(avg_cost.numpy())
            acc_top1_n = np.mean(acc_top1.numpy())
            acc_top5_n = np.mean(acc_top5.numpy())

            train_run_cost += time.time() - train_start
            batch_size = image.shape[0]
            total_samples += batch_size

            if batch_id % args.log_period == 0:
                log_period = 1 if batch_id == 0 else args.log_period
                _logger.info(
                    "epoch[{}]-batch[{}] lr: {:.6f} - Training_loss: {:.6f}; top1: {:.6f}; top5: {:.6f}; avg_reader_cost: {:.6f} s, avg_batch_cost: {:.6f} s, avg_samples: {}, avg_ips: {:.3f} images/s".
                    format(epoch, batch_id,
                           lr.get_lr(), loss_n, acc_top1_n, acc_top5_n,
                           train_reader_cost / log_period, (
                               train_reader_cost + train_run_cost) / log_period,
                           total_samples / log_period, total_samples / (
                               train_reader_cost + train_run_cost)))
                train_reader_cost = 0.0
                train_run_cost = 0.0
                total_samples = 0
            batch_id += 1
            reader_start = time.time()

    ############################################################################################################
    # train loop
    ############################################################################################################
    best_acc1 = 0.0
    best_epoch = 0
    for i in range(args.num_epochs):
        train(i, net)
        acc1 = test(i, net)
        if acc1 > best_acc1:
            best_acc1 = acc1
            best_epoch = i
    if paddle.distributed.get_rank() == 0:
        model_prefix1 = os.path.join(args.model_save_dir, "best_model_pruned_train")
        paddle.save(net.state_dict(), model_prefix1 + ".pdparams")
        paddle.save(opt.state_dict(), model_prefix1 + ".pdopt")
        model_prefix2 = os.path.join(args.model_save_dir, "best_model_pruned_infer")
        paddle.jit.save(net, model_prefix2, input_spec=[paddle.static.InputSpec(shape=[None, 3, 32, 32], dtype='float32')])


def main():
    args = parser.parse_args()
    print_arguments(args)
    compress(args)


if __name__ == '__main__':
    main()

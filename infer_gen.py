# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
'''
Paddle-Lite light python api demo
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
import numpy as np
import argparse
import paddle.vision.transforms as T
import paddle
import logging

from paddlelite.lite import *
from paddleslim.common import get_logger
_logger = get_logger(__name__, level=logging.INFO)



def test():
        batch_id = 0
        acc_top1_ns = []
        acc_top5_ns = []

        eval_reader_cost = 0.0
        eval_run_cost = 0.0
        total_samples = 0
        reader_start = time.time()

        place = paddle.set_device('gpu:3' if True else 'cpu')
        transform1 = T.Compose([ T.Transpose(), T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)) ])
        val_dataset = paddle.vision.datasets.Cifar10(mode="test", backend="cv2", transform=transform1)
        valid_loader = paddle.io.DataLoader(
        val_dataset,
        batch_size=256,
        places=place,
        shuffle=False,
        drop_last=False,
        return_list=True,
        num_workers=4,
        use_shared_memory=True)
        #print(valid_loader())


        
        a = time.time()
        for data in valid_loader():
            eval_reader_cost += time.time() - reader_start
            image = data[0].numpy()
            label = data[1]
            label = paddle.reshape(label, [-1, 1])
            

            eval_start = time.time()

            out = RunModel(args,image)
            out1 = paddle.to_tensor(out)

            acc_top1 = paddle.metric.accuracy(input=out1, label=label, k=1)
            print(acc_top1.numpy())
            acc_top5 = paddle.metric.accuracy(input=out1, label=label, k=5)

            eval_run_cost += time.time() - eval_start
            batch_size = image.shape[0]
            total_samples += batch_size
         
            acc_top1_ns.append(np.mean(acc_top1.numpy()))
            acc_top5_ns.append(np.mean(acc_top5.numpy()))
            batch_id += 1
            reader_start = time.time()
        b = time.time()
        print('time')
        print(b-a)

        _logger.info(
            "Final eval - \033[1;35m acc_top1: {:.6f} \033[0m! ; acc_top5: {:.6f}".format(
                np.mean(np.array(acc_top1_ns)), np.mean(np.array(acc_top5_ns))))
        return np.mean(np.array(acc_top1_ns))


def RunModel(args,image):
    # 1. Set config information
    config = MobileConfig()
    config.set_model_from_file(args.model_dir)

    # 2. Create paddle predictor
    predictor = create_paddle_predictor(config)


    # 3. Set input data
    input_tensor = predictor.get_input(0)
    #input_tensor.from_numpy(np.ones((1, 3, 32, 32)).astype("float32"))
    input_tensor.from_numpy(image.astype("float32"))

    # 4. Run model
    predictor.run()

    # 5. Get output data
    output_tensor = predictor.get_output(0)
    output_data = output_tensor.numpy()
    #print(output_data)
    return output_data

# Command arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_dir", default="./best_model_pruned_infer_opt8.nb", type=str, help="Non-combined Model dir path")

if __name__ == '__main__':
    args = parser.parse_args()
    test()

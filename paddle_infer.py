import argparse
import numpy as np

# 引用 paddle inference 预测库
import paddle.inference as paddle_infer
import time

import paddle.vision.transforms as T
import paddle
import logging

#from paddlelite.lite import *
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
        drop_last=True,
        return_list=True,
        num_workers=1,
        use_shared_memory=True)
        #print(valid_loader())


        
        a = time.time()
        for data in valid_loader():
            eval_reader_cost += time.time() - reader_start
            image = data[0].numpy()
            label = data[1]
            label = paddle.reshape(label, [-1, 1])
            

            eval_start = time.time()

            out_temp = run1(args,image)
            out = run2(args,out_temp)
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



def run1(args,image):

    # 创建 config
    config = paddle_infer.Config(args.model_file1, args.params_file1)

    # 根据 config 创建 predictor
    predictor = paddle_infer.create_predictor(config)

    # 获取输入的名称
    input_names = predictor.get_input_names()
    input_handle = predictor.get_input_handle(input_names[0])

    # 设置输入
    #fake_input = np.random.randn(args.batch_size, 3, 318, 318).astype("float32")
    input_handle.reshape([args.batch_size, 3, 32, 32])
    input_handle.copy_from_cpu(image)

    # 运行predictor
    predictor.run()

    # 获取输出
    output_names = predictor.get_output_names()
    output_handle = predictor.get_output_handle(output_names[0])
    output_data = output_handle.copy_to_cpu() # numpy.ndarray类型
    #print("Output data size is {}".format(output_data.size))
    #print("Output data shape is {}".format(output_data.shape))
    return output_data

def run2(args,image):

    # 创建 config
    config = paddle_infer.Config(args.model_file2, args.params_file2)

    # 根据 config 创建 predictor
    predictor = paddle_infer.create_predictor(config)

    # 获取输入的名称
    input_names = predictor.get_input_names()
    input_handle = predictor.get_input_handle(input_names[0])

    # 设置输入
    #fake_input = np.random.randn(args.batch_size, 3, 318, 318).astype("float32")
    input_handle.reshape([args.batch_size, 128, 4, 4])
    input_handle.copy_from_cpu(image)

    # 运行predictor
    predictor.run()

    # 获取输出
    output_names = predictor.get_output_names()
    output_handle = predictor.get_output_handle(output_names[0])
    output_data = output_handle.copy_to_cpu() # numpy.ndarray类型
    #print("Output data size is {}".format(output_data.size))
    #print("Output data shape is {}".format(output_data.shape))
    return output_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file1", default="./client_model_infer.pdmodel", type=str, help="model filename")
    parser.add_argument("--params_file1", default="./client_model_infer.pdiparams" ,type=str, help="parameter filename")
    parser.add_argument("--model_file2", default="./server_model_infer.pdmodel", type=str, help="model filename")
    parser.add_argument("--params_file2", default="./server_model_infer.pdiparams" ,type=str, help="parameter filename")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    test()

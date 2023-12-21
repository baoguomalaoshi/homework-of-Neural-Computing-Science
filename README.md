# Homework-of-Neural-Computing-Science
## 第3组 组员
刘晨远	杨武略	李帅博	李佐磊	杨佳逸
## 组员分工
杨武略 在ANN转化SNN作业中，将mobilenetv2转化成SNN网络，并观察精度。

李佐磊 使用ResNet50网络在图像分类数据集上进行训练，并使用训练好的模型进行测试。

## 具体工作
1.将MobileNetV2转换为脉冲神经网络。MobileNetV2作为一种高效的图像处理神经网络，在计算机视觉领域广泛应用，而将其转换为脉冲神经网络则涉及了模型结构、参数和工作方式的重大改变。
在这个转换过程中，我们有机会深入理解脉冲神经网络的工作原理，并将连续数值的权重和激活函数转换为基于时间脉冲的编码和传播方式。以获得令人满意的精度和性能。

代码方面，主要修改了Brain-Cog工程内，examples\Perception_and_Learning\Conversion\burst_conversion这个目录下的代码，修改的结果在“burst_conversion”可以看到。

首先，用cifar10数据集训练一个mobilenetv2 ANN网络，训练300个epoch得到的权重为burst_conversion\CIFAR10_MobileNet.pth。最终的test acc为0.855000

然后，把模型转换成snn。核心步骤为卷积用IF单元替换。

snn模型转换时，输出如下所示。

timestep 001: 0.8552215189873418

timestep 003: 0.8552215189873418

timestep 007: 0.8552215189873418

timestep 015: 0.8552215189873418

timestep 031: 0.8552215189873418

timestep 063: 0.8552215189873418

best acc:  0.8552215189873418

可见相比原本的网络，转换后的snn并没有精度损失。这也是神经网络对参数的精度要求不高导致的。

2.使用ResNet50网络在cifar100和CUB200-2011数据集上进行训练。ResNet网络是目前用途最为广泛的深度神经网络，它提出的残差网络可以有效缓解深度网络中梯度消失的问题，常被用于各种任务的encoder。CUB200-2011数据集是一个细粒度图像分类数据集，在该数据上进行训练的目的是想了解脉冲神经网络在细粒度图像分类上的效果。

代码方面，原项目已经提供了网络结构的代码和各种数据集加载的代码，因此没有做过多的修改，主要是对一些参数进行了调整，如单卡训练时Dataloader的num_workers要设置为1，否则数据集加载无法完整，以及单卡时的batch size大小。

(1)cifar100数据集：使用如下命令进行训练，训练设备为一张11G RTX 2080Ti，训练epoch数目为50。

```
python -u main.py --model resnet50 --node-type IFNode --dataset cifar100 --step 4 --batch-size 32 --act-fun QGateGrad --device 0 --num-classes 100
```

最终在测试集上的结果为:

| epoch | top-1 acc(%) |
| ----- | ------------ |
| 10    | 17.04        |
| 20    | 24.18        |
| 30    | 32.11        |
| 40    | 39.47        |
| 50    | 42.88        |

训练epoch为50次最终的精度达到42.88%，跟训练epoch为200次的resnet18仍有一定差距，这主要是由于训练次数不足导致的，当epoch次数相等时，精度差距会消失。

(2)CUB200-2011数据集:还在训练当中
# Homework-of-Neural-Computing-Science
## 第3组 组员
刘晨远	杨武略	李帅博	李佐磊	杨佳逸
## 组员分工
杨武略 在ANN转化SNN作业中，将mobilenetv2转化成SNN网络，并观察精度。

李佐磊 使用ResNet网络在图像分类数据集上进行训练，并使用训练好的模型进行测试。

刘晨远 基于braincog提供的工具，模拟大脑对于音乐的记忆，通过训练生成全新风格的音乐。

李帅博 保证代码的高效性和可扩展性

杨佳逸 协助解释模型结果

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

2.使用ResNet网络在cifar100和CUB200-2011数据集上进行训练。ResNet网络是目前用途最为广泛的深度神经网络，它提出的残差网络可以有效缓解深度网络中梯度消失的问题，常被用于各种任务的encoder。CUB200-2011数据集是一个细粒度图像分类数据集，在该数据上进行训练的目的是想了解脉冲神经网络在细粒度图像分类上的效果。

代码方面，原项目已经提供了网络结构的代码和各种数据集加载的代码，因此没有做过多的修改，主要是对一些参数进行了调整，如单卡训练时Dataloader的num_workers要设置为1，否则数据集加载无法完整，以及单卡时的batch size大小。

(1)cifar100数据集：使用如下命令进行训练，训练设备为一张11G RTX 2080Ti，训练epoch数目为60，网络模型为resnet50。

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
| 60    | 47.56        |

训练epoch为60次最终的精度达到47.56%，跟训练epoch为200次的resnet18仍有一定差距，这主要是由于训练次数不足导致的，当epoch次数相等时，精度差距会消失。

(2)CUB200-2011数据集:使用如下命令进行训练，训练设备为一张11G RTX 2080Ti，训练epoch数目为50，网络模型为resnet18。

```
python -u main.py --model resnet18 --node-type IFNode --dataset CUB2002011 --step 4 --batch-size 4 --act-fun QGateGrad --device 0 --num-classes 200
```

最终在测试集上的结果为:

| epoch | top-1 acc(%) |
| ----- | ------------ |
| 20    | 4.78         |
| 40    | 7.15         |
| 60    | 10.17        |
| 80    | 12.12        |
| 100   | 13.13        |

从目前的实验结果看，在细粒度图像分类数据集上训练效果并不好，epoch=100时的准确率也只有13.13%。

注意：由于braincog的python包没有更新，本实验是在将braincog项目下载到本地后，然后调用项目里面的包实现的，为了项目的简洁性，提供的代码只有图像分类的训练代码，调用的包并没有上传到本项目。

3.使用braincog提供的LIF神经元，模拟大脑中与听觉有关的脑区：前额皮层、听觉皮层，然后在大脑皮层中整合记忆

训练样本：使用.mid文件进行训练，该模型使用了[晴天，周杰伦]、[melody，陶喆]这两首歌进行训练，然后分别生成周杰伦风格和陶喆风格的sample。

代码参考了musicMemory的代码，修改了其中的模型参数并解决其在macOS系统中的兼容性问题，并对部分py文件进行了修改。

目前模型效果所得出结果效果很差，完全听不出来是某某歌手风格的创作（考虑是否是因为braincog中所使用的纯音乐.mid文件而笔者使用的是包含人声的.mid文件），目前还正在加急修改中以期获得更好的结果。

特别感谢梁倩老师在bilibili上的视频，对于实现原理和底层逻辑的讲解很细致，同时也感谢这部分代码的撰写者，代码结构很清晰，很有学习的价值。

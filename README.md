# Homework-of-Neural-Computing-Science
## 第3组 组员
刘晨远	杨武略	李帅博	李佐磊	杨佳逸
## 组员分工
杨武略 在ANN转化SNN作业中，将mobilenetv2转化成SNN网络，并观察精度。

李帅博 参与优化现有模型的性能

杨佳逸 参与优化现有模型的性能
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






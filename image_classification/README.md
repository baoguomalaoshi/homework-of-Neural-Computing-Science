# 图像分类任务
## 训练
使用bash命令进行训练
```
bash train.sh
```
如果想在后台运行，且实时输出日志，使用如下命令：
```
nohup bash train.sh  > nohup.out 2>&1 &
```
## 结果
cifar100 epoch=60 47.56%

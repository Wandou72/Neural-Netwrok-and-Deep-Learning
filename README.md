# Neural-Netwrok-and-Deep-Learning

该文档说明各个代码文件的作用以及如何执行代码训练并测试模型。

## 文档说明
* main.py: 使用numpy库实现了线性变换层、激活函数、对应的参数更新和梯度计算、反向传播方法，并由此搭建两层线性分类器。
* functions.py: 读入MNIST数据集，并随机划分训练数据和验证数据；在测试集上运行模型并获得准确率。
* visualization.py: 使用matplotlib库可视化网络中的参数。

## 代码执行
执行python main.py进行Grid Search得到在验证集上最优表现的模型，并将其保存为bestpara.pkl；使用该模型进行测试集上的评估。
执行visualization.py对上述模型中的网络参数可视化。

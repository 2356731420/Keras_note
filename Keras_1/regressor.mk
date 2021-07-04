#回归训练

##导入 Numpy 数据包和 Keras 包

```python
import numpy as np
np.random.seed(1337)
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
```

##创建数据集

```python
X = np.linspace(-1,1,900)
np.random.shuffle(X)
Y = 0.5*X +2 +np.random.normal(0,0.05,(900,))

plt.scatter(X,Y)
plt.show()
```

## 训练数据

```python
X_train,Y_train = X[:800],Y[:800] #前800个数据用来训练
```

## 测试数据

```python
X_test,Y_test = X[800:],Y[800:] #800个数据之后的数据用来测试
```

## 创建网络结构

```python
model = Sequential()
model.add(Dense(units=1,input_dim=1))
```

## 选择损失函数和训练集选择方式

```python
model.compile(loss='mse',optimizer='sgd')#乱序训练
```

## 训练过程

```python
print('Training-------')
for step in range(1301):
    cost = model.train_on_batch(X_train,Y_train)
    if step % 100==0:
        print('train cost:',cost)
```

## 测试过程

```python
print('\nTesting--------')
cost = model.evaluate(X_test,Y_test,batch_size=40)
print('test cost',cost)
W,b = model.layers[0].get_weights()
print('Weights=',W,'\nbiases=',b)
```

## 结果输出

```python
Y_perd = model.predict(X_test)
plt.scatter(X_test,Y_test)
plt.plot(X_test,Y_perd)
plt.show()
```

所以代码均来自莫烦python

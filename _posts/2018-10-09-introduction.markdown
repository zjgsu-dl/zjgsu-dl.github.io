---
layout: post
title:  "深度学习简介"
date:   2018-10-09 13:40:00 +0800
---

深度学习是机器学习的一个分支，是一种试图使用包含复杂结构或由多重非线性变换构成的多个处理层对数据进行高层抽象的算法。

## 深度学习

台大李宏毅老师的[深度学习教程][lee-slideshare] [![课件][pdf_icon]{:height="20px", :width="20px"}][lee-pdf] [![课件][pptx_icon]{:height="20px", :width="20px"}][lee-pptx] 深入浅出地介绍了深度学习的基本知识，是比较好的入门教程。

在书方面，[《基于深度学习的自然语言处理》][nlp]虽然是讲自然语言处理的，但其中的深度学习部分比较浅显易懂，可以作为深度学习的入门参考。[《深度学习》（又名“花书”）][dlbook]是深度学习领域最为经典的书，其内容较全，讲解较深入，可以作为进一步深入学习的选择。

在网上教程方面，可以参考前面说的李宏毅老师的[Machine Learning and having it deep and structured][lee-course]。此外吴恩达的[deeplearning.ai][dlai]以及斯坦福大学的[CS231n: Convolutional Neural Networks for Visual Recognition][cs231n]也是非常好的参考。

## Tensorflow

[Tensorflow][]是Google开发的开源深度学习框架，也是现在应用最广的深度学习框架。Tensorflow基于Python，可以用pip直接安装：

~~~ bash
pip install tensorflow
~~~

如果使用[Anaconda][]，也可以用conda来进行安装：

~~~ bash
conda install -c anaconda tensorflow
~~~

安装好之后，可以用下面的代码进行测试：

~~~ python
import tensorflow as tf
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)
~~~

Tensorflow更新较快，很多书上的内容都已经过时，建议大家直接参考[官方文档][tf-docs]（[中文][tf-docs-cn]）来进行学习。

[anaconda]: https://anaconda.org/
[cs231n]: http://cs231n.stanford.edu/
[dlai]: https://www.deeplearning.ai/courses/
[dlbook]: https://book.douban.com/subject/27087503/
[lee-course]: http://speech.ee.ntu.edu.tw/~tlkagk/courses_MLDS18.html
[lee-pdf]: http://speech.ee.ntu.edu.tw/~tlkagk/slide/Tutorial_HYLee_Deep.pdf
[lee-pptx]: http://speech.ee.ntu.edu.tw/~tlkagk/slide/Tutorial_HYLee_Deep.pptx
[lee-slideshare]: http://www.slideshare.net/tw_dsconf/ss-62245351
[nlp]: https://book.douban.com/subject/30236842/
[pdf_icon]: /assets/images/pdf.svg
[pptx_icon]: /assets/images/pptx.svg
[tensorflow]: https://www.tensorflow.org
[tf-docs-cn]: https://tensorflow.google.cn/get_started/
[tf-docs]: https://www.tensorflow.org/tutorials/

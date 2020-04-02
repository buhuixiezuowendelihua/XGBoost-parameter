
## 1、XGBoost算法原理：
关于XGBoost算法的原理部分，有兴趣的可以去看[XGBoost的论文](https://arxiv.org/pdf/1603.02754.pdf)和[陈天奇的PPT](https://homes.cs.washington.edu/~tqchen/pdf/BoostedTree.pdf)。

对英文有障碍的朋友可以去看[刘建平博客](https://www.cnblogs.com/pinard/p/10979808.html)总结的非常好。

## 2、XGBoost库比较：
XGBoost有2种Python接口风格。一种是XGBoost自带的原生Python API接口，另一种是sklearn风格的API接口，两者的实现是基本一样的，仅仅有细微的API使用的不同，主要体现在参数命名上，以及数据集的初始化上面。

xgboost库要求我们必须要提供适合的Scipy环境，如果你是使用anaconda安装的Python，你的Scipy环境应该是没有什么问题。

```py
#windows安装
pip install xgboost #安装xgboost库
pip install --upgrade xgboost #更新xgboost库

#导入库
import xgboost as xgb
```
现在，我们有两种方式可以来使用我们的xgboost库。

**第一种方式，是直接使用xgboost库自己的建模流程。**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200323131718117.jpg)

 1. `DMatrix`

```py
xgboost.DMatrix(data, label=None, weight=None, base_margin=None, missing=None, silent=False, feature_names=None, feature_types=None, nthread=None)
```

 2. `params`

```py
params {eta, gamma, max_depth, min_child_weight, max_delta_step, subsample, colsample_bytree,
colsample_bylevel, colsample_bynode, lambda, alpha, tree_method string, sketch_eps, scale_pos_weight, updater,
refresh_leaf, process_type, grow_policy, max_leaves, max_bin, predictor, num_parallel_tree}
```

 3. `train`

```py
xgboost.train(params, dtrain, num_boost_round=10, evals=(), obj=None, feval=None, maximize=False, early_stopping_rounds=None, evals_result=None, verbose_eval=True, xgb_model=None, callbacks=None)
```

 4. `predict`

```py
predict(data, output_margin=False, ntree_limit=0, pred_leaf=False, pred_contribs=False, approx_contribs=False, pred_interactions=False, validate_features=True, training=False)
```

其中最核心的，是DMtarix这个读取数据的类，以及train()这个用于训练的类。与sklearn把所有的参数都写在类中的方式不同，xgboost库中必须先使用字典设定参数集，再使用train来将参数及输入，然后进行训练。会这样设计的原因，是因为XGB所涉及到的参数实在太多，全部写在xgb.train()中太长也容易出错。


**第二种方法，使用xgboost库中的sklearn的API**

```py
xgboost.XGBRegressor (max_depth=3, 
                      learning_rate=0.1, 
                      n_estimators=100, 
                      silent=True, 
                      objective='reg:squarederror', 
                      booster='gbtree', 
                      n_jobs=1, 
                      nthread=None, 
                      gamma=0, 
                      min_child_weight=1, 
                      max_delta_step=0, 
                      subsample=1, 
                      colsample_bytree=1, 
                      colsample_bylevel=1, 
                      reg_alpha=0, 
                      reg_lambda=1, 
                      scale_pos_weight=1, 
                      base_score=0.5, 
                      random_state=0, 
                      seed=None, 
                      missing=None, 
                      importance_type='gain', 
                      **kwargs)


```

```py
xgboost.XGBClassifier(max_depth=3, 
                      learning_rate=0.1, 
                      n_estimators=100, 
                      silent=True, 
                      objective='binary:logistic', 
                      booster='gbtree', 
                      n_jobs=1, 
                      nthread=None, 
                      gamma=0, 
                      min_child_weight=1, 
                      max_delta_step=0, 
                      subsample=1, 
                      colsample_bytree=1, 
                      colsample_bylevel=1, 
                      reg_alpha=0, 
                      reg_lambda=1, 
                      scale_pos_weight=1, 
                      base_score=0.5, 
                      random_state=0, 
                      seed=None, 
                      missing=None, 
                      **kwargs
                     )
```
调用xgboost.train和调用sklearnAPI中的类XGBRegressor，需要输入的参数是不同的，而且看起来相当的不同。但其实，这些参数只是写法不同，功能是相同的。比如说，我们的params字典中的第一个参数eta，其实就是我们XGBRegressor里面的参数learning_rate，他们的含义和实现的功能是一模一样的。只不过在sklearnAPI中，开发团队友好地帮助我们将参数的名称调节成了与sklearn中其他的算法类更相似的样子。


所以对我们来说，**使用xgboost中设定的建模流程来建模，和使用sklearnAPI中的类来建模，模型效果是比较相似的，但是xgboost库本身的运算速度（尤其是交叉验证）以及调参手段比sklearn要简单。**

### 3、XGBoost库参数总结
参数含义 |`xgb.train()`| `XGBRegressor()`|
|--|--|--|
|集成中弱评估器的数量| num_round，默认10| n_estimators，默认100
|训练中是否打印每次训练的结果| slient，默认False| slient，默认True
随机抽样的时候抽取的样本比例，范围(0,1]| subsample，默认1|subsample，默认1|
集成中的学习率，又称为步长以控制迭代速率，常用于防止过拟合|eta，默认0.3取值范围[0,1]|learning_rate，默认0.1取值范围[0,1]
使用哪种弱评估器|xgb_model 可以输入gbtree，gblinear或dart。输入的评估器不同，使用的params参数也不同，每种评估器都有自己的params列表。评估器必须于param参数相匹配，否则报错。|booster 可以输入gbtree，gblinear或dart。gbtree代表梯度提升树，dart是Dropouts meet Multiple Additive Regression Trees，可译为抛弃提升树，在建树的过程中会抛弃一部分树，比梯度提升树有更好的防过拟合功能。输入gblinear使用线性模型。
损失函数|<ol><li>obj:默认binary:logistic使用逻辑回归的损失函数，对数损失log_loss，二分类时使用；</li><li>obj:可选multi:softmax 使用softmax损失函数，多分类时使用;</li><li>obj:可选binary:hinge 使用支持向量机的损失函数，Hinge Loss，二分类时使用</li></ol>|objective：默认reg:squarederror 使用均方误差，回归时使用
L1正则项的参数| alpha，默认0，取值范围[0, +∞]| reg_alpha，默认0，取值范围[0, +∞]|
|L2正则项的参数| lambda，默认1，取值范围[0, +∞] |reg_lambda，默认1，取值范围[0, +∞]|
复杂度的惩罚项| gamma，默认0，取值范围[0, +∞] |gamma，默认0，取值范围[0, +∞]|
树的最大深度 |max_depth，默认6 |max_depth，默认6
每次生成树时随机抽样特征的比例 |colsample_bytree，默认1 |colsample_bytree，默认1
|每次生成树的一层时随机抽样特征的比例 |colsample_bylevel，默认1 |colsample_bylevel，默认1
每次生成一个叶子节点时随机抽样特征的比例|colsample_bynode，默认1 |N.A.
一个叶子节点上所需要的最小即叶子节点上的二阶导数之和类似于样本权重|min_child_weight，默认1 |min_child_weight，默认1
控制正负样本比例，表示为负/正样本比例在样本不平衡问题中使用| scale_pos_weight，默认1 |scale_pos_weight，默认1

**更多调参细节可以到我的GitHub上面查看。**

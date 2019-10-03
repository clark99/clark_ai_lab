# 1 传统关联规则挖掘

## 1.1 简介
- **目标**: 发现事务数据库不同项之间的联系, 这些联系构成的规则, 可以帮助找到某些行为特征,帮忙决策

- **场景**: 超市, 网络浏览偏好, 入侵检测, 生产等领域

- **与序列挖掘的不同**:
    - 不考虑事务内,或者事件之间的先后顺序
    - 只考虑前件,后件
    - 不相交的项集

### 1.1.1 相关的概念

- **全局项I**: I={$i_{1},i_{2},...,i_{j},...,i_{m}$}

- **事务数据库T**: T={$t_{1},t_{2},...,t_{i},...,t_{n}$}

- **项集**: 非空不重复 例如 $I_{1}={i_{1},i_{2},i_{3}}$

- **事务$t_{i}$**: 每个事务对应一个项集

## 1.2 关联规则的度量

### 1.2.1 关联规则

- 关联规则:
    - 前件: 项集X
    - 后件: 项集Y
    - 不想交: X和Y的交集为空集

### 1.2.2 支持度

**项集的支持度**
- 一个项集$I_{i}$的支持度
- 分子: 包含$I_{i}$的事务$t_{i}$的数目
- 分母: 事务数据库T中事务的总数

**关联规则的支持度**
- 关联规则: X前件 -> Y后件
- 分子: 同时包含X和Y的事务的总数. 也是就是包含项集(X并集Y)的事务总数
- 分母: 事务数据库T中事务的总数
- 概率形势符号表达: $$suport(X,Y)=P(X 并集 Y)$$

### 1.2.3 置信度

- 关联规则: X前件 -> Y后件
- 分子: 同时包含X和Y的事务的总数. 也是就是包含项集(X并集Y)的事务总数
- 分母: 事务数据库T中包含X项集的的事务$t_{i}$的数目
- 概率形势符号表达:$$confidence(X,Y)= P(Y|X)$$

### 1.2.4 强关联规则
- **定义**:同时满足最小支持度阈值和最小置信度阈值的关联规则称为强关联规则
- **大数据分析注意事项**
    - 大数据分析推论不必然蕴涵因果关系
    - 只是解释了,一种同时出现的概率.

### 1.2.5 频繁项集
- **定义**:其支持度大于或等于min_sup，则称$I_{i}$为频繁项集

- **缺点**: m个项产生非空项集$2^m$,不具备样本可扩展性

### 1.2.6 支持度

提升度表示含有Y的条件下，同时含有X的概率，与X总体发生的概率之比，即:
$$\operatorname{Lift}(X \Leftarrow Y)=P(X | Y) / P(X)=\text { Con fidence }(X \Leftarrow Y) / P(X)$$

- 提升度大于1则 X⇐Y 是有效的强关联规则，
- 提升度小于等于1则 X⇐Y 是无效的强关联规则 。
- 如果X和Y独立，则有 Lift(X⇐Y)=1 ，因为此时 P(X|Y)=P(X)

## 1.3 传统关联规则的挖掘过程

### 1.3.1 基本过程
- 找到强关联规则的常用**判断标准**
    - **最小支持度(包含)**:
        - 表示规则中的所有项在事务数据库D中同时出现的频度应满足的最小频度
    - **最小置信度(排除)**:
        - 表示规则中前件项的出现暗示后件项出现的概率应满足的最小概率。

- 挖掘强关联规则的两个**基本步骤**:
    - **找频繁项集**:
        - 通过用户给定最小支持度阈值min_sup，寻找所有频繁项集，即仅保留大于或等于最小支持度阈值的项集。
    - **生成强关联规则**: 
        - 通过用户给定最小置信度阈值min_conf，在频繁项集中寻找关联规则，即删除不满足最小置信度阈值的规则

### 1.3.2 寻找频繁项集的基本的算法
- 输入： 全局项集I和事务数据库D，最小支持度阈值min_sup。
- 输出： 所有的频繁项集集合L

**方法**
```
n=|D|;
for (I的每个子集c)
{ i=0;
for (对于D中的每个事务t)
{ if (c是t的子集)
i++;
}
if (i/n≥min_sup)
L=L∪{c}; //将c添加到频繁项集集合L中;
}
```

# 2 apriori算法

## 2.1 算法流程

- **输入**：数据集合D，支持度阈值α
- **输出**：最大的频繁k项集
- **算法流程**：

```
　　　　1）扫描整个数据集，得到所有出现过的数据，作为候选频繁1项集（k=1），频繁0项集为空集。

　　　　2）挖掘频繁k项集

　　　　　　a) 扫描数据计算候选频繁k项集的支持度

　　　　　　b) 去除候选频繁k项集中支持度低于阈值的数据集,得到频繁k项集。如果得到的频繁k项集为空，则直接返回频繁k-1项集的集合作为算法结果，算法结束。如果得到的频繁k项集只有一项，则直接返回频繁k项集的集合作为算法结果，算法结束。

　　　　　　c) 基于频繁k项集，连接生成候选频繁k+1项集。

　　　　3） 令k=k+1，转入步骤2
```

**缺点**:从算法的步骤可以看出，Aprior算法每轮迭代都要扫描数据集，因此在数据集很大，数据种类很多的时候，算法效率很低。


## 2.2 代码实现


```python
import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd
store_data = pd.read_csv(r'E:\ai\ai_lab\ai_case\ai_data\ml\store_data.csv', header=None)  
store_data.head() 
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
      <th>16</th>
      <th>17</th>
      <th>18</th>
      <th>19</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>shrimp</td>
      <td>almonds</td>
      <td>avocado</td>
      <td>vegetables mix</td>
      <td>green grapes</td>
      <td>whole weat flour</td>
      <td>yams</td>
      <td>cottage cheese</td>
      <td>energy drink</td>
      <td>tomato juice</td>
      <td>low fat yogurt</td>
      <td>green tea</td>
      <td>honey</td>
      <td>salad</td>
      <td>mineral water</td>
      <td>salmon</td>
      <td>antioxydant juice</td>
      <td>frozen smoothie</td>
      <td>spinach</td>
      <td>olive oil</td>
    </tr>
    <tr>
      <th>1</th>
      <td>burgers</td>
      <td>meatballs</td>
      <td>eggs</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>chutney</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>turkey</td>
      <td>avocado</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>mineral water</td>
      <td>milk</td>
      <td>energy bar</td>
      <td>whole wheat rice</td>
      <td>green tea</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
records = []  
for i in range(0, store_data.shape[0]):  
    records.append([str(store_data.values[i,j]) for j in range(0, store_data.shape[1])])
```


```python
from apyori import apriori
```


```python
"""
def apriori(transactions, **kwargs):
     Arguments:
         transactions：数据集
         min_support：最小支持度
         min_confidence：最小置信度
         min_lift：最小提升度
         max_length：最小项数.
"""
association_rules = apriori(records, min_support=0.005, min_confidence=0.2, min_lift=4, min_length=2)  
association_results = list(association_rules)
print(len(association_results) )
```

    8
    


```python
for item in association_results:
    pair = item[0] 
    items = [x for x in pair]
    # 输出关联
    print("Rule: " + items[0] + " -> " + items[1])
    # 输出支持度
    print("Support: " + str(item[1]))
    # 输出确信度
    print("Confidence: " + str(item[2][0][2]))
    # 输出提升度
    print("Lift: " + str(item[2][0][3]))
    print("=====================================")
```

    Rule: escalope -> pasta
    Support: 0.005865884548726837
    Confidence: 0.3728813559322034
    Lift: 4.700811850163794
    =====================================
    Rule: whole wheat pasta -> olive oil
    Support: 0.007998933475536596
    Confidence: 0.2714932126696833
    Lift: 4.122410097642296
    =====================================
    Rule: pasta -> shrimp
    Support: 0.005065991201173177
    Confidence: 0.3220338983050847
    Lift: 4.506672147735896
    =====================================
    Rule: escalope -> pasta
    Support: 0.005865884548726837
    Confidence: 0.3728813559322034
    Lift: 4.700811850163794
    =====================================
    Rule: ground beef -> herb & pepper
    Support: 0.006399146780429276
    Confidence: 0.3934426229508197
    Lift: 4.004359721511667
    =====================================
    Rule: whole wheat pasta -> olive oil
    Support: 0.007998933475536596
    Confidence: 0.2714932126696833
    Lift: 4.122410097642296
    =====================================
    Rule: pasta -> nan
    Support: 0.005065991201173177
    Confidence: 0.3220338983050847
    Lift: 4.506672147735896
    =====================================
    Rule: ground beef -> herb & pepper
    Support: 0.006399146780429276
    Confidence: 0.3934426229508197
    Lift: 4.004359721511667
    =====================================
    

# 3 FP-growth算法

## 3.1 FP-tree简介
- 最小支持度阈值:2
    - 频繁项集
- 频繁项集
    - 降序排列
- 事务
    - 删除事务中,非频繁项,保留频繁项
    - 事务中, 按照某个频繁项,降序排列
- 构建tree
    - 初始 null
    - 事务1的 tree
    - 事务2来更新tree的权重(出现次数), 补充tree
- 生成树

## 3.2 挖掘FP树
- 条件FP树
- 叶节点:
    - 相同叶子的路径提取出来
        - 考虑修正
        - 考虑最小支持度阈值:2
        - 提取频繁项
    - 其他叶子节点
        - 提取路径
        - 考虑最小支持度阈值:2
        - 提取频繁项
- 频繁项集

## 3.3 代码实现


```python
from pyfpgrowth import *
import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd
store_data = pd.read_csv(r'E:\ai\ai_lab\ai_case\ai_data\ml\store_data.csv', header=None)  
store_data.head() 
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
      <th>16</th>
      <th>17</th>
      <th>18</th>
      <th>19</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>shrimp</td>
      <td>almonds</td>
      <td>avocado</td>
      <td>vegetables mix</td>
      <td>green grapes</td>
      <td>whole weat flour</td>
      <td>yams</td>
      <td>cottage cheese</td>
      <td>energy drink</td>
      <td>tomato juice</td>
      <td>low fat yogurt</td>
      <td>green tea</td>
      <td>honey</td>
      <td>salad</td>
      <td>mineral water</td>
      <td>salmon</td>
      <td>antioxydant juice</td>
      <td>frozen smoothie</td>
      <td>spinach</td>
      <td>olive oil</td>
    </tr>
    <tr>
      <th>1</th>
      <td>burgers</td>
      <td>meatballs</td>
      <td>eggs</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>chutney</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>turkey</td>
      <td>avocado</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>mineral water</td>
      <td>milk</td>
      <td>energy bar</td>
      <td>whole wheat rice</td>
      <td>green tea</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
records = []  
for i in range(0, store_data.shape[0]):  
    records.append([str(store_data.values[i,j]) for j in range(0, store_data.shape[1])])
```

- **参考**:https://github.com/evandempsey/fp-growth/blob/master/pyfpgrowth/pyfpgrowth.py
- **安装**: pip install pyfpgrowth


```python
patterns = pyfpgrowth.find_frequent_patterns(records, 2)
```


```python
rules = pyfpgrowth.generate_association_rules(patterns, 0.7)
```


```python
for i in range(1,5):
    key=list(rules.keys())[i]
    value=rules[key][0]
    confidence=rules[key][1]
    print(" %s -> %s: %f"%(key,value,confidence))
```

     ('frozen vegetables', 'ground beef', 'mineral water', 'nan', 'oil', 'shrimp', 'spaghetti') -> ('low fat yogurt',): 0.789474
     ('avocado', 'cottage cheese', 'green tea', 'honey', 'low fat yogurt', 'salmon', 'spinach', 'tomato juice', 'vegetables mix', 'whole weat flour') -> ('mineral water',): 1.000000
     ('burgers', 'chocolate', 'french fries', 'ham', 'low fat yogurt', 'spaghetti') -> ('eggs', 'nan'): 8.000000
     ('cider', 'french wine', 'milk') -> ('green tea', 'nan'): 5.666667
    


```python

```

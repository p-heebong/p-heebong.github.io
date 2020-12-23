---
title: 예제를 통해 알아보는 PyTorch Geometric 기본개념
date: 2020-03-09 12:06:00
categories: [Deep Learning, Tutorial]
tags: [Deep Learning, Graph Neural Network, PyTorch]
use_math: true
seo:
  date_modified: 2020-03-09 03:25:56 +0900
---



다음 글은 [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric) 라이브러리 설명서에 있는  [Introduction by Example](https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html#) 를 참고하여 작성했습니다.

<br/>

<img src="https://pytorch-geometric.readthedocs.io/en/latest/_static/pyg_logo_text.svg" width="400" height="400">

<br/>

최근 Graph Neural Network에 대해 관심이 많아 공부하던 도중  <kbd>PyTorch Geometric</kbd>라는 라이브러리를 알게되었습니다. 실제 코드를 작성해보지 않으면, 평생 사용할 수 없을 것 같아 해당 라이브러리의 Docs를 번역하여 글을 작성했습니다. 여러 예제를 통해 **PyTorch Geometric 라이브러리에서 등장하는 5가지 기본 개념**에 대해 살펴보겠습니다.

<br/>

- TOC 
{:toc}
<br/>

---

<br/>



## **그래프의 데이터 핸들링**

그래프는 단순히 노드(node 또는 vertex)와 그 노드를 연결하는 간선(edge)을 하나로 모아 놓은 자료 구조 입니다. 그래프를 구성하는 노드와 엣지들을 모아놓은 집합을 각각 $V, E$ 라고 했을 때, 그래프는 $G=(V,E)$ 로 표현할 수 있습니다.

PyTorch Geometric 에서 하나의 그래프는 `torch_geometric.data.Data` 라는 인스턴스로 표현됩니다.  
특히, 이 인스턴스는 다음과 같은 default 속성을 갖고 있습니다.

<br/>

- `data.x` : 노드특징 행렬
  - [num_nodes, num_node_features]
- `data.edge_index` : 그래프의 연결성
  - [2, num_edges]
- `data.edge_attr` : 엣지특징 행렬
  - [num_edges, num_edge_features]
- `data.y` : 학습하고 싶은 대상 (타겟)
  - 그래프 레벨 → [num_nodes, *]
  - 노드 레벨 → [1, *]
- `data.pos` : 노드위치 행렬
  - [num_nodes, num_dimensions]

<br/>

사실 위에 있는 속성들은 필수가 아니라 **옵션**입니다. 즉, 자신이 구성하고 싶은 속성들을 다양하게 모델링할 수 있습니다. 하지만 일반적으로 그래프 데이터는 노드와 엣지로 표현하기 때문에 위의 속성들로 표현하는 것이 적합해 보입니다.

기존의 `torchvision`은 이미지와 타겟으로 구성된 튜플 형태로 데이터를 정의했습니다.  
그와 다르게,  `PyTorch Geometric`은 조금 더 그래프에 직관적인 형태의 데이터 구조를 갖고 있습니다.

<br/>

그럼 Data 클래스를 사용해 그래프 인스턴스를 만들어보겠습니다.

```python
import torch
from torch_geometric.data import Data

edge_index = torch.tensor([[0, 1, 1, 2],
                        [1, 0, 2, 1]], dtype=torch.long)
x = torch.tensor(([-1], [0], [1], dtype=torch.float))

data = Data(x=x, edge_index=edge_index)
>>> Data(edge_index=[2, 4], x=[3, 1])
```
- `edge_index` : (2,4) 크기의 행렬 → 4개의 엣지조합
- `x` : (3,1) 크기의 행렬 → 3개의 노드와 각 노드의 특징은 1개

<br/>

일반적으로 엣지는 노드의 순서쌍으로 나타내는 경우가 많습니다.  
따라서 (v1, v2) 와 같은 자료형 구조가 익숙할 때가 많습니다.  
이런 경우, `contiguous()` 를 사용해 동일한 그래프로 표현할 수 있습니다. 

```python
import torch
from torch_geometric.data import Data

edge_index = torch.tensor([[0, 1],
                        [1, 0],
                        [1, 2],
                        [2, 1]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

data = Data(x=x, edge_index=edge_index.t().contiguous())
>>> Data(edge_index=[2, 4], x=[3, 1])
```

<br/>

<figure>
  <img src="https://pytorch-geometric.readthedocs.io/en/latest/_images/graph.svg" width="400" height="400">
    <figcaption><center>우리가 정의한 data 인스턴스의 실제 그래프</center></figcaption>
</figure>



<br/>

<br/>

<br/>

추가적으로, `torch_geometric.data.Data` 클래스는 다음과 같은 함수를 제공합니다.

- `data.keys` : 해당 속성 이름
- `data.num_nodes` : 노드 총 개수
- `data.num_edges` : 엣지 총 개수
- `data.contains_isolated_nodes()` : 고립 노드 여부 확인
- `data.contains_self_loops()` : 셀프 루프 포함 여부 확인
- `data.is_directed()` : 그래프의 방향성 여부 확인

<br/>

그래프론에서 자주 사용하는 루프, 고립된 노드, 방향성 등 그래프 특징을 반영한 함수들이 있네요.  
소스코드를 뜯어보면, 어떤 알고리즘을 사용했는지까지 알 수 있다는 생각이 듭니다.

<br/>

<br/>

<br/>

## **공통 벤치마크 데이터셋**

PyTorch Geometric은 **다양한 공통 벤치마크 데이터셋**을 포함하고 있습니다.  
해당 데이터셋의 종류는 [torch_geometric.datasets](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html) 에서 확인할 수 있습니다.

각 데이터셋마다 그래프 데이터의 속성이 다르기 때문에 사용되는 함수가 다를 수 있습니다.  
그래프하면 빠질 수 없는 데이터셋인, **ENZYMES** 과 **Cora** 에 대한 예시를 살펴보겠습니다.

<br/>

다음은 **ENZYMES** 데이터셋을 불러오는 예제입니다.

```python
from torch_geometric.datasets import TUDataset

dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
>>> ENZYMES(600)

len(dataset)
>>> 600

dataset.num_classes
>>> 6

dataset.num_node_features
>>> 3
```

- `num_classes` : 그래프의 클래스 수
- `num_node_features` : 노드의 특징 수

<br/>

ENZYMES 데이터셋에는 6종류의 클래스를 가진 600개의 그래프가 있는 것을 확인할 수 있습니다.  
그래프 하나를 가져올 때는 어떻게 할까요?

```python
data = dataset[0]
>>> Data(edge_index=[2, 168], x=[37, 3], y=[1])

data.is_undirected()
>>> True

train_dataset = dataset[:540]
>>> ENZYMES(540)

test_dataset = dataset[540:]
>>> ENZYMES(60)

dataset = dataset.shuffle()
>>> ENZYMES(600)
```

- `슬라이싱`을 통해 하나의 그래프 데이터를 가져올 수 있습니다.
- `edge_index=[2, 168]` → 총 84개의 엣지
- `x=[37, 3]` → 총 37개의 노드와 3개의 노드특성
- `y=[1]` → 그래프 레벨 타겟
- `dataset.shuffle()` → 데이터셋 셔플

<br/>

다음은 **Cora** 데이터셋을 불러오는 예제입니다.  
**Cora** 데이터셋은주로 semi-supervised graph node classification task를 위한 데이터셋으로 사용됩니다.

```python
from torch_geometric.datasets import Planetoid

dataset = Planetoid(root='/tmp/Cora', name='Cora')
>>> Cora()

len(dataset)
>>> 1

dataset.num_classes
>>> 7

dataset.num_node_features
>>> 1433
```

- `Cora()` : 데이터셋 전체가 하나의 그래프
- `num_classes` : 클래스 수 (그래프가 아니라 노드임을 알 수 있음)
- `num_node_features` : 1433개의 노드특성

<br/>

앞에서 봤던 **ENNZYMES** 과 다르게, **Cora** 데이터셋은 조금 다른 속성을 갖고 있습니다.  
주로 (준지도학습) 노드예측 task에 사용되기 때문에 추가적인 속성이 존재하는 것을 볼 수 있습니다.

```python
data = dataset[0]
>>> Data(edge_index=[2, 10556], test_mask=[2708],
         train_mask=[2708], val_mask=[2708], x=[2708, 1433], y=[2708])

data.is_undirected()
>>> True

data.train_mask.sum().item()
>>> 140

data.val_mask.sum().item()
>>> 500

data.test_mask.sum().item()
>>> 1000
```

- `data = dataset[0]` : `slicing` 을 통해 그래프가 아닌 노드 하나를 가져옵니다.
- `train_mask` : 학습하기 위해 사용하는 노드들을 가리킴
- `val_mask` : 검증 시 사용하는 노드들을 가리킴
- `test_mask` : 테스트 시 사용하는 노드들을 가리킴

<br/>

<br/>

<br/>

## **미니배치**

많은 뉴럴 네트워크들이 배치 단위로 학습하듯이, **Pytorch Geometric**도 sparse block diagonal adjacency matrices를 만들어 미니배치를 통해 병렬화처리를 수행합니다. 

기존 `torch`에서는 `torch.utils.data.DataLoader`를 통해 배치 단위로 데이터를 처리했습니다.  
`torch_geometric` 에서는 `torch_geometric.data.DataLoader`를 통해 그래프 단위 데이터를 처리하게 됩니다.

```pyhton
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader

dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES', use_node_attr=True)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch in loader:
    batch
    >>> Batch(batch=[1082], edge_index=[2, 4066], x=[1082, 21], y=[32])

    batch.num_graphs
    >>> 32
```

- `DataLoader` : 앞에서 봤던 `torch_geometric.data.Data` 클



## **데이터 변환**

역시, 데이터를 전처리하기 위해 사용하는 함수도 있습니다. 우리가 잘 아는 `torchvision`에서는 `torchvision.transforms.Compose`를 통해 여러 이미지 전처리 함수들을 결합해 사용합니다.

이와 비슷하게 Pytorch Gemotric도 `Data` 객체를 `



다음은 ShapeNet dataset를 활용해 transforms을 적용한 예제입니다.

```python
from torch_geometric.datasets import ShapeNet

dataset = ShapeNet(root='/tmp/ShapeNet', categories=['Airplane'])

dataset[0]
>>> Data(pos=[2518, 3], y=[2518])
```

- ShapeNet은 17000건의 3D형태의 점구름(point clouds) 데이터입니다. 총 16개의 카테고리로 구성되어 있습니다.
- `pos=[2518, 3]` : 2518개의 점데이터와 3차원임을 나타냅니다.



```python
import torch_geometric.transforms as T
from torch_geometric.datasets import ShapeNet

dataset = ShapeNet(root='/tmp/ShapeNet', categories=['Airplane'],
                    pre_transform=T.KNNGraph(k=6),
                    transform=T.RandomTranslate(0.01))

dataset[0]
>>> Data(edge_index=[2, 15108], pos=[2518, 3], y=[2518])
```

- `pre_transform = T.KNNGraph(k=6)` : KNN을 통해 데이터를 그래프 형태로 변형합니다.
  - 결과값으로 `edge_index`가 추가된 것을 확인할 수 있습니다. (즉, 연결상태 생성)
- `transform = T.RandomTranslate(0.01)` : 각 노드의 위치를 







## 그래프로 학습하기

앞에서 다음과 같은 내용을 배웠습니다.

- 그래프 데이터 핸들링하기
- `dataset`, `dataloader` 인스턴스 생성하기
- `transforms` 를 사용해 데이터를 변환하기



이제 Graph Neural Network을 활용해 분류 문제를 해결해보겠습니다.  
다음은 간단한 **GCN layer**를 구성한 뒤,  **Cora** 데이터셋에 적용하는 예제입니다.

```python
from torch_geometric.datasets import Planetoid

dataset = Planetoid(root='/tmp/Cora', name='Cora')
>>> Cora()
```

- Cora 데이터셋은 2708개의 "scientific publications"으로 구성된 데이터입니다.
- 하나의 논문은 여러 논문들을 인용할 수 있는데, 이를 연결한 네트워크가 바로 citation network 입니다.
- citation network을 하나의 그래프로 본다면, 각 논문은 노드로 볼 수 있고 인용 관계가는 엣지가 됩니다.
- 또한, 논문에서 등장하는 1433개의 특정단어들을 모아 하나의 단어사전으로 만들고, 각 논문마다 단어의 등장 여부를 feature vector로 만들어줌으로써 노드의 특징을 반영할 수 있게 됩니다.



2개의 `GCNConv` layer를 사용합니다. 

```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)
```



아래 과정부터는 기존 `pytorch`와 상당히 유사합니다.

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
```

- 이미 정의되어 있는 `train_mask` 를 사용해 학습 데이터를 구분합니다.
- `dataloader` 를 정의할 때, `train_mask` 를 사용해서 구현할 수도 있습니다.

<br/>

```python
model.eval()
_, pred = model(data).max(dim=1)
correct = float (pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
acc = correct / data.test_mask.sum().item()
print('Accuracy: {:.4f}'.format(acc))
>>> Accuracy: 0.8150
```

- 마찬가지로, `test_mask` 를 사용해 평가 데이터를 구분합니다.

<br/>

---

<br/>

GNN에 대한 연구는 꽤 진척이 있는 상태고, 다양한 분야에도 적용한 사례들이 많습니다.  
예전에 간단한 GCN 구조를 `pytorch` 로 구현하면서, 힘들었던 기억이 있었는데요.  
`torch_geometric` 을 사용한다면, 확실히 좀 더 쉽지 않을까 생각됩니다.



제가 생각한  `torch geometric` 의 강점은 크게 2가지 입니다.

1. `torch_geometric.nn` 을 통해 다양한 레이어들을 구성할 수 있다.
   - 크게는 Convoluional Layer와 Pooling Layer로 구성
   - 최신 GNN 논문들이 빠르게 반영됨
2. `torch_geometric.datasets` 을 통해 다양한 벤치마크 데이터셋을 쉽게 이용할 수 있다.
   - 각 그래프 데이터셋마다 특징이 다른 것을 잘 구현함
   - 이는 `torch_geometric.data` 와 연관되어 있어 그래프 데이터를 빠르게 살펴볼 수 있음






















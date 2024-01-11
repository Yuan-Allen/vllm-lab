# Report

## vllm调度过程分析

vllm的调度单位是`SequenceGroup`。vllm的scheduler维持了三个队列，分别为`waiting`队列，`running`队列和`swapped`队列。
- 一个`SequenceGroup`包含了一组由相同prompt生成的`Sequence`。
- `waiting`队列包含了还没开始运行，只有prompt的`SequenceGroup`。
- `running`队列包含了这一轮迭代要用来作为输入的`SequenceGroup`。
- `swapped`队列包含了由于GPU显存资源不足，被暂时swap到CPU memory的`SequenceGroup`。

调度过程：
- vllm首先判断`swapped`是否为空，如果为空，则尽量（例如不超过配置的`max_num_seqs`）将`waiting`队列中的`SequenceGroup`加入到`running`队列中。在遍历`waiting`队列的过程中，若由于超出`max_num_seqs`等原因加入失败会`break`遍历，进入下一阶段。
- 将`running`队列按照调度算法提供的优先级排序，然后遍历`running`队列，查看当前是否有足够的GPU block（vllm中的一种memory资源抽象）给该`SequenceGroup`使用，若没有则将队列的最后一个`SequenceGroup`抢占掉。
    - 抢占有两种方式可以使用，`recompute`和`swap`。`recompute`会将该`SequenceGroup`生成的`Sequence`直接free掉，然后重新加到`waiting`队列的最前面；`swap`则会把该`SequenceGroup`移到`swapped`队列。
- 如果这轮调度没有发生抢占，vllm最后会对`swapped`队列按照调度算法提供的优先级进行排序，然后按顺序尽量把`swapped`队列中的`SequenceGroup`加入到`running`队列。若加入失败则会`break`遍历。

简而言之，vllm的调度就是把`waiting`队列和`swapped`队列中的`SequenceGroup`尽量加入到`running`队列中，其中`running`队列占用的资源太多会触发抢占，将队列尾部的`SequenceGroup`移到`swap`队列（对应也会swap到CPU memory），或者抛弃生成内容，重新加入到`waiting`队列中。

vllm调度和传统CPU调度的一个很大不同就是，CPU调度是由于计算资源有限，而vllm调度是由于memory资源有限。当memory资源不足时，`running`队列就会考虑把排序后队列里面最后的`SequenceGroup`进行抢占。除此之外，`running`队列也要遵守一些相关的配置，例如在里面同时执行的`Sequence`数量不能超过指定的`max_num_seqs`（默认是256）等。


## Policy

### 机制

vllm中的调度是基于优先级的，即添加一种新的调度算法，只需阐明这种调度算法中每个元素的优先级是如何计算的，然后在调度时会根据对应获取的优先级对元素进行排序，具体如下。其中`get_priority`就是由指定的调度策略提供。
```python
class Policy:

    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
    ) -> float:
        raise NotImplementedError

    def sort_by_priority(
        self,
        now: float,
        seq_groups: Deque[SequenceGroup],
    ) -> Deque[SequenceGroup]:
        return deque(
            sorted(
                seq_groups,
                key=lambda seq_group: self.get_priority(now, seq_group),
                reverse=True,
            ))
```

### FIFO

`FIFO`（vllm中称FCFS，即first-come-first-server）是vllm中默认自带的调度策略。该调度算法使用`now - seq_group.arrival_time`计算优先级，即任务到达时间越早，优先级越高。

```python
class FCFS(Policy):

    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
    ) -> float:
        return now - seq_group.arrival_time
```

### STATIC

我们为vllm添加了静态优先级的调度方式，代码中称作`STATIC`。在这种调度方式下，用户可以通过在`SamplingParams`设定参数来手动指定任务的优先级。

```python
# In sampling_params.py
    def __init__(
        self,
        # -- snip --
        priority: Optional[float] = None,
    ) -> None:
    # -- snip --
```

```python
# In policy.py
class STATIC(Policy):

    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
    ) -> float:
        if seq_group.sampling_params.priority is None:
            return float('-inf')
        return seq_group.sampling_params.priority
```

### SJF

我们还为vllm添加了SJF（Shortest-Job-First）调度策略，即短任务优先。在该调度策略下，`max_tokens`越小，任务优先级越高。

```python
class SJF(Policy):

    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
    ) -> float:
        return -seq_group.sampling_params.max_tokens
```

### LDF

我们还为vllm添加了一种自己设计的调度策略，即LDF（Largest-Data-First），数据量最大的任务（`SequenceGroup`）优先。该调度策略会统计`SequenceGroup`中所有`Sequence`的数据大小，数据量越大则优先级越高。

```python
# Largest Data First
class LDF(Policy):
    
        def get_priority(
            self,
            now: float,
            seq_group: SequenceGroup,
        ) -> float:
            data_size = 0
            for _, seq in seq_group.seqs_dict.items():
                data_size += len(seq.data.prompt_token_ids) + len(seq.data.output_token_ids)
            return data_size
```

### LCFS

与vllm自带的`FCFS`相对应，我们为vllm添加了`LCFS`调度算法，即到达时间越晚，优先级越高。

```python
class LCFS(Policy):

    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
    ) -> float:
        return seq_group.arrival_time - now
```

## Benchmark

### 测试用例([`benchmarks/benchmark_lab.py`](benchmarks/benchmark_lab.py))

测试命令如下：

```shell
py benchmark.py --tokenizer facebook/opt-125m --request-rate $para1 --dataset $para2 --num-prompts $para3
```

本次测试选用模型为`facebook/opt-125m`，以`num-prompts`参数指定总请求数量，以`dataset`限定输入输出的句子长度————以所传参数为中心的泊松分布，以`request-rate`指定请求频率————两次请求间隔时长为以所传参数为中心的泊松分布。

通过调整测试用例输入参数，模拟不同用户请求场景：

| 模拟场景   | 请求频率(r/s) | 句子长度 | 请求量 |
|------------|---------------|----------|--------|
| 密集短对话 | 1000          | 20       | 1000   |
| 密集长对话 | 1000          | 80       | 1000   |
| 稀松短对话 | 100           | 20       | 100    |
| 稀松长对话 | 100           | 80       | 100    |
通过更改vllm所选用的调度算法，对比不同调度算法在不同场景下的性能。

所有用例**测量三次取平均值**。

### 测试结果

#### 各调度算法性能

| 调度算法 | 模拟场景   | 请求频率(r/s) | 句子长度 | 请求量 | 总执行时间(s) | 吞吐量(r/s) | 平均请求时延(s) |
|----------|------------|---------------|----------|--------|---------------|-------------|-----------------|
| FIFO     | 密集短对话 | 1000          | 20       | 1000   | 5.98          | 167.23      | 3.25            |
|          | 密集长对话 | 1000          | 80       | 1000   | 20.87         | 47.92       | 11.64           |
|          | 稀松短对话 | 100           | 20       | 100    | 1.50          | 66.74       | 0.33            |
|          | 稀松长对话 | 100           | 80       | 100    | 2.70          | 37.02       | 0.78            |
| STATIC   | 密集短对话 | 1000          | 20       | 1000   | 6.15          | 162.57      | 3.40            |
|          | 密集长对话 | 1000          | 80       | 1000   | 21.79         | 45.89       | 12.60           |
|          | 稀松短对话 | 100           | 20       | 100    | 1.79          | 55.95       | 0.59            |
|          | 稀松长对话 | 100           | 80       | 100    | 2.70          | 37.08       | 0.74            |
| SJF      | 密集短对话 | 1000          | 20       | 1000   | 6.30          | 158.63      | 3.23            |
|          | 密集长对话 | 1000          | 80       | 1000   | 22.74         | 43.97       | 12.96           |
|          | 稀松短对话 | 100           | 20       | 100    | 1.51          | 66.40       | 0.34            |
|          | 稀松长对话 | 100           | 80       | 100    | 2.73          | 36.61       | 0.73            |
| LDF      | 密集短对话 | 1000          | 20       | 1000   | 6.81          | 146.89      | 3.91            |
|          | 密集长对话 | 1000          | 80       | 1000   | 22.85         | 43.76       | 13.29           |
|          | 稀松短对话 | 100           | 20       | 100    | 1.57          | 63.52       | 0.38            |
|          | 稀松长对话 | 100           | 80       | 100    | 2.66          | 37.54       | 0.72            |
| LCFS     | 密集短对话 | 1000          | 20       | 1000   | 6.03          | 165.85      | 3.40            |
|          | 密集长对话 | 1000          | 80       | 1000   | 24.17         | 43.37       | 13.92           |
|          | 稀松短对话 | 100           | 20       | 100    | 1.56          | 64.07       | 0.37            |
|          | 稀松长对话 | 100           | 80       | 100    | 2.85          | 35.13       | 0.92            |

#### 各场景调度算法对比

| 模拟场景   | 调度算法 | 总执行时间(s) | 吞吐量(r/s) | 平均请求时延(s) |
|------------|----------|---------------|-------------|-----------------|
| 密集短对话 | FIFO     | 5.98          | 167.23      | 3.25            |
|            | STATIC   | 6.15          | 162.57      | 3.40            |
|            | SJF      | 6.30          | 158.63      | 3.23            |
|            | LDF      | 6.81          | 146.89      | 3.91            |
|            | LCFS     | 6.03          | 165.85      | 3.40            |
| 密集长对话 | FIFO     | 20.87         | 47.92       | 11.64           |
|            | STATIC   | 21.79         | 45.89       | 12.60           |
|            | SJF      | 22.74         | 43.97       | 12.96           |
|            | LDF      | 22.85         | 43.76       | 13.29           |
|            | LCFS     | 24.17         | 43.37       | 13.92           |
| 稀松短对话 | FIFO     | 1.50          | 66.74       | 0.33            |
|            | STATIC   | 1.79          | 55.95       | 0.59            |
|            | SJF      | 1.51          | 66.40       | 0.34            |
|            | LDF      | 1.57          | 63.52       | 0.38            |
|            | LCFS     | 1.56          | 64.07       | 0.37            |
| 稀松长对话 | FIFO     | 2.70          | 37.02       | 0.78            |
|            | STATIC   | 2.70          | 37.08       | 0.74            |
|            | SJF      | 2.73          | 36.61       | 0.73            |
|            | LDF      | 2.66          | 37.54       | 0.72            |
|            | LCFS     | 2.85          | 35.13       | 0.92            |

#### 结果分析

通过同一调度不同场景的对比与分析：
- 随着输入输出句子长度的增加，请求负载增加，网络需要处理的token增加，吞吐量下降，请求时延增加。
- 随着请求频率的的增加，server端单位时间内处理请求数量增加，吞吐量增加，直至达到瓶颈；等待时间增加，时延变长。

不同调度算法的对比与分析：

在`STATIC`策略中，由于优先级是由用户手动配置的，因此在测试中的测试结果不具备参考意义。提供该调度策略的作用更多是给用户提供了更多可配置的空间，以更好的适配特殊场景。

在**密集短对话**和**密集长对话**场景下，`FIFO`的性能最优，这也许是由于vllm针对`FIFO`策略特别做了优化，即由于新到来的请求会加入到`waiting`队列的尾部，而`recompute`的请求会加入到`waiting`队列的头部，因此`waiting`队列天生就是按照`FIFO`有序的，原作者就省去了对`waiting`队列的排序。在我们的实现中也针对`FIFO`保留了该优化。在该场景下，该优化带来的提升显得更为可观。

在**稀松短对话**和**稀疏长对话**的场景下，几种调度算法的测试结果都很接近，可能是在请求频率较低的情况下，不同调度算法带来的差别可以忽略不计。

其实总而言之，在我们的测试场景下，几种调度算法带来的影响其实都微乎其微，其测试所得结果都差不多。由于我们运行的模型较为轻量（`facebook/opt-125m`），即使我们将请求频率和句子长度调高，由于`max_num_seqs`的限制，也远远触及不到GPU的显存瓶颈。在这种情况下，数据和网络的通信开销波动甚至是泊松分布带来的波动可能都已经覆盖了这调度算法之间带来的性能差别。

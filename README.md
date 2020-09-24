# paperspace-project
Skeleton project using paperspace

```
version: 2

workflows:
  single-node:
    steps:
      -
        name: "single-node"
        command: experiment.run_single_node
        params:
          command: python train.py
          container: tensorflow/tensorflow:1.13.1-gpu-py3
          machineType: "C3"


version: 1
type: "singlenode"
worker:
  container: "tensorflow/tensorflow:1.13.1-py3"
  command: "pip install -r requirements.txt && python train.py"
  machine-type: "C3"
model:
  type: Sklearn
  path: '/artifacts'
```
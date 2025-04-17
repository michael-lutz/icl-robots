# icl-robots

In context learning exploration for robotic locomotion.

## Installation

Install from PyPI:

```bash
pip install iclrobot
```

## Example Usage

To start training the humanoid walking task, run:
```bash
python -m iclrobot.dh_walking
```

You should be able to directly click the link to visualize the tensorboard logs.
Otherwise, you migth have to manually run:

```bash
tensorboard --logdir=<path to iclrobot>/dh_walking/logs
```


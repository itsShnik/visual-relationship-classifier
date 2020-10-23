# visual-relationship-classifier
Classifier to identify visual relationships in an image. Furthermore, I have added the code to obtain visual relationships for the VQAv2 dataset. These relationships can be used with models like VLBERT (as in this repo) to improve visual question answering.

## Architecture of Visual Relationship Classifier

![Architecture](figs/vis_rel_architecture.png?raw=true)

This architecture is taken from Yao et al. ECCV 2018.

## Setup

To install all the required libraries, execute the following command.

```
pip install -r requirements.txt
```

Install the Visual Genome dataset images, objects and relationships from [here](https://visualgenome.org/api/v0/api_home.html). Put them in a single folder. The labels in the visual genome dataset are extremely noisy. Use the notebook ``relationships.ipynb`` to clean and preprocess the data. This will create four files in the ``data/`` folder.

The ``data/`` folder should have the following structure now.

```
data/
	|
	|----images/
	|----relationships_train.json
	|----relationships_val.json
	|----relationships_test.json
	|----relationship_classes.json
```

With the ``images/`` subfolder containing all the visual genome images at one place.

## Training

To train the model on 4 GPUs in a distributed data parallel training setting, run

```
./scripts/dist_run_single.sh 4 vis_rel/train_end2end.py cfgs/vis_rel/frcnn.yml output
```

The checkpoints will be saved in the ``outputs/`` directory.

You can explore other training options such as single gpu training or data parallel training in the scripts directory.

## Creating Visual Relationships for VQAv2 dataset

Setup the data for VQAv2 in the data folder and according to the [VL-BERT](https://github.com/jackroos/VL-BERT) repo directions. Now, just run the following command

```
python createVisRel/createVisRel.py
```

and the visual relationships will be saved in a pickle file under the ``data/`` folder.

## Wandb

The experiments that I have conducted can be found on this [wandb link](https://wandb.ai/shnik/visual-relationships).

## Acknowledgement

A lot of thanks to @jackroos for his repository [VL-BERT](https://github.com/jackroos/VL-BERT). It helped me a lot to develop the codebase.

# pokedex-resnet50-keras

It's a resnet50 implementation using keras (TF as backend) to classify pokemons.

# Results
![]()
![]()
![]()
![]()
![]()

# Usage
Download the dataset [Dataset](https://drive.google.com/open?id=1Y-h94ZxPN79ILOtwEjt3ZBmGT9769PyX)

### Train
`python3 train.py --dataset dataset --model pokedex-keras/pokedex.model --le pokedex-keras/le.pickle -e 50`

##### Parameters:
```
> python3 pokedex-keras/train.py --help
usage: train.py [-h] -d DATASET [-m MODEL] [-l LE] [-p PLOT] [-r LEARN_RATE]
                [-b BATCH_SIZE] [-e EPOCHS]

optional arguments:
  -h, --help            show this help message and exit
  -d DATASET, --dataset DATASET
                        path to input dataset
  -m MODEL, --model MODEL
                        path to trained model
  -l LE, --le LE        path to label encoder
  -r LEARN_RATE, --learn_rate LEARN_RATE
                        train learn rate
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        how much images will be handdled by epoch
  -e EPOCHS, --epochs EPOCHS
                        how much epochs your n_network will train
```

### Predict
 `python3 predict.py --image path/to/image.png --model trained.model --le le.pickle`

##### Parameters
```
> python3 predict.py -h
usage: predict.py [-h] [-m MODEL] [-l LE] -i IMAGE

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        path to trained model
  -l LE, --le LE        path to label encoder
  -i IMAGE, --image IMAGE
                        image to predict

```

#### References


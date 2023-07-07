# AdvXL-Adversarial-Cross-Laser-Attack
Code for Adversarial Cross Laser Attack(AdvXL)

## Requirements
see requirements.txt


## To evaluate physical experimental results:
```bash
cd physcialExp
python run.py physcialExperiment
python run.py indoorMul
python run.py outdoorMulBlue
python run.py outdoorMulRed
```
## To execute digital experiment:
```bash
cd digitalExp
python run.py vgg19 10 10 20
```
`vgg19`can be replaced by `inception_v3` or `densenet121`

The three parameters after mean **the number of candidate points**, **initialization number** and **fitting number**

for the budget of 300 queries can be : `30 20 250` or `50 50 200` or something else

The more candidate points and initialization times there are, the lower limit of the average query times is, but the efficiency of finding the best advantage is higher during the subsequent fitting process.

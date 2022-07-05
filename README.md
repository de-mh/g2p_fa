# g2p_fa

A Grapheme to Phoneme model using LSTM implemented in pytorch

## Installation

`pip install g2p_fa`

## Usage:

```

>>> from g2p_fa import G2P_Fa
>>> g2p = G2P_Fa()
>>> g2p('سلام')
'sælɒːm'
>>> g2p('طلا')
'tʰælɒː'
>>> g2p('تلاش')
'tʰælɒːʃ'

```
## Training

Create a csv file with Persian text at the first column and IPA in second colmn. for example:

`ابتکار,ʔebtʰekʰɒːɾ`

Then create an instance of 'G2P_Fa' wihout loading checkpoint:
```

>>> from g2p_fa import G2P_Fa
>>> g2p = G2P_Fa(checkpoint=None)

```
And then train the model with csv file:
```
>>> g2p.train('data.csv',epoch=20)
len train: 18968, len valid: 4743
initial loss: 3.5005286693573
Epoch 1 / 20    Train Loss: 3.264, Valid loss: 2.996
Epoch 2 / 20    Train Loss: 2.937, Valid loss: 2.898
Epoch 3 / 20    Train Loss: 2.851, Valid loss: 2.828
Epoch 4 / 20    Train Loss: 2.768, Valid loss: 2.790
Epoch 5 / 20    Train Loss: 2.664, Valid loss: 2.836
Epoch 6 / 20    Train Loss: 2.579, Valid loss: 2.855
Epoch 7 / 20    Train Loss: 2.573, Valid loss: 2.820
Epoch 8 / 20    Train Loss: 2.510, Valid loss: 2.865
Epoch 9 / 20    Train Loss: 2.491, Valid loss: 2.849
Epoch 10 / 20   Train Loss: 2.417, Valid loss: 2.837
Epoch 11 / 20   Train Loss: 2.421, Valid loss: 2.817
Epoch 12 / 20   Train Loss: 2.370, Valid loss: 2.884
Epoch 13 / 20   Train Loss: 2.350, Valid loss: 2.872
Epoch 14 / 20   Train Loss: 2.318, Valid loss: 2.797
Epoch 15 / 20   Train Loss: 2.317, Valid loss: 2.653
Epoch 16 / 20   Train Loss: 2.316, Valid loss: 2.634
Epoch 17 / 20   Train Loss: 2.292, Valid loss: 2.629
Epoch 18 / 20   Train Loss: 2.215, Valid loss: 2.709
Epoch 19 / 20   Train Loss: 2.208, Valid loss: 2.581
Epoch 20 / 20   Train Loss: 2.182, Valid loss: 2.568
```
Then you can save the model:
```
>>> g2p.save('SAVE_PATH')
```
For using your saved model you have to pass the checkpoint
```
>>> g2p = G2P_Fa(checkpoint='SAVE_PATH')
```
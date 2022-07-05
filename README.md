# g2p_fa
A Grapheme to Phoneme model using LSTM implemented in pytorch
## usage:
```

>>> from g2p_fa import G2P_Fa
>>> g2p = G2P_Fa()
>>> g2p('طلا')
'tʰælɒː'
>>> g2p('تلاش')
'tʰælɒːʃ'

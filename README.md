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

## Predicting the price of your cell phone with a Neural Network! 
#### Have you ever wondered if you are paying too much for your cell phone? Let's train a model using multi-class classification to find out! 

#### Notes
- Dataset is taken from [Kaggle](https://www.kaggle.com/datasets/atefehmirnaseri/cell-phone-price)
- This problem is a multi-class classification problem
- We will be building a Nerual Network with tensorflow
- This Neural Network will use the softmax algo and logistic loss function


```python
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

import statistics
```


```python
#Understand the data set 
df = pd.read_csv('CellPhone_train.csv')

Y_train = np.array(df.pop('price_range'))
X_train = np.array(df)
X_features = df.columns.values

m,n = X_train.shape

print(f"X_train.shape: {X_train.shape} Y_train.shape: {Y_train.shape} \n")
print(f"There are {m} training set and {n} features in the dataset, the {n} features are: ")
for i in range(n):
    print(f"{i+1}: {X_features[i]}")

print(f"\n There are {len(np.unique(Y_train))} distinct output values or classes")

```

    X_train.shape: (2000, 20) Y_train.shape: (2000,) 
    
    There are 2000 training set and 20 features in the dataset, the 20 features are: 
    1: battery_power
    2: blue
    3: clock_speed
    4: dual_sim
    5: fc
    6: four_g
    7: int_memory
    8: m_dep
    9: mobile_wt
    10: n_cores
    11: pc
    12: px_height
    13: px_width
    14: ram
    15: sc_h
    16: sc_w
    17: talk_time
    18: three_g
    19: touch_screen
    20: wifi
    
     There are 4 distinct output values or classes



```python
#building the neural network
model = Sequential([
    tf.keras.Input(shape=(20,)),
    Dense(units=18, activation='relu'),
    Dense(units=4, activation='linear'),
    ])
model.compile(loss=SparseCategoricalCrossentropy(from_logits=True), optimizer=Adam(learning_rate=1e-2))
model.summary()
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "sequential"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape           </span>┃<span style="font-weight: bold">       Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ dense (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">18</span>)             │           <span style="color: #00af00; text-decoration-color: #00af00">378</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>)              │            <span style="color: #00af00; text-decoration-color: #00af00">76</span> │
└─────────────────────────────────┴────────────────────────┴───────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">454</span> (1.77 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">454</span> (1.77 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>



#### Structure
The Nerual Network will have ***3*** layers
- First Layer: Input layer and will take the shape of the input parameter which has **20** different features
    > will have shape $(m, 20)$ where $m$ is the number of training set.
- Second Layer: Hidden layer and will have **18** neurons. Total # of params is $20*18(w) + 18(b) = 378$
    > will have output of shape $(m, 18)$ where $m$ is the number of training set.
- Output Layer: Since it is a classification problem with 4 outputs, it will have **4** neurons. Total # of params is $18*4(w) + 4 = 76$
    > will have output shape $(m, 4)$ where $m$ is the number of training set.


```python
#train the model
model.fit(X_train, Y_train, epochs=500)
```

    Epoch 1/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 253us/step - loss: 189.6533
    Epoch 2/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 204us/step - loss: 7.9861
    Epoch 3/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 202us/step - loss: 3.5992
    Epoch 4/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 199us/step - loss: 3.7222
    Epoch 5/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 205us/step - loss: 7.6736
    Epoch 6/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 196us/step - loss: 5.6260
    Epoch 7/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 208us/step - loss: 3.4080
    Epoch 8/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 201us/step - loss: 2.3414
    Epoch 9/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 200us/step - loss: 2.1996
    Epoch 10/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 198us/step - loss: 3.7292
    Epoch 11/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 191us/step - loss: 2.1246
    Epoch 12/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 193us/step - loss: 1.9319
    Epoch 13/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 196us/step - loss: 4.0779
    Epoch 14/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 191us/step - loss: 2.0092
    Epoch 15/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 197us/step - loss: 1.7011
    Epoch 16/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 198us/step - loss: 4.0670
    Epoch 17/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 546us/step - loss: 2.4764
    Epoch 18/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 201us/step - loss: 2.8017
    Epoch 19/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 200us/step - loss: 2.4078
    Epoch 20/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 198us/step - loss: 1.3441
    Epoch 21/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 208us/step - loss: 3.8642
    Epoch 22/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 199us/step - loss: 1.5220
    Epoch 23/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 202us/step - loss: 2.2716
    Epoch 24/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 202us/step - loss: 2.0383
    Epoch 25/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 198us/step - loss: 1.4918
    Epoch 26/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 197us/step - loss: 2.0677
    Epoch 27/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 198us/step - loss: 1.4634
    Epoch 28/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 550us/step - loss: 1.3120
    Epoch 29/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 205us/step - loss: 4.0195
    Epoch 30/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 201us/step - loss: 1.7691
    Epoch 31/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 199us/step - loss: 1.6790
    Epoch 32/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 200us/step - loss: 1.1663
    Epoch 33/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 203us/step - loss: 1.3594
    Epoch 34/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 211us/step - loss: 0.9688
    Epoch 35/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 202us/step - loss: 1.5361
    Epoch 36/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 220us/step - loss: 1.3327
    Epoch 37/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 206us/step - loss: 0.9953
    Epoch 38/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 206us/step - loss: 1.4770
    Epoch 39/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 207us/step - loss: 1.0912
    Epoch 40/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 200us/step - loss: 1.4668
    Epoch 41/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 206us/step - loss: 2.1158
    Epoch 42/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 203us/step - loss: 0.7761
    Epoch 43/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 201us/step - loss: 0.7738
    Epoch 44/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 200us/step - loss: 0.8397
    Epoch 45/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 205us/step - loss: 0.9156
    Epoch 46/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 197us/step - loss: 1.0238
    Epoch 47/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 267us/step - loss: 0.7469
    Epoch 48/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 206us/step - loss: 0.9825
    Epoch 49/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 205us/step - loss: 0.7887
    Epoch 50/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 207us/step - loss: 0.8930
    Epoch 51/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 211us/step - loss: 0.9678
    Epoch 52/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 217us/step - loss: 0.6745
    Epoch 53/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 211us/step - loss: 0.9740
    Epoch 54/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 221us/step - loss: 0.7083
    Epoch 55/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 207us/step - loss: 0.8340
    Epoch 56/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 215us/step - loss: 0.6616
    Epoch 57/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 208us/step - loss: 0.7413
    Epoch 58/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 214us/step - loss: 0.7375
    Epoch 59/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 210us/step - loss: 0.7953
    Epoch 60/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 212us/step - loss: 0.5767
    Epoch 61/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 212us/step - loss: 0.5455
    Epoch 62/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 564us/step - loss: 0.6182
    Epoch 63/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 217us/step - loss: 0.6207
    Epoch 64/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 207us/step - loss: 0.5084
    Epoch 65/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 208us/step - loss: 0.5960
    Epoch 66/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 211us/step - loss: 0.4963
    Epoch 67/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 212us/step - loss: 0.5358
    Epoch 68/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 215us/step - loss: 0.5112
    Epoch 69/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 206us/step - loss: 0.5088
    Epoch 70/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 211us/step - loss: 0.4942
    Epoch 71/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 213us/step - loss: 0.5156
    Epoch 72/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 217us/step - loss: 0.5428
    Epoch 73/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 206us/step - loss: 0.5095
    Epoch 74/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 216us/step - loss: 0.4928
    Epoch 75/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 587us/step - loss: 0.4930
    Epoch 76/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 215us/step - loss: 0.5349
    Epoch 77/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 211us/step - loss: 0.4917
    Epoch 78/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 221us/step - loss: 0.5185
    Epoch 79/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 213us/step - loss: 0.4520
    Epoch 80/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 224us/step - loss: 0.4941
    Epoch 81/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 211us/step - loss: 0.5062
    Epoch 82/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 218us/step - loss: 0.4670
    Epoch 83/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 264us/step - loss: 0.4427
    Epoch 84/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 212us/step - loss: 0.5052
    Epoch 85/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 211us/step - loss: 0.4578
    Epoch 86/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 277us/step - loss: 0.4710
    Epoch 87/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 212us/step - loss: 0.4570
    Epoch 88/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 209us/step - loss: 0.4996
    Epoch 89/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 211us/step - loss: 0.4766
    Epoch 90/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 213us/step - loss: 0.4677
    Epoch 91/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 210us/step - loss: 0.4552
    Epoch 92/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 214us/step - loss: 0.4382
    Epoch 93/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 210us/step - loss: 0.4593
    Epoch 94/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 214us/step - loss: 0.4521
    Epoch 95/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 209us/step - loss: 0.4484
    Epoch 96/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 573us/step - loss: 0.4379
    Epoch 97/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 210us/step - loss: 0.4437
    Epoch 98/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 215us/step - loss: 0.4601
    Epoch 99/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 213us/step - loss: 0.4500
    Epoch 100/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 208us/step - loss: 0.5043
    Epoch 101/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 213us/step - loss: 0.4565
    Epoch 102/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 207us/step - loss: 0.4399
    Epoch 103/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 214us/step - loss: 0.4544
    Epoch 104/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 206us/step - loss: 0.4218
    Epoch 105/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 207us/step - loss: 0.4119
    Epoch 106/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 290us/step - loss: 0.4401
    Epoch 107/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 215us/step - loss: 0.4432
    Epoch 108/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 220us/step - loss: 0.4513
    Epoch 109/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 216us/step - loss: 0.4504
    Epoch 110/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 210us/step - loss: 0.4773
    Epoch 111/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 211us/step - loss: 0.4419
    Epoch 112/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 215us/step - loss: 0.4285
    Epoch 113/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 214us/step - loss: 0.4505
    Epoch 114/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 214us/step - loss: 0.4787
    Epoch 115/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 390us/step - loss: 0.4535
    Epoch 116/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 226us/step - loss: 0.4933
    Epoch 117/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 209us/step - loss: 0.4573
    Epoch 118/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 210us/step - loss: 0.4371
    Epoch 119/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 220us/step - loss: 0.4549
    Epoch 120/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 206us/step - loss: 0.4328
    Epoch 121/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 213us/step - loss: 0.4111
    Epoch 122/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 200us/step - loss: 0.4232
    Epoch 123/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 568us/step - loss: 0.4353
    Epoch 124/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 193us/step - loss: 0.4296
    Epoch 125/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 202us/step - loss: 0.5145
    Epoch 126/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 205us/step - loss: 0.4445
    Epoch 127/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 196us/step - loss: 0.4388
    Epoch 128/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 200us/step - loss: 0.5580
    Epoch 129/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 207us/step - loss: 0.4343
    Epoch 130/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 199us/step - loss: 0.4116
    Epoch 131/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 568us/step - loss: 0.4271
    Epoch 132/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 194us/step - loss: 0.4077
    Epoch 133/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 195us/step - loss: 0.4102
    Epoch 134/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 196us/step - loss: 0.4461
    Epoch 135/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 205us/step - loss: 0.4094
    Epoch 136/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 201us/step - loss: 0.4361
    Epoch 137/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 196us/step - loss: 0.5203
    Epoch 138/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 201us/step - loss: 0.4220
    Epoch 139/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 276us/step - loss: 0.4025
    Epoch 140/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 201us/step - loss: 0.4458
    Epoch 141/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 200us/step - loss: 0.4189
    Epoch 142/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 208us/step - loss: 0.4506
    Epoch 143/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 199us/step - loss: 0.3855
    Epoch 144/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 203us/step - loss: 0.4088
    Epoch 145/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 201us/step - loss: 0.4635
    Epoch 146/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 453us/step - loss: 0.4228
    Epoch 147/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 197us/step - loss: 0.4922
    Epoch 148/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 205us/step - loss: 0.4117
    Epoch 149/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 202us/step - loss: 0.4381
    Epoch 150/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 199us/step - loss: 0.4313
    Epoch 151/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 202us/step - loss: 0.4304
    Epoch 152/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 203us/step - loss: 0.5194
    Epoch 153/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 210us/step - loss: 0.4188
    Epoch 154/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 207us/step - loss: 0.5009
    Epoch 155/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 204us/step - loss: 0.4369
    Epoch 156/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 218us/step - loss: 0.4737
    Epoch 157/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 221us/step - loss: 0.4188
    Epoch 158/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 198us/step - loss: 0.4327
    Epoch 159/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 222us/step - loss: 0.3996
    Epoch 160/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 208us/step - loss: 0.3984
    Epoch 161/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 198us/step - loss: 0.4581
    Epoch 162/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 206us/step - loss: 0.4282
    Epoch 163/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 199us/step - loss: 0.3992
    Epoch 164/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 199us/step - loss: 0.3906
    Epoch 165/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 201us/step - loss: 0.4004
    Epoch 166/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 237us/step - loss: 0.4737
    Epoch 167/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 200us/step - loss: 0.3672
    Epoch 168/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 210us/step - loss: 0.4071
    Epoch 169/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 210us/step - loss: 0.4618
    Epoch 170/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 193us/step - loss: 0.4472
    Epoch 171/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 199us/step - loss: 0.4207
    Epoch 172/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 199us/step - loss: 0.3804
    Epoch 173/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 552us/step - loss: 0.4428
    Epoch 174/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 216us/step - loss: 0.4445
    Epoch 175/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 207us/step - loss: 0.3993
    Epoch 176/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 208us/step - loss: 0.4279
    Epoch 177/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 209us/step - loss: 0.4213
    Epoch 178/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 216us/step - loss: 0.4086
    Epoch 179/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 207us/step - loss: 0.3481
    Epoch 180/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 204us/step - loss: 0.3935
    Epoch 181/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 572us/step - loss: 0.4812
    Epoch 182/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 293us/step - loss: 0.4430
    Epoch 183/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 194us/step - loss: 0.4602
    Epoch 184/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 208us/step - loss: 0.4569
    Epoch 185/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 207us/step - loss: 0.3841
    Epoch 186/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 198us/step - loss: 0.4131
    Epoch 187/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 202us/step - loss: 0.4001
    Epoch 188/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 199us/step - loss: 0.3754
    Epoch 189/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 210us/step - loss: 0.3811
    Epoch 190/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 558us/step - loss: 0.4234
    Epoch 191/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 213us/step - loss: 0.3676
    Epoch 192/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 266us/step - loss: 0.3964
    Epoch 193/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 193us/step - loss: 0.3994
    Epoch 194/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 197us/step - loss: 0.3963
    Epoch 195/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 201us/step - loss: 0.4015
    Epoch 196/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 200us/step - loss: 0.3566
    Epoch 197/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 199us/step - loss: 0.4146
    Epoch 198/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 200us/step - loss: 0.4533
    Epoch 199/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 204us/step - loss: 0.3809
    Epoch 200/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 627us/step - loss: 0.4129
    Epoch 201/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 201us/step - loss: 0.5071
    Epoch 202/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 217us/step - loss: 0.4027
    Epoch 203/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 203us/step - loss: 0.3622
    Epoch 204/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 201us/step - loss: 0.4791
    Epoch 205/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 201us/step - loss: 0.3573
    Epoch 206/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 201us/step - loss: 0.4118
    Epoch 207/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 612us/step - loss: 0.4275
    Epoch 208/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 202us/step - loss: 0.3682
    Epoch 209/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 197us/step - loss: 0.4819
    Epoch 210/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 204us/step - loss: 0.4382
    Epoch 211/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 199us/step - loss: 0.3536
    Epoch 212/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 201us/step - loss: 0.3464
    Epoch 213/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 191us/step - loss: 0.4250
    Epoch 214/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 287us/step - loss: 0.3639
    Epoch 215/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 199us/step - loss: 0.3859
    Epoch 216/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 191us/step - loss: 0.3695
    Epoch 217/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 197us/step - loss: 0.4196
    Epoch 218/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 194us/step - loss: 0.4541
    Epoch 219/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 207us/step - loss: 0.3525
    Epoch 220/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 199us/step - loss: 0.3563
    Epoch 221/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 272us/step - loss: 0.4045
    Epoch 222/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 196us/step - loss: 0.4403
    Epoch 223/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 212us/step - loss: 0.3599
    Epoch 224/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 204us/step - loss: 0.3542
    Epoch 225/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 194us/step - loss: 0.3684
    Epoch 226/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 195us/step - loss: 0.3655
    Epoch 227/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 547us/step - loss: 0.3496
    Epoch 228/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 209us/step - loss: 0.4598
    Epoch 229/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 199us/step - loss: 0.3675
    Epoch 230/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 193us/step - loss: 0.3678
    Epoch 231/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 198us/step - loss: 0.4786
    Epoch 232/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 196us/step - loss: 0.3858
    Epoch 233/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 199us/step - loss: 0.3368
    Epoch 234/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 559us/step - loss: 0.3890
    Epoch 235/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 205us/step - loss: 0.3429
    Epoch 236/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 199us/step - loss: 0.3653
    Epoch 237/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 193us/step - loss: 0.3374
    Epoch 238/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 195us/step - loss: 0.4715
    Epoch 239/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 196us/step - loss: 0.4367
    Epoch 240/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 195us/step - loss: 0.3965
    Epoch 241/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 589us/step - loss: 0.3152
    Epoch 242/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 206us/step - loss: 0.4334
    Epoch 243/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 201us/step - loss: 0.4030
    Epoch 244/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 199us/step - loss: 0.3435
    Epoch 245/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 195us/step - loss: 0.4368
    Epoch 246/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 199us/step - loss: 0.4649
    Epoch 247/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 201us/step - loss: 0.4262
    Epoch 248/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 635us/step - loss: 0.3453
    Epoch 249/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 202us/step - loss: 0.3371
    Epoch 250/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 213us/step - loss: 0.3345
    Epoch 251/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 201us/step - loss: 0.3616
    Epoch 252/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 201us/step - loss: 0.3777
    Epoch 253/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 205us/step - loss: 0.3463
    Epoch 254/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 610us/step - loss: 0.3277
    Epoch 255/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 205us/step - loss: 0.3546
    Epoch 256/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 197us/step - loss: 0.3890
    Epoch 257/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 200us/step - loss: 0.3320
    Epoch 258/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 205us/step - loss: 0.4805
    Epoch 259/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 197us/step - loss: 0.3673
    Epoch 260/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 200us/step - loss: 0.3234
    Epoch 261/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 630us/step - loss: 0.3816
    Epoch 262/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 198us/step - loss: 0.3810
    Epoch 263/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 206us/step - loss: 0.3327
    Epoch 264/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 205us/step - loss: 0.3537
    Epoch 265/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 201us/step - loss: 0.3476
    Epoch 266/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 199us/step - loss: 0.3505
    Epoch 267/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 193us/step - loss: 0.4483
    Epoch 268/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 572us/step - loss: 0.3315
    Epoch 269/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 214us/step - loss: 0.3277
    Epoch 270/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 199us/step - loss: 0.3326
    Epoch 271/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 201us/step - loss: 0.3432
    Epoch 272/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 202us/step - loss: 0.3731
    Epoch 273/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 196us/step - loss: 0.3345
    Epoch 274/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 195us/step - loss: 0.3329
    Epoch 275/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 576us/step - loss: 0.3417
    Epoch 276/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 200us/step - loss: 0.2995
    Epoch 277/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 209us/step - loss: 0.3322
    Epoch 278/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 197us/step - loss: 0.3898
    Epoch 279/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 198us/step - loss: 0.3489
    Epoch 280/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 199us/step - loss: 0.4103
    Epoch 281/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 196us/step - loss: 0.3310
    Epoch 282/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 645us/step - loss: 0.3507
    Epoch 283/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 206us/step - loss: 0.3870
    Epoch 284/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 198us/step - loss: 0.3624
    Epoch 285/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 199us/step - loss: 0.3206
    Epoch 286/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 197us/step - loss: 0.3811
    Epoch 287/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 197us/step - loss: 0.3412
    Epoch 288/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 199us/step - loss: 0.3452
    Epoch 289/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 259us/step - loss: 0.3440
    Epoch 290/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 202us/step - loss: 0.3639
    Epoch 291/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 202us/step - loss: 0.3984
    Epoch 292/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 199us/step - loss: 0.4477
    Epoch 293/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 194us/step - loss: 0.3081
    Epoch 294/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 199us/step - loss: 0.3106
    Epoch 295/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 574us/step - loss: 0.3040
    Epoch 296/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 218us/step - loss: 0.3537
    Epoch 297/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 198us/step - loss: 0.3227
    Epoch 298/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 192us/step - loss: 0.3335
    Epoch 299/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 199us/step - loss: 0.3494
    Epoch 300/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 210us/step - loss: 0.3011
    Epoch 301/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 194us/step - loss: 0.3298
    Epoch 302/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 551us/step - loss: 0.3606
    Epoch 303/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 213us/step - loss: 0.3705
    Epoch 304/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 214us/step - loss: 0.3782
    Epoch 305/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 196us/step - loss: 0.4093
    Epoch 306/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 195us/step - loss: 0.3495
    Epoch 307/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 200us/step - loss: 0.3179
    Epoch 308/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 197us/step - loss: 0.3206
    Epoch 309/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 642us/step - loss: 0.3093
    Epoch 310/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 203us/step - loss: 0.3279
    Epoch 311/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 205us/step - loss: 0.3261
    Epoch 312/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 196us/step - loss: 0.3058
    Epoch 313/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 196us/step - loss: 0.3360
    Epoch 314/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 198us/step - loss: 0.3239
    Epoch 315/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 198us/step - loss: 0.3003
    Epoch 316/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 643us/step - loss: 0.3187
    Epoch 317/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 198us/step - loss: 0.4024
    Epoch 318/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 197us/step - loss: 0.3427
    Epoch 319/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 201us/step - loss: 0.4949
    Epoch 320/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 190us/step - loss: 0.3294
    Epoch 321/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 200us/step - loss: 0.3101
    Epoch 322/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 214us/step - loss: 0.3236
    Epoch 323/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 292us/step - loss: 0.2845
    Epoch 324/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 195us/step - loss: 0.3213
    Epoch 325/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 202us/step - loss: 0.3530
    Epoch 326/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 204us/step - loss: 0.3377
    Epoch 327/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 217us/step - loss: 0.3239
    Epoch 328/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 204us/step - loss: 0.3786
    Epoch 329/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 199us/step - loss: 0.3264
    Epoch 330/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 620us/step - loss: 0.3280
    Epoch 331/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 201us/step - loss: 0.3184
    Epoch 332/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 195us/step - loss: 0.3053
    Epoch 333/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 204us/step - loss: 0.3112
    Epoch 334/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 201us/step - loss: 0.3120
    Epoch 335/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 194us/step - loss: 0.3110
    Epoch 336/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 195us/step - loss: 0.3283
    Epoch 337/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 651us/step - loss: 0.4004
    Epoch 338/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 196us/step - loss: 0.3401
    Epoch 339/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 209us/step - loss: 0.3890
    Epoch 340/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 201us/step - loss: 0.4221
    Epoch 341/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 192us/step - loss: 0.3113
    Epoch 342/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 200us/step - loss: 0.3414
    Epoch 343/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 199us/step - loss: 0.3484
    Epoch 344/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 590us/step - loss: 0.3347
    Epoch 345/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 206us/step - loss: 0.3277
    Epoch 346/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 201us/step - loss: 0.3901
    Epoch 347/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 197us/step - loss: 0.2873
    Epoch 348/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 193us/step - loss: 0.2910
    Epoch 349/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 196us/step - loss: 0.3174
    Epoch 350/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 198us/step - loss: 0.2931
    Epoch 351/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 212us/step - loss: 0.3055
    Epoch 352/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 294us/step - loss: 0.2717
    Epoch 353/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 197us/step - loss: 0.3558
    Epoch 354/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 200us/step - loss: 0.3258
    Epoch 355/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 203us/step - loss: 0.3003
    Epoch 356/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 196us/step - loss: 0.3270
    Epoch 357/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 210us/step - loss: 0.3304
    Epoch 358/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 201us/step - loss: 0.3197
    Epoch 359/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 644us/step - loss: 0.3182
    Epoch 360/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 199us/step - loss: 0.3057
    Epoch 361/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 197us/step - loss: 0.4353
    Epoch 362/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 200us/step - loss: 0.3595
    Epoch 363/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 198us/step - loss: 0.3042
    Epoch 364/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 199us/step - loss: 0.3122
    Epoch 365/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 193us/step - loss: 0.2922
    Epoch 366/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 555us/step - loss: 0.3059
    Epoch 367/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 283us/step - loss: 0.3249
    Epoch 368/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 201us/step - loss: 0.2963
    Epoch 369/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 192us/step - loss: 0.2944
    Epoch 370/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 201us/step - loss: 0.3195
    Epoch 371/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 199us/step - loss: 0.3474
    Epoch 372/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 209us/step - loss: 0.3269
    Epoch 373/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 207us/step - loss: 0.3305
    Epoch 374/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 651us/step - loss: 0.3110
    Epoch 375/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 201us/step - loss: 0.2752
    Epoch 376/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 196us/step - loss: 0.2910
    Epoch 377/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 206us/step - loss: 0.3086
    Epoch 378/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 197us/step - loss: 0.2900
    Epoch 379/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 207us/step - loss: 0.3128
    Epoch 380/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 200us/step - loss: 0.3841
    Epoch 381/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 567us/step - loss: 0.3075
    Epoch 382/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 204us/step - loss: 0.2891
    Epoch 383/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 198us/step - loss: 0.3919
    Epoch 384/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 195us/step - loss: 0.3007
    Epoch 385/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 204us/step - loss: 0.3488
    Epoch 386/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 204us/step - loss: 0.4248
    Epoch 387/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 202us/step - loss: 0.2623
    Epoch 388/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 201us/step - loss: 0.2749
    Epoch 389/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 642us/step - loss: 0.2728
    Epoch 390/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 194us/step - loss: 0.3365
    Epoch 391/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 199us/step - loss: 0.3399
    Epoch 392/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 207us/step - loss: 0.3431
    Epoch 393/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 198us/step - loss: 0.3082
    Epoch 394/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 199us/step - loss: 0.3020
    Epoch 395/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 195us/step - loss: 0.3541
    Epoch 396/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 563us/step - loss: 0.2824
    Epoch 397/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 227us/step - loss: 0.3682
    Epoch 398/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 199us/step - loss: 0.2721
    Epoch 399/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 199us/step - loss: 0.3693
    Epoch 400/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 194us/step - loss: 0.3176
    Epoch 401/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 196us/step - loss: 0.2778
    Epoch 402/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 195us/step - loss: 0.2787
    Epoch 403/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 204us/step - loss: 0.4581
    Epoch 404/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 615us/step - loss: 0.2655
    Epoch 405/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 198us/step - loss: 0.2792
    Epoch 406/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 199us/step - loss: 0.3246
    Epoch 407/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 204us/step - loss: 0.3377
    Epoch 408/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 202us/step - loss: 0.3144
    Epoch 409/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 198us/step - loss: 0.3420
    Epoch 410/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 197us/step - loss: 0.2658
    Epoch 411/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 194us/step - loss: 0.3169
    Epoch 412/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 610us/step - loss: 0.2760
    Epoch 413/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 199us/step - loss: 0.2841
    Epoch 414/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 213us/step - loss: 0.2824
    Epoch 415/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 199us/step - loss: 0.3036
    Epoch 416/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 197us/step - loss: 0.2616
    Epoch 417/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 196us/step - loss: 0.2752
    Epoch 418/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 200us/step - loss: 0.3178
    Epoch 419/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 190us/step - loss: 0.3064
    Epoch 420/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 296us/step - loss: 0.3269
    Epoch 421/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 198us/step - loss: 0.2842
    Epoch 422/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 195us/step - loss: 0.2916
    Epoch 423/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 202us/step - loss: 0.4570
    Epoch 424/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 203us/step - loss: 0.3146
    Epoch 425/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 206us/step - loss: 0.2804
    Epoch 426/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 203us/step - loss: 0.3596
    Epoch 427/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 198us/step - loss: 0.3110
    Epoch 428/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 297us/step - loss: 0.3951
    Epoch 429/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 202us/step - loss: 0.2897
    Epoch 430/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 194us/step - loss: 0.3334
    Epoch 431/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 216us/step - loss: 0.2547
    Epoch 432/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 199us/step - loss: 0.2548
    Epoch 433/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 201us/step - loss: 0.3427
    Epoch 434/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 198us/step - loss: 0.2789
    Epoch 435/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 582us/step - loss: 0.3260
    Epoch 436/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 211us/step - loss: 0.2598
    Epoch 437/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 194us/step - loss: 0.3221
    Epoch 438/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 201us/step - loss: 0.3108
    Epoch 439/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 192us/step - loss: 0.2566
    Epoch 440/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 196us/step - loss: 0.2597
    Epoch 441/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 211us/step - loss: 0.3108
    Epoch 442/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 197us/step - loss: 0.3656
    Epoch 443/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 492us/step - loss: 0.3187
    Epoch 444/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 200us/step - loss: 0.4023
    Epoch 445/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 199us/step - loss: 0.2929
    Epoch 446/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 221us/step - loss: 0.2489
    Epoch 447/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 198us/step - loss: 0.3592
    Epoch 448/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 199us/step - loss: 0.3097
    Epoch 449/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 199us/step - loss: 0.4151
    Epoch 450/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 545us/step - loss: 0.2677
    Epoch 451/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 217us/step - loss: 0.2469
    Epoch 452/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 201us/step - loss: 0.3101
    Epoch 453/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 200us/step - loss: 0.4067
    Epoch 454/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 197us/step - loss: 0.2856
    Epoch 455/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 202us/step - loss: 0.3025
    Epoch 456/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 197us/step - loss: 0.2682
    Epoch 457/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 211us/step - loss: 0.2935
    Epoch 458/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 610us/step - loss: 0.2389
    Epoch 459/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 203us/step - loss: 0.2679
    Epoch 460/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 204us/step - loss: 0.3306
    Epoch 461/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 203us/step - loss: 0.2960
    Epoch 462/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 219us/step - loss: 0.2662
    Epoch 463/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 199us/step - loss: 0.2581
    Epoch 464/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 206us/step - loss: 0.2661
    Epoch 465/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 195us/step - loss: 0.5221
    Epoch 466/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 548us/step - loss: 0.2686
    Epoch 467/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 198us/step - loss: 0.2598
    Epoch 468/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 202us/step - loss: 0.2987
    Epoch 469/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 200us/step - loss: 0.2967
    Epoch 470/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 199us/step - loss: 0.2725
    Epoch 471/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 195us/step - loss: 0.2740
    Epoch 472/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 198us/step - loss: 0.2728
    Epoch 473/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 568us/step - loss: 0.2530
    Epoch 474/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 272us/step - loss: 0.2623
    Epoch 475/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 198us/step - loss: 0.4370
    Epoch 476/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 196us/step - loss: 0.2642
    Epoch 477/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 199us/step - loss: 0.3010
    Epoch 478/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 196us/step - loss: 0.2668
    Epoch 479/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 208us/step - loss: 0.2925
    Epoch 480/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 198us/step - loss: 0.2906
    Epoch 481/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 556us/step - loss: 0.2801
    Epoch 482/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 254us/step - loss: 0.3421
    Epoch 483/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 197us/step - loss: 0.2966
    Epoch 484/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 211us/step - loss: 0.2979
    Epoch 485/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 193us/step - loss: 0.3883
    Epoch 486/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 192us/step - loss: 0.2827
    Epoch 487/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 196us/step - loss: 0.3570
    Epoch 488/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 201us/step - loss: 0.2406
    Epoch 489/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 589us/step - loss: 0.2709
    Epoch 490/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 220us/step - loss: 0.2555
    Epoch 491/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 193us/step - loss: 0.2424
    Epoch 492/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 202us/step - loss: 0.3071
    Epoch 493/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 198us/step - loss: 0.2249
    Epoch 494/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 210us/step - loss: 0.2789
    Epoch 495/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 201us/step - loss: 0.2468
    Epoch 496/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 194us/step - loss: 0.3293
    Epoch 497/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 638us/step - loss: 0.2891
    Epoch 498/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 202us/step - loss: 0.2340
    Epoch 499/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 211us/step - loss: 0.2728
    Epoch 500/500
    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 197us/step - loss: 0.2677





    <keras.src.callbacks.history.History at 0x167a037a0>



- Now that the model is trained, we can use the original dataset, $X\_train$ to calculate the predicted results, $yhat$.
- Let's then compare $yhat$ with $Y\_train$ to calculate the accuracy of the model. 


```python
#predict and view accuracy 
logits = model(X_train)

#Here, we are using softmax algo as the activation function
f_x=tf.nn.softmax(logits).numpy() #shape of f_x is (2000, 4)

yhat = []

#softmax activation returns us with a list of probabilities of the respective indices being the "correct" value
#we need to find out what is the highest probability within that list and return the indices
for i in range(m):
    yhat.append(np.argmax(f_x[i])) #np.argmax returns the indices of element with the highest numerical value within a list 

print(f"Accuracy is {np.count_nonzero(yhat==Y_train)/m}")
```

    Accuracy is 0.836


#### Hooray! 
We managed to achieve accuracy of >0.80. Let's use our model to predict the price categories of a random set of data given in CellPhone_test.csv


```python
test  = pd.read_csv("CellPhone_test.csv")

#drop the id column
test = test.drop(labels=['id'], axis=1)
test = test.reindex(labels=X_features, axis=1)

#convert pandas dataframe to numpy array
test.to_numpy()

#plug the data into the model
prediction = model.predict(test)
function=tf.nn.softmax(prediction).numpy()

#softmax function
results = [np.argmax(function[row]) for row in range(20)]

print(results)
```

    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 398us/step
    [2, 3, 2, 3, 1, 2, 3, 1, 3, 0, 3, 3, 0, 0, 2, 0, 2, 1, 3, 2]


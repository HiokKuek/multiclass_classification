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
    .
    .
    .
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


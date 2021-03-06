from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam



def mlp(n_obs, n_action, n_hidden_layer=1, n_neuron_per_layer=16,
        activation='relu', loss='mse'):
  """ A multi-layer perceptron """
  n_action += 1
  #print(n_action)
  model = Sequential()
  model.add(Dense(n_neuron_per_layer, input_dim=n_obs, activation=activation))
  for _ in range(n_hidden_layer):
    model.add(Dense(n_neuron_per_layer, activation=activation))
  model.add(Dense(n_action, activation='sigmoid'))
  model.compile(loss=loss, optimizer=Adam())
  print(model.summary())
  return model
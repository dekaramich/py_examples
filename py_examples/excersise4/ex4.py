"""
The problem at hand involves modeling the dynamics of Mackey-Glass chaotic time
series. The goal is to develop a predictive model that captures the underlying
patterns and dynamics within these time series.

The proposed approach is based on making a model for a single trajectory that
predicts differences instead of target values. By predicting the differences
between consecutive target values, we can assess whether the model is capturing
the changes and trends in the data rather than simply replicating the original target values.
"""

import numpy as np
import tensorflow as tf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import GRU, Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Define random seed generator
np.random.seed(42) # make stable outputs across runs

# Define parameters
L = 10 # look-back timesteps
batch_size = 32 # batch size
n_epochs = 20 # number of epochs
patience = 10  # number of epochs to wait for improvement in val. loss
trajectory = 0 # trajectory to work on

# read the data (needs to be in the same folder as ex4.py)
root_dir = "."
dataset = "MackeyGlass"
data_path = os.path.join(root_dir, "data", dataset)

def get_data(data_path, subset):
    with open(os.path.join(data_path, subset), 'rb') as f:
        data = pickle.load(f)
    return data

# get train, val, test subsets
data_subsets = {}
for subset in ["train.pickle", "val.pickle", "test.pickle"]:
    file_name = os.path.splitext(subset)[0]
    data = get_data(data_path, subset)
    data_subsets[file_name] = data['data']

# dir to save the figures
exc = "excersise4"
images_path = os.path.join(root_dir, "images")
os.makedirs(images_path, exist_ok=True)
    
def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(images_path, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    plt.savefig(path, format=fig_extension, dpi=resolution) 

# Work on one trajectory
training_data = data_subsets['train'][trajectory]
validation_data = data_subsets['val'][trajectory]
testing_data = data_subsets['test'][trajectory]

# Plot ACF and PACF
#plot_acf(training_data, lags=L)
#save_fig("ACF_traj_"+str(trajectory), tight_layout=True, fig_extension="png", resolution=300)
#plt.show()

# Plot ACF and PACF
#plot_pacf(training_data, lags=L)
#save_fig("PACF_traj_"+str(trajectory), tight_layout=True, fig_extension="png", resolution=300)
#plt.show()

# Get training, validation and testing batches
train_generator = TimeseriesGenerator(training_data, training_data, length=L, batch_size=1)
val_generator = TimeseriesGenerator(validation_data, validation_data, length=L, batch_size=1)
test_generator = TimeseriesGenerator(testing_data, testing_data, length=L, batch_size=1)

# fun to prepare batches
def get_batches(generator):
    input_data = []
    output_data = []

    for i in range(1, len(generator)):
        batch_x, batch_y = generator[i]
        prev_batch_x, prev_batch_y = generator[i - 1]
        input_data.append(batch_x)  # Append the input differences
        output_data.append(batch_y - prev_batch_y)  # Append the outputs to the output_data list
        
    # Create empty lists to store the stacked arrays
    stacked_input_list = []
    stacked_output_list = []

    # Iterate through each array in input_data and output_data
    for arr_input, arr_output in zip(input_data, output_data):
        # Stack the L elements of the current array vertically
        stacked_input = np.vstack([arr_input[i, :L] for i in range(arr_input.shape[0])])
        stacked_output = np.vstack([arr_output[i, :L] for i in range(arr_output.shape[0])])

        # Append the stacked arrays to the result lists
        stacked_input_list.append(stacked_input)
        stacked_output_list.append(stacked_output)

    # Concatenate the stacked arrays from the lists
    result_input = np.concatenate(stacked_input_list, axis=0)
    result_output = np.concatenate(stacked_output_list, axis=0)

    result_input = np.reshape(result_input, (len(result_output), L))
    
    return result_input, result_output

# Get training and validation batches
X_train, y_train = get_batches(train_generator)
X_val, y_val = get_batches(val_generator)
X_test, y_test = get_batches(test_generator)

# Scale the data, inputs and outputs
scaler_x, scaler_y = MinMaxScaler(feature_range=(-1, 1)), MinMaxScaler(feature_range=(-1, 1))

training_data_scaled = scaler_x.fit_transform(X_train)
training_differences_scaled = scaler_y.fit_transform(y_train)

validation_data_scaled = scaler_x.transform(X_val)
validation_differences_scaled = scaler_y.transform(y_val)

testing_data_scaled = scaler_x.transform(X_test)
testing_differences_scaled = scaler_y.transform(y_test)

# Calculate the number of batches for training
num_batches = len(training_data_scaled) // batch_size

# Reshape the data into batches
# training
training_input_data_batches = np.array_split(training_data_scaled[:num_batches * batch_size], num_batches)
training_output_data_batches = np.array_split(training_differences_scaled[:num_batches * batch_size], num_batches)

# validation
validation_input_data_batches = np.array_split(validation_data_scaled[:num_batches * batch_size], num_batches)
validation_output_data_batches = np.array_split(validation_differences_scaled[:num_batches * batch_size], num_batches)

# testing
testing_input_data_batches = np.array_split(testing_data_scaled[:num_batches * batch_size], num_batches)
testing_output_data_batches = np.array_split(testing_differences_scaled[:num_batches * batch_size], num_batches)

# Create a simple GRU model
# (further tuning is needed, 
# (number of hidden layers, number of nodes, Look back time steps, etc))
model = Sequential()
model.add(GRU(64, input_shape=(L, 1), activation='tanh'))
model.add(Dense(1))

# Define Adam optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
optimizer.build(model.trainable_variables)

# Define loss function
def custom_mean_squared_error(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return tf.reduce_mean(tf.square(y_true - y_pred))

model.compile(optimizer=optimizer, loss=custom_mean_squared_error)

# fun for model training
def train_step(batch_x, batch_y):
    with tf.GradientTape() as tape:
        # forward pass
        predictions = model(batch_x)
        
        # scaled loss
        loss_value = custom_mean_squared_error(batch_y, predictions)
        
        real_predictions = scaler_y.inverse_transform(predictions)
        real_batch_y = scaler_y.inverse_transform(batch_y)
        
        # "unscaled" loss
        real_loss_value = custom_mean_squared_error(real_batch_y, real_predictions)

    # Perform backward pass
    # Compute gradients
    gradients = tape.gradient(loss_value, model.trainable_variables)

    # Update weights
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss_value, real_loss_value

# fun for model evaluation
def eval_predict(inputs, outputs):
    predictions = []
    real_prediction_all = []
    real_targets_all = []
    val_loss = 0.0  # Variable to store the validation loss
    num_batches = 0
    
    for i in range(len(inputs)):
        batch_x = tf.convert_to_tensor(inputs[i]) # inputs
        batch_y = tf.convert_to_tensor(outputs[i]) # outputs
    
        predictions_batch = model(batch_x) # forward pass
        predictions.append(predictions_batch)
        
        real_predictions = scaler_y.inverse_transform(predictions_batch) # "unscaled" predictions
        real_targets = scaler_y.inverse_transform(batch_y) # "unscaled" targets
 
        real_prediction_all.append(real_predictions)
        real_targets_all.append(real_targets)
        
        real_loss_value = custom_mean_squared_error(real_targets, real_predictions) # "unscaled" loss
        
        val_loss += real_loss_value
        num_batches += 1
        
    # Calculate average validation loss
    avg_val_loss = val_loss / num_batches

    return real_prediction_all, real_targets_all, avg_val_loss

# Train the model
# Due to the stochastic nature of the training algorithm,
# it is advisable to conduct multiple runs to obtain mean and 
# standard deviation values in order to assess the consistency of the results.
# (omitted here)

train_losses, val_losses = [], [] # initialize lists for losses
min_val_loss = float('inf')  # initialize with a large value
consecutive_no_improvement = 0 # for early stopping 

# Training
print("Training starts...")
for epoch in range(n_epochs):
    
    train_loss = 0.0
    train_real_loss = 0.0
    num_batches = 0
    
    if epoch == 0:
        # Calculate loss before training
        _, _, avg_train_0_loss = eval_predict(training_input_data_batches, training_output_data_batches)
        print("Training Loss before training: {:.4e}, ".format(avg_train_0_loss.numpy()))

    print("Epoch:", epoch + 1)
    # Train on batches
    for i in range(len(training_input_data_batches)):
        batch_x = tf.convert_to_tensor(training_input_data_batches[i])
        batch_y = tf.convert_to_tensor(training_output_data_batches[i])
        
        loss, real_loss = train_step(batch_x, batch_y)
        train_loss += loss # scaled loss
        train_real_loss += real_loss # "unscaled" loss
        num_batches += 1

    # Calculate average training loss
    avg_train_loss = train_real_loss / num_batches
    train_losses.append(avg_train_loss)
    
    # Calculate average validation loss
    _, _, avg_val_loss = eval_predict(validation_input_data_batches, validation_output_data_batches)
    val_losses.append(avg_val_loss)
  
    print("Training Loss: {:.4e}, ".format(avg_train_loss.numpy()), "Validation Loss: {:.4e}".format(avg_val_loss))

    # Check if the current validation loss is lower than the minimum
    if avg_val_loss < min_val_loss:
        # Save the model
        model.save("best_model.h5")
        min_val_loss = avg_val_loss
        print("Model saved at epoch", epoch + 1)
        consecutive_no_improvement = 0
    else:
        consecutive_no_improvement += 1

    # Check if training should stop
    if consecutive_no_improvement >= patience:
        print("No improvement in validation loss for", patience, "consecutive epochs. Stopping training.")
        break
print("Training ended")

# At the end of training, load the best model
model = tf.keras.models.load_model("best_model.h5", custom_objects={"custom_mean_squared_error": custom_mean_squared_error})

# Plot the training and validation losses
plt.plot(train_losses,  label='Training Loss')
plt.plot(val_losses,  label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
save_fig("train_val_losses_traj_"+str(trajectory), tight_layout=True, fig_extension="png", resolution=300)
plt.show()

# fun for One-step predictions
def onestep_predictions(inputs):
    onestep_predictions = []
    # Iterate through all batches
    for i in range(len(inputs)):
        # Prepare inputs
        batch_x = tf.convert_to_tensor(inputs[i].reshape(1,L))
        
        # Prediction
        predicted_difference = model(batch_x)

        # Inverse prediction
        predicted_difference = scaler_y.inverse_transform(predicted_difference)

        # Append predictions to list
        onestep_predictions.append(predicted_difference)

    # Format the output
    onestep_predictions = np.concatenate(onestep_predictions, axis=0)
    return onestep_predictions

# Print trajectory
print("Trajectory",trajectory)

# Get one-step predictions
print("One-step predictions")

# 1step predictions for training
onestep_training = onestep_predictions(training_data_scaled)

# 1step predictions for validation
onestep_validation = onestep_predictions(validation_data_scaled)

# 1step predictions for testing
onestep_testing = onestep_predictions(testing_data_scaled)

# Calculate basic regression metrics for 1-step predictions
# Training subset
# Calculate MSE
mse_train_1s = custom_mean_squared_error(y_train, onestep_training)
# Calculate R2
r2_train_1s = r2_score(y_train, onestep_training)
print("Training Metrics:\nMSE: {:.4e}, R2: {:2.2f}%".format(mse_train_1s, r2_train_1s * 100))

# Validation subset
# Calculate MSE
mse_val_1s = custom_mean_squared_error(y_val, onestep_validation)
# Calculate R2
r2_val_1s = r2_score(y_val, onestep_validation)
print("Validtaion Metrics:\nMSE: {:.4e}, R2: {:2.2f}%".format(mse_val_1s, r2_val_1s * 100))

# Testing subset
# Calculate MSE
mse_test_1s = custom_mean_squared_error(y_test, onestep_testing)
# Calculate R2
r2_test_1s = r2_score(y_test, onestep_testing)
print("Testing Metrics:\nMSE: {:.4e}, R2: {:2.2f}%".format(mse_test_1s, r2_test_1s * 100))

# Plot target difference vs predicted difference (testing subset)
plt.plot(y_test, label='Target Differences')
plt.plot(onestep_testing, label='Predicted Differences')
plt.xlabel('Time step')
plt.ylabel('Difference Value')
plt.title(f"1-step, Testing subset, Trajectory {trajectory}")
plt.grid(True, which='both', linestyle=':')
plt.legend(loc='lower left')
save_fig("onestep_test_y_timestep_traj_"+str(trajectory), tight_layout=True, fig_extension="png", resolution=300)
plt.show()

# Plot target difference vs predicted difference (testing subset)
plt.plot(y_test, marker='.', linestyle='-', label='Target Differences')
plt.plot(onestep_testing, marker='.', linestyle='-', label='Predicted Differences')
plt.xlabel('Time step')
plt.ylabel('Difference Value')
plt.title(f"1-step, Testing subset, Trajectory {trajectory}")
plt.grid(True, which='both', linestyle=':')
plt.legend(loc='lower left')
# Set the x-axis limits to a specific range
x_start = 450  # Starting index of the range
x_end = 500  # Ending index of the range
plt.xlim(x_start, x_end)
save_fig("onestep_test_y_timestep_zoomed_traj_"+str(trajectory), tight_layout=True, fig_extension="png", resolution=300)
plt.show()


# Plot the target data on the first y-axis, and predicted differences on secondary y-axis (testing subset)
fig, ax1 = plt.subplots()
ax1.plot(testing_data[L+1:], label='target', color='blue', marker='.', linestyle='-')
ax1.set_ylabel('Value', color='blue')
ax1.set_xlabel('Time step', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
# Create a second y-axis
ax2 = ax1.twinx()
# Plot the predictions (onestep_testing) on the second y-axis
ax2.plot(onestep_testing, label='predictions differences', color='red', marker='.', linestyle='-')
#ax2.plot(y_test, label='target differences', color='cyan', marker='.', linestyle='--')
ax2.set_ylabel('Differences', color='red')
ax2.tick_params(axis='y', labelcolor='red')
# Set title and legend
plt.title(f"1-step, Testing subset, Trajectory {trajectory}")
#plt.legend(loc='upper right')
# Set the x-axis limits to a specific range
x_start = 500  # Starting index of the range
x_end = 600  # Ending index of the range
plt.xlim(x_start, x_end)
plt.ylim(-0.5, 0.5)
#plt.legend()
# Show grid lines in both x-axis and y-axis for the primary axis
ax1.grid(axis='x', linestyle=':')
ax1.grid(axis='y', linestyle=':')
# Show minor grid lines in both x-axis and y-axis for the secondary axis
ax2.grid(axis='y', which='both', linestyle=':', alpha=0.5)
save_fig("onestep_test_y_target_timesteps_traj_"+str(trajectory), tight_layout=True, fig_extension="png", resolution=300)
# Show the plot
plt.show()
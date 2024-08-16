from keras.models import model_from_json

# Load the model architecture
with open('model_config.json', 'r') as json_file:
    model_json = json_file.read()

# Modify the configuration string to remove "time_major"
model_json = model_json.replace('"time_major": false,', '')

# Recreate the model
model = model_from_json(model_json)

# Load the weights
model.load_weights('stock_model_50.h5')

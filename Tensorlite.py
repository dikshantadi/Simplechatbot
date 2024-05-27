import tensorflow as tf

# Load the Keras model
model = tf.keras.models.load_model('chatbot_model.h5')

# Ensure the input shape matches the model's input requirements
input_shape = model.input_shape[1:]

# Create a concrete function from the model
run_model = tf.function(lambda x: model(x))
concrete_func = run_model.get_concrete_function(tf.TensorSpec([1] + list(input_shape), model.inputs[0].dtype))

# Convert the model to TFLite format
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
tflite_model = converter.convert()

# Save the converted TFLite model
with open('chatbot_model.tflite', 'wb') as f:
    f.write(tflite_model)

# Optionally, optimize the model
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()

# Save the quantized TFLite model
with open('chatbot_model_quant.tflite', 'wb') as f:
    f.write(tflite_quant_model)

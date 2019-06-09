import tensorflow as tf
import model as grmodel

tf.keras.backend.set_learning_phase(0)
model = grmodel.get_model()
model.load_weights("trained_model/model.h5")
export_path = 'exported_model'

# Fetch the Keras session and save the model
# The signature definition is defined by the input and output tensors
# And stored with the default serving key
with tf.keras.backend.get_session() as sess:
    tf.saved_model.simple_save(
        sess,
        export_path,
        inputs={'input_1:0': model.input},
        outputs={t.name: t for t in model.outputs})
    print("input", model.input)
    print("output", {t.name: t for t in model.outputs})
    print("Exported model!")

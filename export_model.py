import tensorflow as tf
import model as grmodel

#tf.keras.backend.set_learning_phase(0)
# model = grmodel.get_model_v2()
# model.load_weights("trained_models/v3/model.h5")
model = tf.keras.models.load_model("trained_models/v3/model.h5")
export_path = 'exported_models/v3'

# Fetch the Keras session and save the model
# The signature definition is defined by the input and output tensors
# And stored with the default serving key
with tf.keras.backend.get_session() as sess:
    print("input", model.input)
    print("output", {t.name: t for t in model.outputs})
    tf.saved_model.simple_save(
        sess,
        export_path,
        inputs={'conv3d_input:0': model.input},
        outputs={t.name: t for t in model.outputs})

    print("Exported model!")

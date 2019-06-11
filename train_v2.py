import tensorflow as tf
import model as grmodel
import generators
import os

BATCH_SIZE = 64
EPOCHS = 8
TOTAL_EXAMPLES = 118562
TRAIN_LIST_PATH = "D:/Datasets/20bn-jester/jester-v1-train.csv"
EVAL_LIST_PATH = "D:/Datasets/20bn-jester/jester-v1-validation.csv"
OUTPUT_PATH = "trained_model_v2"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
session = tf.Session(config=config)
tf.keras.backend.set_session(session)
tf.logging.set_verbosity(tf.logging.ERROR)

print("Loading model")
model = grmodel.get_model_v2()
model.summary()

print("Starting Training")
data_ids = generators.get_data_ids_from_file(TRAIN_LIST_PATH)[:TOTAL_EXAMPLES]
eval_ids = generators.get_data_ids_from_file(EVAL_LIST_PATH)
labels = generators.get_labels()
train_generator = generators.DataGeneratorV2(data_ids, labels, **{'batch_size': BATCH_SIZE})
eval_generator = generators.DataGeneratorV2(eval_ids, labels, **{'batch_size': BATCH_SIZE})

model.fit_generator(train_generator, validation_data=eval_generator, steps_per_epoch=int(TOTAL_EXAMPLES / BATCH_SIZE),
                    validation_steps=32, epochs=EPOCHS, verbose=1, max_queue_size=20, workers=10,
                    use_multiprocessing=False, callbacks=[
        tf.keras.callbacks.TensorBoard(log_dir=OUTPUT_PATH, write_graph=True, batch_size=BATCH_SIZE, update_freq=50),
        tf.keras.callbacks.ModelCheckpoint(os.path.join(OUTPUT_PATH, "model.{epoch:02d}.ckpt"), save_weights_only=True,
                                           verbose=1)
    ])

model.save(os.path.join(OUTPUT_PATH, "model.h5"))

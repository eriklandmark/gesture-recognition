import tensorflow as tf
import model as grmodel
import generators

BATCH_SIZE = 14
EPOCHS = 6
TOTAL_EXAMPLES = 118562
TRAIN_LIST_PATH = "D:/Datasets/20bn-jester/jester-v1-train.csv"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
session = tf.Session(config=config)
tf.keras.backend.set_session(session)
tf.logging.set_verbosity(tf.logging.ERROR)

print("Loading model")
model = grmodel.get_model()
model.summary()

print("Starting Training")
data_ids = generators.get_data_ids_from_file(TRAIN_LIST_PATH)[:TOTAL_EXAMPLES]
labels = generators.get_labels()
train_generator = generators.DataGenerator(data_ids, labels, **{'batch_size': BATCH_SIZE})

model.fit_generator(train_generator, steps_per_epoch=int(TOTAL_EXAMPLES / BATCH_SIZE), epochs=EPOCHS, verbose=1,
                    max_queue_size=10, workers=4, use_multiprocessing=False, callbacks=[
        tf.keras.callbacks.TensorBoard(log_dir="logs", histogram_freq=0, write_grads=False, write_graph=True,
                                       batch_size=BATCH_SIZE, embeddings_freq=0, update_freq='batch'),
        tf.keras.callbacks.ModelCheckpoint("trained_model/model.{epoch:02d}.ckpt", save_weights_only=True, verbose=1)
    ])
    
model.save("trained_model/model.h5")

import model as grmodel
import tensorflow as tf
import generators

USE_GENERATOR = False
EVAL_LIST_PATH = "D:/Datasets/20bn-jester/jester-v1-validation.csv"

# model = grmodel.get_model_v2()
# model.load_weights("trained_model_v2/model.h5")
model = tf.keras.models.load_model("trained_models/v3/model.h5")
model.summary()

labels = generators.get_labels()
data_ids = generators.get_data_ids_from_file(EVAL_LIST_PATH)
generator = generators.DataGeneratorV2(data_ids, labels, **{"verbose":False, "usable_frames": 16})

if USE_GENERATOR:
    score = model.evaluate_generator(generator, steps=10, workers=10, verbose=1)
else:
    X, Y = generator.create_batch(data_ids[6000:7000], labels, verbose=True)
    print(X.shape)
    score = model.evaluate(X, Y, verbose=1)

print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

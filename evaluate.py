import model as grmodel
import generators

USE_GENERATOR = False
EVAL_LIST_PATH = "D:/Datasets/20bn-jester/jester-v1-validation.csv"

model = grmodel.get_model()
model.load_weights("trained_model/model.h5")
# model.summary()

labels = generators.get_labels()
data_ids = generators.get_data_ids_from_file(EVAL_LIST_PATH)

if USE_GENERATOR:
    generator = generators.DataGenerator(data_ids, labels, verbose=False)
    score = model.evaluate_generator(generator, steps=200, workers=10, verbose=1)
else:
    data = data_ids[2092:2093]
    print(data)
    X, Y = generators.create_batch(data, labels, verbose=True)
    score = model.evaluate(X, Y, verbose=1)

print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

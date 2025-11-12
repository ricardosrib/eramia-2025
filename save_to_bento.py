import bentoml
import pickle

model = pickle.loads(open('data/models/model.pickle', 'rb').read())

# salva modelo dentro do bentoML
saved_model = bentoml.sklearn.save_model(
    'bbb-model',
    model,
    signatures={
        "predict": {"batchable": True},
        "predict_proba": {"batchable": True}
    }
)

print(saved_model)
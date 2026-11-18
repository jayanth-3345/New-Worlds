import mlflow
import pandas as pd
import skops.io as sio

model_path = (
    "mlruns/1/models/"
    "m-d7ca0885a7f8463a94c9227a69d92725/artifacts/model.skops"
)

print("Loading saved Random Forest model...")

model = sio.load(
    model_path,
    trusted=[
        "numpy.dtype",
        "sklearn.tree._tree.Tree"
    ]
)

sample = pd.DataFrame([{
    "pl_rade": 1.0,
    "pl_eqt": 280.0,
    "pl_insol": 1.0,
    "st_teff": 5778.0,
    "st_mass": 1.0,
    "st_rad": 1.0
}])

prediction = model.predict(sample)

print("Prediction:", prediction[0])
print("Potentially habitable:", "Yes" if prediction[0] == 1 else "No")
print("Saved model loaded and prediction completed!")

from giza.agents.model import GizaModel
import pandas as pd
from torch.utils.data import DataLoader
from ferplus import FerPlus
    
training_data = FerPlus('Training', '/home/alexandre/Documents/repos/facial-recog/FERPlus/fer2013new.csv', '/home/alexandre/Documents/repos/facial-recog/FERPlus/')

batch_size = 2**12

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

img_list = []

for X, y in training_data:
    img_list.append(X.numpy().flatten())

print(len(img_list))

img_dataframe = pd.DataFrame(img_list)

def prediction(input, model_id, version_id):
    model = GizaModel(id=model_id, version=version_id)

    (result, proof_id) = model.predict(
        input_feed={"input": input}, verifiable=True, model_category="XGB"
    )

    return result, proof_id


MODEL_ID = 845  # Update with your model ID
VERSION_ID = 8  # Update with your version ID

print(img_list[0])
result = prediction(img_list[0], MODEL_ID, VERSION_ID)
print(result)
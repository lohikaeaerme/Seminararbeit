import torch
from plotting import plot
# ----------------------------
# Inference
# ----------------------------
def inference (model, val_dl, device, df, print_predictions=False):
  correct_prediction = 0
  total_prediction = 0

  # Disable gradient updates
  with torch.no_grad():
    for index, data in enumerate(val_dl):
      # Get the input features and target labels, and put them on the GPU
      inputs, labels = data[0].to(device), data[1].to(device)

      # Normalize the inputs
      inputs_m, inputs_s = inputs.mean(), inputs.std()
      inputs = (inputs - inputs_m) / inputs_s

      # Get predictions
      outputs = model(inputs)

      # Get the predicted class with the highest score
      _, prediction = torch.max(outputs,1)

      # Save pictures
      if index <= 1:
        plot(inputs, labels, index, 0, df, data[2], prediction)

      # Count of predictions that matched the target label
      correct_prediction += (prediction == labels).sum().item()
      total_prediction += prediction.shape[0]
      if print_predictions:
        print(f'label: {labels}, prediction: {prediction}')
    
  acc = correct_prediction/total_prediction
  print(f'Accuracy: {acc:.2f}, Total items: {total_prediction}')
  return acc
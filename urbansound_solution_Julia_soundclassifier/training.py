import torch
from torch import nn
from tqdm import tqdm
from validation import inference
from loss import LabelSmoothCrossEntropyLoss


# ----------------------------
# Training Loop
# ----------------------------
def training(model, train_dl, num_epochs, device, fold_nr, df, test_dl= None):
  # Loss Function, Optimizer and Scheduler
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.AdamW(model.parameters(),lr=0.0001)
  scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.0001,
                                                steps_per_epoch=int(len(train_dl)),
                                                epochs=num_epochs,
                                                anneal_strategy='linear')

  test_acc = []
  train_acc = []
  losses = []
  # Repeat for each epoch
  for epoch in tqdm(range(num_epochs)):
    running_loss = 0.0
    correct_prediction = 0
    total_prediction = 0

    # Repeat for each batch in the training set
    for i, data in enumerate(train_dl):
        # Get the input features and target labels, and put them on the GPU
        inputs, labels = data[0].to(device), data[1].to(device)

        # standardize the inputs
        inputs_m, inputs_s = inputs.mean(), inputs.std()
        inputs = (inputs - inputs_m) / inputs_s

        # # save pictures
        # if i <= 1:
        #    indices_in_df = data[2]
        #    plot(inputs, labels, fold_nr, i, epoch, df, indices_in_df)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Keep stats for Loss and Accuracy
        running_loss += loss.item()

        # Get the predicted class with the highest score
        _, prediction = torch.max(outputs,1)
        # Count of predictions that matched the target label
        correct_prediction += (prediction == labels).sum().item()
        total_prediction += prediction.shape[0]
    
    # Print stats at the end of the epoch
    num_batches = len(train_dl)
    avg_loss = running_loss / num_batches
    acc = correct_prediction/total_prediction
    print(f'for trainingdata: Epoch: {epoch}, Loss: {avg_loss:.2f}, Accuracy: {acc:.2f}')

    print('On test Data')
    if(test_dl != None):
      test_acc.append(inference(model, test_dl, device, (fold_nr+1)%10, df))
      train_acc.append(acc)
      losses.append(avg_loss)

  print('Finished Training')
  return test_acc, train_acc, losses
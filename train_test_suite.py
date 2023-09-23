import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score
import wandb


def train(dataloader: DataLoader,
          model: nn.Module,
          loss_fn: nn.modules.loss._Loss,  # type hints are stupid and difficult for this
          optimizer: torch.optim.Optimizer,
          device: str,
          verbose: bool = False) -> float:
    size = len(dataloader.dataset)
    model.train()  # note we didn't define this, it must be in the parent class

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)  # send the work to the device

        prediction = model(X)  # compute the prediction
        loss = loss_fn(prediction, y)  # compute the loss value

        # now do back prop
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()  # zero the gradients

        if batch % 100 == 0 and verbose:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f} [{current: >5d}/{size:>5d}]")

    return loss.item()  # return the final loss


def test(dataloader: DataLoader,
         model: nn.Module,
         loss_fn: nn.modules.loss._Loss,  # type hints are stupid and difficult for this
         device: str,
         verbose: bool = False) -> tuple:

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0

    y_pred_list = []
    y_true_list = []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            prediction = model(X)

            y_true = y.numpy().tolist()
            y_pred = (prediction.argmax(1) == y).type(torch.float).numpy().tolist()

            y_pred_list.append(y_pred)
            y_true_list.append(y_true)

            test_loss += loss_fn(prediction, y).item()
            correct += (prediction.argmax(1) == y).type(torch.float).sum().item()  # wtf is this doing?!

    test_loss /= num_batches
    correct /= size

    if verbose:
        print(f"Test Error: \n Accuracy: {(100 * correct): >0.1f}%, avg loss: {test_loss: >8f} \n")

    return correct, test_loss, sum(y_pred_list, []), sum(y_true_list, [])

def train_and_test_model(
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        model: nn.Module,
        loss_fn: nn.modules.loss._Loss,  # type hints are stupid and difficult for pytorch
        optimizer: torch.optim.Optimizer,
        device: str,
        epochs: int = 10,
        verbose: bool = False,
        wandb=None) -> dict:

    training_losses = []
    testing_losses = []
    testing_accuracies = []
    epoch = []

    iterator = tqdm(range(epochs)) if not verbose else range(epochs)
    for t in iterator:
        if verbose:
            print(f"Epoch {t + 1}:\n ----------------------------------------------------------")

        training_loss = train(train_dataloader, model, loss_fn, optimizer, device, verbose=verbose)
        training_losses.append(training_loss)

        test_acc, test_loss, y_pred_list, y_true_list = test(test_dataloader, model, loss_fn, device, verbose=verbose)
        testing_losses.append(test_loss)
        testing_accuracies.append(test_acc)
        epoch.append(t)

        if not verbose:
            iterator.set_description(
                f"Training Loss: {training_loss:.2f} Testing Loss: {test_loss:.2f} Accuracy {test_acc:.2f}"
            )

        if wandb:
            wandb.log({"training loss": training_loss})
            wandb.log({"testing loss": test_loss})
            wandb.log({"accuracy": test_acc})
            wandb.log({"epoch": t})


    wandb.log({"Best_F1": f1_score(y_true_list, y_pred_list)})

    return {"training_loss": training_losses,
            "testing_loss": testing_losses,
            "testing_accuracy": testing_accuracies,
            "epoch": epoch,
            "Best_F1": f1_score(y_true_list, y_pred_list),
            "y_true": y_true_list, "y_pred": y_pred_list}


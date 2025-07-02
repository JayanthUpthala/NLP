
import sys
import Utils
import torch
from Network import Network
import numpy as np

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_loader = Utils.get_data_loader(batch_size=1)
    loss_fn = torch.nn.MSELoss()  # or torch.nn.CrossEntropyLoss()
    hidden_size = 5  # need to experiment by changing this to 1 or 3 or 5 or 30
    net = Network(hidden_size).to(device)  # create the network
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    num_epochs = 200

    net.train()
    for i in range(num_epochs):
        running_loss = 0
        for batch_idx, (x, y) in enumerate(data_loader):
            x = x.to(device).type(torch.float32)
            y = y.to(device).type(torch.float32)
            # use torch.long if using cross entropy loss
            # y = y.to(device).type(torch.long)
            
            optimizer.zero_grad()  # clear the gradients
            outputs = net(x)  # calls forward function in the Network
            loss = loss_fn(outputs, y)
            loss.backward()  # compute gradients
            optimizer.step()  # update weights and biases
            running_loss += loss.item()
        
        print('epoch:', i, ' loss=', running_loss)

    # ------------ compute accuracy ----------
    net.eval()
    accuracy = 0
    data_loader = Utils.get_data_loader(batch_size=200)
    diter = iter(data_loader)
    x, y = next(diter)
    x = x.to(device).type(torch.float32)
    y = y.to(device).type(torch.long)
    
    out = net(x)
    Z = np.array(out.cpu().detach())
    
    y = np.argmax(y.cpu().detach(), axis=1)  # when using MSE loss, need to comment this line if using CrossEntropyLoss
    Z = np.argmax(Z, axis=1)
    
    y_np = np.array(y.cpu().detach())
    count = (Z == y_np).sum()
    accuracy_percent = count / len(y_np)
    
    print('accuracy =', accuracy_percent)
    Utils.plot_decision_boundary(net, x.cpu().detach(), y.detach().cpu())

if __name__ == "__main__":
    sys.exit(int(main() or 0))

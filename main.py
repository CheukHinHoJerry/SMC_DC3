import numpy as np
import pandas as pd
import xarray as xr
from unet.unet_model import ResNetUNet, UNet
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
import torch.optim as optim

class DataSet(torch.utils.data.Dataset):
    def __init__(self, dataSet_x, dataSet_y):
        self.dataSet_x = dataSet_x
        self.dataSet_y = dataSet_y

    def __len__(self):
        return len(self.dataSet_x)

    def __getitem__(self, index):
        return (self.dataSet_x[index], self.dataSet_y[index])

def load_data(file_path):
    ds_sample = xr.load_dataset(file_path)
    print('Coordinate inforomation: ', ds_sample.dims)
    print(ds_sample)
    return ds_sample


def plot_samples(ds):
    ds['vel'].mean('time').plot()
    ds['vel'].isel(time=range(9)).plot(col='time', col_wrap=3)
    fig, ax = plt.subplots(figsize=(15, 12))
    time_id = 10
    xy_resample = np.array(range(0, 96, 4)) + 2
    ds['vel'].isel(time=time_id).plot(ax=ax)
    plt.show()
    (ds.isel(time=time_id, xf=xy_resample, yf=xy_resample).plot.quiver(x='xf', y='yf', u='u', v='v', ax=ax,
                                                                       pivot='tail', color='white'))
    ax.set_aspect('equal')
    plt.show()

# TODO: If the model train too slowly, you may try to use the AMP (automatic mixed precision)
def train(net, low_res_data, high_res_data, batch_size, lr, device, validation_ratio, patience = 10):
    data_input = torch.unsqueeze(torch.tensor(low_res_data['u'].data), 1)
    data_output = torch.unsqueeze(torch.tensor(high_res_data['u'].data), 1)
    for var in ['v', 'vel', 'std', 'temp', 'absolute_height']:
        data_input = torch.cat((data_input, torch.unsqueeze(torch.tensor(low_res_data[var].data), 1)), dim = 1)
        data_output = torch.cat((data_output, torch.unsqueeze(torch.tensor(high_res_data[var].data), 1)), dim = 1)
    train_input, valid_input, train_target, valid_target = train_test_split(data_input, data_output,
                                                                                      train_size = 1 - validation_ratio)
    print("shape of train input, valid_input, train_target, valid_target: ", train_input.shape, valid_input.shape, train_target.shape, valid_target.shape)
    TrainDataSet = DataSet(train_input, train_target)
    ValidDataSet = DataSet(valid_input, valid_target)
    trainloader = torch.utils.data.DataLoader(TrainDataSet, batch_size=batch_size, pin_memory=True)
    validloader = torch.utils.data.DataLoader(ValidDataSet, batch_size=batch_size, pin_memory=True)
    num_of_epoch = 2000
    min_val_loss = np.inf
    min_percentage_error = np.inf
    loss_func = torch.nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 30, 0.5)
    time_start = torch.cuda.Event(enable_timing=True)
    time_end = torch.cuda.Event(enable_timing=True)
    for epoch in range(num_of_epoch):
        time_start.record()
        epoch_loss = 0
        stop = 0
        for data in trainloader:
            inputs, target = data
            if torch.cuda.is_available():
                inputs, target = inputs.to(device), target.to(device)
            # Forward Pass
            pred = net(inputs)
            loss = loss_func(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # record loss
            train_loss = loss.item() * len(inputs)
            epoch_loss += train_loss

        val_loss = 0
        net.eval()  # Optional when not using Model Specific layer

        for data in validloader:
            inputs, target = data
            if torch.cuda.is_available():
                inputs, target = inputs.to(device), target.to(device)
            # Forward Pass
            with torch.no_grad():
                valid_pred = net(inputs).to(device)

                # record  loss
                loss = loss_func(valid_pred, target)
                val_loss = val_loss + loss.item() * len(inputs)

        scheduler.step()

        print("Epoch", epoch + 1)
        print("Training loss: ", epoch_loss / len(trainloader), "Validation Loss: ",
              val_loss / len(validloader))

        if min_val_loss > val_loss:
            print(
                f'__Validation Loss Decreased({min_val_loss / len(validloader):.6f}--->{val_loss / len(validloader):.6f}) \t Saving The Model__')
            min_val_loss = val_loss
            # Saving State Dict
            torch.save(net.state_dict(),
                       './trained_model/trial_model.pth')

        else:
            stop += 1
            if stop >= patience:
                print("early stop")
                return
        print("current min validation percentage error = ", min_percentage_error)
        time_end.record()
        print("")


def main():
    # Loading ERA5 data
    ds_era5 = xr.load_dataset('data/perdigao_era5_2020.nc')
    print(ds_era5['u100'].shape)
    ds_era5['vel100'] = np.sqrt(ds_era5['u100'] ** 2 + ds_era5['v100'] ** 2)
    ds_era5['vel100'].attrs = {'long_name': '100 meter horizontal wind speed', 'units': 'm/s'}
    print(ds_era5['vel100'].shape)
    # Loading LES data into memory
    ds_low_res_sample = load_data('data_samples/perdigao_low_res_1H_2020_01.nc')
    ds_high_res_sample = load_data('data_samples/perdigao_high_res_1H_2020_01.nc')
    plot_samples(ds_low_res_sample)
    # Change here to adapt to your data, here the ResUnet is used for segmentation, for the current challenge, n_channels = n_class since the input and output channel of data should be the same
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNetUNet(n_channels=6, n_class=6)
    model.to(device=device)
    train(model, ds_low_res_sample, ds_high_res_sample, batch_size=32, lr = 0.01, device = device, validation_ratio = 0.05)

if __name__ == '__main__':
    main()

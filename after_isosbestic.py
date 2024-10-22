import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from matplotlib import gridspec
from sklearn.metrics import *
import matplotlib.patches as patches
import torch
import torch.nn as nn
import torch.optim as optim

class afterisosbestic:
    def __init__(self, time, absorptions):
        self.wavelengths = torch.tensor(absorptions[:, 0]) 
        self.absorptions =  np.array(absorptions[:, 1:])
        self.time = torch.tensor(time).reshape(1, -1).float()
                
        minimum = np.where(self.absorptions.min(axis=0)>=0, 0., self.absorptions.min(axis=0))
        self.absorptions -= minimum # Correcting negative absorptions
        self.absorptions = torch.tensor(self.absorptions)

        self.a = torch.tensor(1, requires_grad=True, dtype=torch.float32) # allowing a adjustment in the reagent spectrum, to avoid insolublility errors
        self.abs1 = self.absorptions[:, 0].reshape(-1, 1).clone().detach().requires_grad_(False) # Spectrum of the reagent = Spectrum at time 0 sec Irradiation
        self.abs2 = (self.absorptions[:, -1]).reshape(-1, 1).clone().detach().requires_grad_(True) # Spectrum of the product
        self.k = torch.tensor(0.01, requires_grad=True, dtype=torch.float32) # Degradation kinetic constant of the reagent 
        self.k1 = torch.tensor(0.001, requires_grad=True, dtype=torch.float32) # Degradation kinetic constant of the products occuring at later times 
        
        self.loss_list = []
        
    def __str__(self):
        k = self.k.detach().numpy()
        k1 = self.k1.detach().numpy()

        string = "First order product Model:\n"
        string += f"\tk = {k:.5f}\n"  
        string += f"\tk1 = {k1:.5f}\n"  
        string += f"Metrics:\n"
        string += f"\tR_Square: {self.R_Square():.5f}\n"  
        string += f"\tAdjusted R_Square: {self.Adj_R_Square():.5f}"  
        return string    
    
    def forward(self, time): # Total behavior of the solution during the Irradiation
        abspred = self.a * self.abs1 * torch.exp(-self.k * time) + self.abs2 * (1 - torch.exp(-self.k * time))*torch.exp(-self.k1 * time)
        return abspred
    
    def forward_reagent(self, time): # Behavior of the Degradation of the reagent
        abspred = self.a * self.abs1 * torch.exp(-self.k * time)
        return abspred
    
    def forward_product(self, time): # Behavior of the formation of the product(s) and possible degradation of them
        abspred =  self.abs2 * (1 - torch.exp(-self.k * time))*torch.exp(-self.k1 * time)
        return abspred
    
    def fit(self, epochs = 5_000, lr=1e-3):
        
        MSE = nn.MSELoss(reduction='sum')
        optimizer = optim.Adam([self.a, self.k, self.abs2, self.k1], lr = lr)

        for epoch in range(epochs):
            optimizer.zero_grad() # Make zero the partial derivates in each epoch
            abspred = self.forward(self.time) # Forward pass
            mse_loss = MSE(abspred, self.absorptions) # Calculate and print Mean Squared Error (MSE)
            self.loss_list.append(mse_loss.detach())
            mse_loss.backward() # Backward pass (compute the partial derivates for this epoch)            
            optimizer.step() # Update the model's parameters using a gradient descent family of methods (example Adam, Stochastic gradient descent etc.)
            if epoch % 1000 == 0: 
                print(f'Epoch = {epoch}, MSE = {mse_loss.item()}')

                
    def plot_model(self, λ, save=False): # function to plot the behaviour on a specific wavelength of the spectrum
        mpl.rc('font', family='Times New Roman')

        font = {'color': 'maroon',
                'weight': 'normal',
                'size': 14
                }

        fig, ax = plt.subplots()  # Create a new figure and axis
        x = torch.tensor(np.arange(0, self.time[0][-1], 0.1))
        val = int(2*(λ-self.wavelengths[0]))
        pred = self.forward(x).detach().numpy()
        reagent = self.forward_reagent(x).detach().numpy()
        product = self.forward_product(x).detach().numpy()
        
        ax.plot(x, pred[val,:], color="maroon", linewidth=1, label='Solution\'s absorbance')
        ax.plot(x, reagent[val,:], color="#000080", label='Reagent\'s absorbance')
        ax.plot(x, product[val,:], color="#808000", label='Product\'s absorbance')

        ax.scatter(self.time, self.absorptions[val,:], color="black", marker="o")
        
        plt.title(f'Solution\'s absorbance at λ = {λ}nm', fontsize=18, y=1.06, color="maroon", x = 0.75)
        plt.legend(fontsize=11)
        ax.set_ylabel('Absorbance', labelpad=14, fontdict=font)
        ax.set_xlabel("Time (sec)", labelpad=14, fontdict=font)
        ax.grid(color="gray", linestyle='--', linewidth=0.7, alpha=1)
        ax.tick_params(axis='both', which='both', labelsize=12)        

        # Create the table
        ax_table = ax.inset_axes([1.04, 0.2, 0.5, 0.6], transform=ax.transAxes)
        ax_table.axis('off')  # Hide axes
        data = np.concatenate((self.time.reshape(-1,1), self.absorptions[val,:].reshape(-1,1)), axis=1)
        df = pd.DataFrame(data=data, columns=["Time (sec)", "Absorbance"])
        table = ax_table.table(cellText=np.round(df.values,3), colLabels=["Time (sec)", "Absorbance"], cellLoc='center', loc='center')
        table.scale(1, 2)
        table.auto_set_font_size(False)
        table.set_fontsize(13)

        for cell in table.get_celld().values():
            cell.set_fontsize(13) 
            
        if save:
            plt.savefig(f'{save}.png', bbox_inches='tight')
            
        
    def plot_predictions(self, x=None): # Plotting predictions
        if x is None:
            x = self.time[0]
        preds = self.forward(x)
        plt.plot(self.wavelengths, preds.detach().numpy(), label=x.detach().numpy())
        plt.legend()
        plt.xlim(self.wavelengths.min(), self.wavelengths.max())
        plt.ylim(ymin=0)
        
    def plot_data(self): # Plotting experimental data
        plt.plot(self.wavelengths, self.absorptions, label=self.time[0].detach().numpy())
        plt.legend()
        plt.xlim(self.wavelengths.min(), self.wavelengths.max())
        plt.ylim(ymin=0)
        
    def plot_reagent(self, x=None): # Plotting data for the degradation of the reagent
        if x is None:
            x = self.time[0]
        preds = self.forward_reagent(x)
        plt.xlim(self.wavelengths.min(), self.wavelengths.max())
        plt.ylim(ymin=0)
        plt.plot(self.wavelengths, preds.detach().numpy(), label=x.detach().numpy())
        plt.legend()
        
    def plot_product(self, x=None): # Plotting predictions for the formation of product
        if x is None:
            x = self.time[0]
        preds = self.forward_product(x)
        plt.xlim(self.wavelengths.min(), self.wavelengths.max())
        plt.ylim(ymin=0)
        plt.plot(self.wavelengths, preds.detach().numpy(), label=x.detach().numpy())
        plt.legend()
        
    def plot_loss(self): # Plotting loss per epoch
        plt.plot(range(0,len(self.loss_list)), self.loss_list);
        
    def loss_per_pred(self): # Average lose per prediction
        preds = self.forward(self.time)
        loss = np.abs(preds.detach().numpy() - self.absorptions.detach().numpy())
        mean_loss = np.mean(loss)
        return mean_loss
    
    def loss_per_wv(self): # Average lose per prediction per wavelenght
        preds = self.forward(self.time)
        loss = np.abs(preds.detach().numpy() - self.absorptions.detach().numpy())
        mean_loss = np.mean(loss, axis=1)
        plt.plot(self.wavelengths, mean_loss)
        mean_loss = np.concatenate((self.wavelengths.detach().numpy().reshape(-1,1), mean_loss.reshape(-1,1)), axis=1)
        return mean_loss
        
    def R_Square(self):
        pred = self.forward(self.time)
        self.r_Square = r2_score(self.absorptions, pred.detach().numpy())
        return self.r_Square
    
    def Adj_R_Square(self):
        k = 1 # predictor, pH
        self.R_Square()
        self.Adj_r_Square = 1 - ((1 - self.r_Square) * (len(self.absorptions) - 1) / (len(self.absorptions) - k - 1))
        return self.Adj_r_Square
import torch
import torch.nn as nn

class TARnetICFR(nn.Module):
    def __init__(self, input_dim, reg_l2, hidden_dim):
        super().__init__()
        
        self.phi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )

        self.y0_hidden = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            #nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1)
        )

        self.y1_hidden = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward_treated(self, treated):
        phi = self.phi(treated)
        #y0 = self.y0_hidden(phi)
        y1 = self.y1_hidden(phi)
        #out = torch.cat((y0, y1), dim=1)
        return y1
    
    def forward_control(self, control):
        phi = self.phi(control)
        #y0 = self.y0_hidden(phi)
        y0 = self.y0_hidden(phi)
        #out = torch.cat((y0, y1), dim=1)
        return y0
    
    def forward_ate(self, control, treated):
        mu1 = torch.mean(self.forward_treated(control))
        mu0 = torch.mean(self.forward_control(treated))
        tau_est = mu1 - mu0
        
        return tau_est
    
    def forward(self, x):  
        
        control = x[x[:,-1]==0]
        treated = x[x[:,-1]==1]
        y1 = self.forward_treated(treated)
        y0 = self.forward_control(control)
        tau_est = self.forward_ate(control, treated)
                
        return y1, y0, tau_est
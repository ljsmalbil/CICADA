import torch
import torch.nn as nn
import torch.nn.functional as F

class CNDT_ICFR(nn.Module):
    def __init__(self, depth, num_features, dropout_rate, hidden_size, used_features_rate, device, l1_coefficient=0.01):
        super().__init__()
        
        self.depth = depth
        self.device = device
        self.dropout_rate = dropout_rate
        self.hidden_size = hidden_size
        self.num_leaves = 2 ** depth

        # Create a mask for the randomly selected features.
        num_used_features = int(num_features * used_features_rate)

        one_hot = torch.eye(num_features)
        sampled_feature_indices = torch.randperm(num_features)[:num_used_features]
        self.used_features_mask = one_hot[sampled_feature_indices]
        self.used_features_mask = self.used_features_mask.to(self.device)

        # Initialize the weights of the classes in leaves.
        self.pi = nn.Parameter(torch.randn(self.num_leaves))

        
        self.decision_fn_t = nn.Sequential(
            nn.Linear(in_features=num_used_features, out_features=hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=hidden_size, out_features=self.num_leaves))

        self.decision_fn_c = nn.Sequential(
            nn.Linear(in_features=num_used_features, out_features=hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=hidden_size, out_features=self.num_leaves))

    def forward_treated(self, features_treated):
        features = features_treated
        batch_size = features.shape[0]

        # Apply the feature mask to the input features.
        features = torch.matmul(features, self.used_features_mask.t())  # [batch_size, num_used_features]
        # Compute the routing probabilities.
        decisions = F.sigmoid(self.decision_fn_t(features))  # [batch_size, num_leaves]
        decisions = decisions.view(batch_size, self.num_leaves, 1)  # [batch_size, num_leaves, 1]
        # Concatenate the routing probabilities with their complements.
        decisions = torch.cat((decisions, 1 - decisions), dim=2)  # [batch_size, num_leaves, 2]

        mu = torch.ones(batch_size, 1, 1).to(self.device)

        begin_idx = 1
        end_idx = 2
        # Traverse the tree in breadth-first order.
        for level in range(self.depth):
            mu = mu.view(batch_size, -1, 1)  # [batch_size, 2 ** level, 1]
            mu = mu.repeat(1, 1, 2)  # [batch_size, 2 ** level, 2]
            level_decisions = decisions[:, begin_idx:end_idx, :]  # [batch_size, 2 ** level, 2]
            mu = mu * level_decisions  # [batch_size, 2**level, 2]
            begin_idx = end_idx
            end_idx = begin_idx + 2 ** (level + 1)

        mu = mu.view(batch_size, self.num_leaves)  # [batch_size, num_leaves]
        outputs = torch.matmul(mu, self.pi)  # [batch_size, 1]
        
        return outputs 
    
    def forward_control(self, features_control):
        features = features_control
        batch_size = features.shape[0]

        # Apply the feature mask to the input features.
        features = torch.matmul(features, self.used_features_mask.t())  # [batch_size, num_used_features]
        # Compute the routing probabilities.
        decisions = F.sigmoid(self.decision_fn_c(features))  # [batch_size, num_leaves]
        decisions = decisions.view(batch_size, self.num_leaves, 1)  # [batch_size, num_leaves, 1]
        # Concatenate the routing probabilities with their complements.
        decisions = torch.cat((decisions, 1 - decisions), dim=2)  # [batch_size, num_leaves, 2]

        mu = torch.ones(batch_size, 1, 1).to(self.device)

        begin_idx = 1
        end_idx = 2
        # Traverse the tree in breadth-first order.
        for level in range(self.depth):
            mu = mu.view(batch_size, -1, 1)  # [batch_size, 2 ** level, 1]
            mu = mu.repeat(1, 1, 2)  # [batch_size, 2 ** level, 2]
            level_decisions = decisions[:, begin_idx:end_idx, :]  # [batch_size, 2 ** level, 2]
            mu = mu * level_decisions  # [batch_size, 2**level, 2]
            begin_idx = end_idx
            end_idx = begin_idx + 2 ** (level + 1)

        mu = mu.view(batch_size, self.num_leaves)  # [batch_size, num_leaves]
        outputs = torch.matmul(mu, self.pi)  # [batch_size, 1]
        
        return outputs 

    def forward_ate(self, features_control, features_treated):
        mu1 = torch.mean(self.forward_treated(features_control))
        mu0 = torch.mean(self.forward_control(features_treated))
        tau_est = mu1 - mu0
        
        return tau_est

    def forward(self, features_treated, features_control):        
        output_treated = self.forward_treated(features_treated)
        output_control = self.forward_control(features_control)
        tau_est = self.forward_ate(features_control, features_treated)
                
        return output_treated, output_control, tau_est
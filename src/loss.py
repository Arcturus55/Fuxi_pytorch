import torch
from torch import nn
import torch.nn.functional as F

class MAELossForMultiLabel(nn.Module):
    """ MAELossForMultiLabel definition """

    def __init__(self, batch_size=1, h_size=720, w_size=1440, feature_dims=69, level_feature_size=5, pressure_level_num=13, loss_weight=0.25):
        super(MAELossForMultiLabel, self).__init__()
        
        self.batch_size = batch_size
        self.h_size = h_size
        self.w_size = w_size
        self.feature_dims = feature_dims
        self.level_feature_size = level_feature_size
        self.pressure_level_num = pressure_level_num
        self.surface_feature_size = self.feature_dims - self.level_feature_size * self.pressure_level_num
        self.loss_weight = loss_weight

    def get_loss(self, x):
        """
        Computes the loss.
        Args:
            x (Tensor)
        Returns:
            Return the loss.
        """
        y = x.float()
        y = y.mean()
        y = y.type_as(x)
        return y

    def forward(self, x, x_surface, labels):
        """MAELossForMultiLabel forward function."""
        label = labels[..., :-self.surface_feature_size]
        label_surface = labels[..., -self.surface_feature_size:]
        label = label.reshape(self.batch_size, self.h_size, self.w_size, self.level_feature_size,
                              self.pressure_level_num)
        label = label.permute(0, 3, 4, 1, 2)
        label_surface = label_surface.reshape(self.batch_size, self.h_size, self.w_size, -1)
        label_surface = label_surface.permute(0, 3, 1, 2)
        x1 = torch.abs(x - label)
        x2 = torch.abs(x_surface - label_surface)
        loss_x = self.get_loss(x1)
        loss_surface = self.get_loss(x2)
        loss = loss_x + self.loss_weight * loss_surface
        return loss
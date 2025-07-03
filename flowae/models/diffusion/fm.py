import torch

from models import register


@register('fm')
class FM:
    
    def __init__(self, sigma_min=1e-5, timescale=1.0):
        self.sigma_min = sigma_min
        self.prediction_type = None
        self.timescale = timescale
    
    def alpha(self, t):
        return 1.0 - t
    
    def sigma(self, t):
        return self.sigma_min + t * (1.0 - self.sigma_min)
    
    def A(self, t):
        return 1.0
    
    def B(self, t):
        return -(1.0 - self.sigma_min)

    def _get_reduction_dims(self, x):
        """Get appropriate dimensions for loss reduction based on tensor shape"""
        if x.dim() == 4:
            # Images: [batch, channels, height, width]
            return [1, 2, 3]
        elif x.dim() == 3:
            # Audio: [batch, channels, samples] or [batch, latent_dim, time_frames]
            return [1, 2]
        elif x.dim() == 2:
            # 1D signals: [batch, samples]
            return [1]
        else:
            # Fallback: reduce over all non-batch dimensions
            return list(range(1, x.dim()))
    
    def get_betas(self, n_timesteps):
        return torch.zeros(n_timesteps) # Not VP and not supported
    
    def add_noise(self, x, t, noise=None):
        noise = torch.randn_like(x) if noise is None else noise
        s = [x.shape[0]] + [1] * (x.dim() - 1)
        x_t = self.alpha(t).view(*s) * x + self.sigma(t).view(*s) * noise
        return x_t, noise
    
    def loss(self, net, x, t=None, net_kwargs=None, return_loss_unreduced=False, return_all=False):
        if net_kwargs is None:
            net_kwargs = {}
        
        if t is None:
            t = torch.rand(x.shape[0], device=x.device)
        # print('x shape: ', x.shape)
        x_t, noise = self.add_noise(x, t)
        # print('x_t shape: ', x_t.shape)
        pred = net(x_t, t=t * self.timescale, **net_kwargs)
        # print('pred shape: ', pred.shape)
        
        target = self.A(t) * x + self.B(t) * noise # -dxt/dt
        # print('target shape: ', target.shape)
        # print('return_loss_unreduced: ', return_loss_unreduced, 'return_all: ', return_all)
        if return_loss_unreduced:
            print('pred shape: ', pred.shape, 'target shape: ', target.shape)
            reduce_dims = self._get_reduction_dims(x)
            loss = ((pred.float() - target.float()) ** 2).mean(dim=reduce_dims)
            # loss = ((pred.float() - target.float()) ** 2).mean(dim=[1, 2, 3])
            if return_all:
                return loss, t, x_t, pred
            else:
                return loss, t
        else:
            # here we go
            loss = ((pred.float() - target.float()) ** 2).mean()
            if return_all:
                return loss, x_t, pred
            else:
                return loss
    
    def get_prediction(
        self,
        net,
        x_t,
        t,
        net_kwargs=None,
        uncond_net_kwargs=None,
        guidance=1.0,
    ):
        if net_kwargs is None:
            net_kwargs = {}
        pred = net(x_t, t=t * self.timescale, **net_kwargs)
        if guidance != 1.0:
            assert uncond_net_kwargs is not None
            uncond_pred = net(x_t, t=t * self.timescale, **uncond_net_kwargs)
            pred = uncond_pred + guidance * (pred - uncond_pred)
        return pred
    
    def convert_sample_prediction(self, x_t, t, pred):
        M = torch.tensor([
            [self.alpha(t), self.sigma(t)],
            [self.A(t), self.B(t)],
        ], dtype=torch.float64)
        M_inv = torch.linalg.inv(M)
        sample_pred = M_inv[0, 0].item() * x_t + M_inv[0, 1].item() * pred
        return sample_pred

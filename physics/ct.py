import torch
import numpy as np
from .radon import Radon, IRadon
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

class CT():
    def __init__(self, img_width, radon_view, uniform=True, circle=False, device='cuda:0'):
        if uniform:
            theta = np.linspace(0, 180, radon_view, endpoint=False)#稀疏角
            theta_all = np.linspace(0, 180, 180, endpoint=False)
        else:
            theta = torch.arange(radon_view)#有限角
            theta_all = torch.arange(radon_view)

        self.radon = Radon(img_width, theta, circle).to(device)
        self.radon_all = Radon(img_width, theta_all, circle).to(device)
        self.iradon_all = IRadon(img_width, theta_all, circle).to(device)
        self.iradon = IRadon(img_width, theta, circle).to(device)
        self.radont = IRadon(img_width, theta, circle, use_filter=None).to(device)

    def A(self, x):
        return self.radon(x)

    def A_all(self, x):
        return self.radon_all(x)

    def A_all_dagger(self, x):
        return self.iradon_all(x)

    def A_dagger(self, y):
        return self.iradon(y)

    def AT(self, y):
        return self.radont(y)
    
    def degradation_fn_fbp(self,x):
        sinogram = self.A(x)
        bp = self.AT(sinogram)
        return bp
    
    def Getbp(self,x):
        sinogram = self.A(x)
        bp = self.AT(sinogram)
        return bp
    
    def Getfbp(self,x):
        sinogram = self.A(x)
        fbp = self.A_dagger(sinogram)
        return fbp


class CT_LA():
    """
    Limited Angle tomography
    """
    def __init__(self, img_width, radon_view, uniform=True, circle=False, device='cuda:0'):
        if uniform:
            theta = np.linspace(0, 180, radon_view, endpoint=False)
        else:
            theta = torch.arange(radon_view)
        self.radon = Radon(img_width, theta, circle).to(device)
        self.iradon = IRadon(img_width, theta, circle).to(device)
        self.radont = IRadon(img_width, theta, circle, use_filter=None).to(device)

    def A(self, x):
        return self.radon(x)

    def A_dagger(self, y):
        return self.iradon(y)

    def AT(self, y):
        return self.radont(y)

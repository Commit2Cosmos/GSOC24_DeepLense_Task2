import torch
from torch import nn
import torch.nn.functional as F


def gkern2d(l=21, sig=3, device='cpu'):
    """Returns a 2D Gaussian kernel array."""
    ax = torch.arange(-l // 2 + 1., l // 2 + 1., device=device)
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2. * sig ** 2))
    return kernel


class Shift(nn.Module):
    def __init__(self, in_planes, kernel_size=3):
        super(Shift, self).__init__()
        self.in_planes = in_planes
        self.kernel_size = kernel_size
        self.channels_per_group = self.in_planes // (self.kernel_size ** 2)
        if self.kernel_size == 3:
            self.pad = 1
        elif self.kernel_size == 5:
            self.pad = 2
        elif self.kernel_size == 7:
            self.pad = 3

    def forward(self, x):
        c, h, w = x.size()
        x_pad = F.pad(x, (self.pad, self.pad, self.pad, self.pad))

        cat_layers = []

        for i in range(self.in_planes):
            # Parse in row-major
            for y in range(0, self.kernel_size):
                y2 = y+h
                for x in range(0, self.kernel_size):
                    x2 = x+w
                    xx = x_pad[i:i+1,y:y2,x:x2]
                    cat_layers += [xx]
        return torch.cat(cat_layers, 0)


class BilateralFilter(nn.Module):
    """BilateralFilter computes:
        If = 1/W * Sum_{xi C Omega}(I * f(||I(xi)-I(x)||) * g(||xi-x||))
    """

    def __init__(self, channels=3, k=7, height=101, width=101, sigma_space=10, sigma_color=0.1, device='cpu'):
        super().__init__()

        # space gaussian kernel
        self.gw = gkern2d(k, sigma_space, device=device)

        self.g = torch.tile(self.gw.reshape(channels, k*k, 1, 1), (1, 1, height, width))
        # shift
        self.shift = Shift(channels, k)
        self.sigma_color = 2*sigma_color**2

        self.to(device=device)


    def forward(self, I):
        Is = self.shift(I).data
        Iex = I.expand(*Is.size())
        D = (Is-Iex)**2 # here we are actually missing some sum over groups of channels
        De = torch.exp(-D / self.sigma_color)
        Dd = De * self.g
        W_denom = torch.sum(Dd, dim=1)
        If = torch.sum(Dd*Is, dim=1) / W_denom
        return If


class EdgeDetectionTransform:
        def __init__(self, kernel_size=3, device='cpu'):
            self.kernel_size = kernel_size

            # Define edge detection kernel
            self.kernel = torch.tensor([[-1, -1, -1],
                                        [-1,  8, -1],
                                        [-1, -1, -1]], dtype=torch.float32).to(device)
            
            # self.kernel = torch.tensor([[0, -1, 0],
            #                             [-1,  4, -1],
            #                             [0, -1, 0]], dtype=torch.float32).to(device)

        def __call__(self, img: torch.Tensor):
            # Apply 2D convolution with edge detection kernel
            edge_tensor = torch.nn.functional.conv2d(img.unsqueeze(0),
                                                    self.kernel.unsqueeze(0).unsqueeze(0), 
                                                    padding=1)
            
            # Normalize the output tensor
            edge_tensor = torch.clamp(edge_tensor, 0, 1)  # Clip values to [0, 1]
            
            return edge_tensor.squeeze(0)
        


if __name__ == '__main__':
    import os
    import time
    import numpy as np
    import matplotlib.pyplot as plt

    from bm3d_denoise import BM3D_1st_step, BM3D_2nd_step


    k = 3
    device = 'cpu'

    root = "data/easy"

    imgs = np.load(os.path.join(root, "X_train.npy"))
    data = torch.tensor(imgs).transpose(1, 3)

    num_imgs = 6

    to_filter = torch.from_numpy(np.array([data[i][0].unsqueeze(0) for i in range(num_imgs)]))

    (dim1, dim2, dim3) = to_filter[0].shape

    apply_bilat = BilateralFilter(dim1, k, dim2, dim3, sigma_space=7, sigma_color=0.2, device="mps")

    start_time = time.time()

    filtered = []

    apply_edge_det = EdgeDetectionTransform(kernel_size=9, device='mps')


    for im in to_filter:
        # im: torch.Tensor = apply_bilat(im.to('mps'))
        # im = apply_edge_det(im.to(device='mps'))

        im: np.ndarray = im.squeeze(0).numpy()
        im = im / im.max() * 255


        im = BM3D_1st_step(im)
        # im = BM3D_2nd_step(basic_im, im)

        im = (im - np.min(im)) / (np.max(im) - np.min(im))
        # im: torch.Tensor = torch.from_numpy(im).to(device='mps', dtype=torch.float32)

        # im = apply_bilat(im.unsqueeze(0)).squeeze(0).to('cpu')
        filtered.append(im)


    print("Duration: ", time.time() - start_time)

    filtered = np.array(filtered)

    print(to_filter.shape)
    print(filtered.shape)

    to_plot = torch.concat((to_filter.squeeze(1), torch.from_numpy(filtered)))

    fig, axes = plt.subplots(2, num_imgs, sharex='all', sharey='all', figsize=(12, 9))
    plt.axis('off')

    axes = axes.flatten()

    for i, (ax, imim) in enumerate(zip(axes, to_plot)):
        ax.imshow(imim)

    plt.tight_layout()
    plt.show()
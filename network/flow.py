import numpy as np
import torch
import torch.nn as nn
from utils.network_utils import get_embedder, contraction
import nvdiffrast.torch as dr
from utils.base_utils import Timing


class FactorizedGaussianSampler(torch.nn.Module):
    def __init__(self, d, mu=0., sig=1.):
        super().__init__()
        sig_ = torch.tensor(sig).cuda()
        mu_ = torch.tensor(mu).cuda()
        prior_1d = torch.distributions.normal.Normal(mu_, sig_)
        self.d = d
        self.prior = prior_1d

    def log_prob(self, x):
        return torch.sum(self.prior.log_prob(x), -1, keepdim=True)

    def forward(self, shape):
        x = self.prior.sample((*shape, self.d))
        logj = -self.log_prob(x)
        return x, logj

class UniformSampler(torch.nn.Module):
    """Factorized uniform prior
    Note that tensorflow distribution objects cannot easily be moved devices so specify the right
    device at initialization.
    """
    def __init__(self, d, low=0., high=1., device='cuda'):
        super().__init__()
        # Copy data
        low_ = torch.tensor(low, dtype=torch.get_default_dtype())
        high_ = torch.tensor(high, dtype=torch.get_default_dtype())

        if device is not None:
            low_ = low_.to(device)
            high_ = high_.to(device)

        self.d = d
        self.prior = torch.distributions.Uniform(low_, high_)

    def log_prob(self, x):
        return torch.sum(self.prior.log_prob(x), -1, keepdim=True)

    def forward(self, shape):
        x = self.prior.sample((*shape, self.d))
        logj = -self.log_prob(x)
        return x, logj
    
class SphereSampler(torch.nn.Module):
    """
        sample angles from the sphere uniformly
        https://zhuanlan.zhihu.com/p/25988652
    """
    def __init__(self, d, device='cuda'):
        super().__init__()
        self.d = d
        self.angle = None
        
    def set_angle(self, num_samples=256):
        begin_elevation = 1
        ratio = (begin_elevation + 90) / 180
        num_points = int(num_samples // (1 - ratio))
        phi = (np.sqrt(5) - 1.0) / 2.
        phis = []
        thetas = []
        for n in range(num_points - num_samples, num_points):
            z = 2. * n / num_points - 1.
            phis.append(2 * np.pi * n * phi % (2 * np.pi))
            # PDF of theta is cos(theta) -> CDF is sin(theta)
            thetas.append(np.arcsin(z))
        phi = torch.tensor(phis, device="cuda", dtype=torch.float32) / (2 * np.pi)
        theta = torch.tensor(thetas, device="cuda", dtype=torch.float32) / (0.5 * np.pi)
        self.angle = torch.stack([phi, theta], dim=-1)

    def log_prob(self, x):
        # pdf of the first dimension is 1 -> log(1) = 0
        return torch.cos(x[..., 1:] * (0.5 * np.pi)).log()

    def forward(self, shape):
        if self.angle is None or self.angle.shape[0] != shape[1]:
            self.set_angle(shape[1])
        x = self.angle.expand(*shape, 2)
        if self.training:
            x = torch.cat([(x[..., :1] + torch.rand_like(x[..., 1:])) % 1, x[..., 1:]], dim=-1)
        x = x.clamp(1e-6, 1-1e-6)
        logj = -self.log_prob(x)
        return x, logj

class GGXSampler(torch.nn.Module):
    """
        sample angles using GGX distribution
    """
    def __init__(self, d, device='cuda'):
        super().__init__()
        self.d = d
        self.device=device
        self.a = 0.2 ** 2  # squared roughness
        
    def log_prob(self, x):
        a2 = self.a ** 2
        cos_theta2 = x[..., 1:]
        pdf = a2 / (cos_theta2 * (a2 - 1) + 1) ** 2
        return pdf.clamp_min(1e-6).log()

    def forward(self, shape):
        angle = torch.rand(*shape, 2, device=self.device)
        e_phi, e_theta = torch.split(angle, 1, dim=-1)
        
        a2 = self.a ** 2
        cos_theta = ((1.0 - e_theta) / (1.0 + (a2 - 1.0) * e_theta).clamp_min(1e-6)).clamp_min(1e-6).sqrt() # pn,sn,1
        
        cos_theta2 = cos_theta ** 2
        x = torch.cat([e_phi, cos_theta2], dim=-1)
        x = x.clamp(1e-6, 1-1e-6)
        
        logj = -self.log_prob(x)
        return x, logj


def inverse_sigmoid(x):
    return torch.log((x / (1 - x)).clamp_min(1e-6))

class InvertibleAnalyticSigmoid(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def flow(self, x, logj, feature, return_jacobian=False):
        y = torch.sigmoid(x).clamp(1e-6, 1-1e-6)
        if return_jacobian:
            logj = logj + torch.sum(torch.log((y * (1 - y)).clamp_min(1e-6)), dim=-1, keepdim=True)
            return y, logj
        else:
            return y

    def flow_inv(self, x, logj, feature, return_jacobian=False):
        y = inverse_sigmoid(x)
        if return_jacobian:
            logj = logj - torch.sum(torch.log((x * (1 - x)).clamp_min(1e-6)), dim=-1, keepdim=True)
            return y, logj
        else:
            return y
        
class Reshift(torch.nn.Module):
    """Un-trainable activation: shift and scale data

    The input x is transformed as x*scale + offset. Scale and offset are untrainable scalars.
    """
    def __init__(self, scale=2., offset=-1.):
        """

        Parameters
        ----------
        scale: float
        offset: float
        """
        super(Reshift, self).__init__()
        self.scale = torch.nn.Parameter(data=torch.scalar_tensor(scale), requires_grad=False)
        self.offset = torch.nn.Parameter(data=torch.scalar_tensor(offset), requires_grad=False)

    def forward(self, x):
        return x*self.scale + self.offset

def modified_softmax(v,w):
    v=torch.exp(v)
    return (v / ((v[:,:,:-1]+v[:,:,1:])/2 * w).sum(-1, keepdim=True))
    # torch.mul((v[:,:,:-1]+v[:,:,1:])/2,w)
    # vnorms=torch.cumsum(torch.mul((v[:,:,:-1]+v[:,:,1:])/2,w),axis=-1)
    # vnorms_tot=vnorms[:, :, -1].clone() 
    # return torch.div(v,torch.unsqueeze(vnorms_tot,axis=-1)) 


class ElementWisePWLinearTransform:
    """Invertible piecewise-quadratic transformations over the unit hypercube

    Implements a batched bijective transformation `h` from the d-dimensional unit hypercube to itself,
    in an element-wise fashion (each coordinate transformed independently)

    In each direction, the bijection is a piecewise-quadratic transform with b bins
    where the forward transform has bins with adjustable width. The transformation in each bin is
    then a quadratic spline. The network predicts the bin width w_tilde and the vertex height v_tilde of the
    derivative of the transform for each direction and each point of the batch. They are normalized such that: 
    1. h(0) = 0
    2. h(1) = 1
    3. h is monotonous
    4. h is continuous

    Conditions 1. to 3. ensure the transformation is a bijection and therefore invertible.
    The inverse is also an element-wise, piece-wise quadratic transformation.
    """
    def flow_inv(self, x, q_tilde, return_jacobian=True):
        logj = None

        # TODO do a bottom-up assesment of how we handle the differentiability of variables

        # Compute the bin width w
        N, k, b = q_tilde.shape
        Nx, kx = x.shape
        assert N == Nx and k == kx, "Shape mismatch"

        w = 1. / b

        # Compute the normalized bin heights by applying a softmax function on the bin dimension
        q = 1. / w * torch.softmax(q_tilde, dim=2).clamp_min(1e-6)

        # x is in the mx-th bin: x \in [0,1],
        # mx \in [[0,b-1]], so we clamp away the case x == 1
        mx = torch.clamp(torch.floor(b * x), 0, b - 1).to(torch.long)
        # Need special error handling because trying to index with mx
        # if it contains nans will lock the GPU. (device-side assert triggered)
        if torch.any(torch.isnan(mx)).item() or torch.any(mx < 0) or torch.any(mx >= b):
            import pdb;pdb.set_trace()
            raise ValueError("NaN detected in PWLinear bin indexing")

        # We compute the output variable in-place
        out = x - mx * w  # alpha (element of [0.,w], the position of x in its bin

        # Multiply by the slope
        # q has shape (N,k,b), mxu = mx.unsqueeze(-1) has shape (N,k) with entries that are a b-index
        # gather defines slope[i, j, k] = q[i, j, mxu[i, j, k]] with k taking only 0 as a value
        # i.e. we say slope[i, j] = q[i, j, mx [i, j]]
        slopes = torch.gather(q, 2, mx.unsqueeze(-1)).squeeze(-1)
        out = out * slopes

        # Compute the integral over the left-bins.
        # 1. Compute all integrals: cumulative sum of bin height * bin weight.
        # We want that index i contains the cumsum *strictly to the left* so we shift by 1
        # leaving the first entry null, which is achieved with a roll and assignment
        q_left_integrals = torch.roll(torch.cumsum(q, 2) * w, 1, 2)
        q_left_integrals[:, :, 0] = 0

        # 2. Access the correct index to get the left integral of each point and add it to our transformation
        out = out + torch.gather(q_left_integrals, 2, mx.unsqueeze(-1)).squeeze(-1)

        # Regularization: points must be strictly within the unit hypercube
        # Use the dtype information from pytorch
        eps = torch.finfo(out.dtype).eps
        out = out.clamp(
            min=eps,
            max=1. - eps
        )
        if return_jacobian:
            # The jacobian is the product of the slopes in all dimensions
            logj = torch.sum(torch.log(slopes), dim=-1, keepdim=True)
            return out, logj
        else:
            return out

    def flow(self, y, q_tilde, return_jacobian=True):
        # Compute the bin width w
        N, k, b = q_tilde.shape
        Ny, ky = y.shape
        assert N == Ny and k == ky, "Shape mismatch"

        w = 1. / b

        # Compute the normalized bin heights by applying a softmax function on the bin dimension
        q = 1. / w * torch.softmax(q_tilde, dim=2).clamp_min(1e-6)

        # Compute the integral over the left-bins in the forward transform.
        # 1. Compute all integrals: cumulative sum of bin height * bin weight.
        # We want that index i contains the cumsum *strictly to the left* so we shift by 1
        # leaving the first entry null, which is achieved with a roll and assignment
        q_left_integrals = torch.roll(torch.cumsum(q, 2) * w, 1, 2)
        q_left_integrals[:, :, 0] = 0

        # We can figure out which bin each y belongs to by finding the smallest bin such that
        # y - q_left_integral is positive

        edges = (y.unsqueeze(-1) - q_left_integrals).detach()
        # y and q_left_integrals are between 0 and 1 so that their difference is at most 1.
        # By setting the negative values to 2., we know that the smallest value left
        # is the smallest positive
        edges[edges < 0] = 2.
        edges = torch.clamp(torch.argmin(edges, dim=2), 0, b - 1).to(torch.long)

        # Need special error handling because trying to index with mx
        # if it contains nans will lock the GPU. (device-side assert triggered)
        if torch.any(torch.isnan(edges)).item() or torch.any(edges < 0) or torch.any(edges >= b):
            import pdb;pdb.set_trace()
            raise ValueError("NaN detected in PWLinear bin indexing")

        # Gather the left integrals at each edge. See comment about gathering in q_left_integrals
        # for the unsqueeze
        q_left_integrals = q_left_integrals.gather(2, edges.unsqueeze(-1)).squeeze(-1)

        # Gather the slope at each edge.
        q = q.gather(2, edges.unsqueeze(-1)).squeeze(-1)

        # Build the output
        x = (y - q_left_integrals) / q + edges * w

        # Regularization: points must be strictly within the unit hypercube
        # Use the dtype information from pytorch
        eps = torch.finfo(x.dtype).eps
        x = x.clamp(
            min=eps,
            max=1. - eps
        )

        # Prepare the jacobian
        logj = None
        if return_jacobian:
            logj = - torch.sum(torch.log(q), dim=-1, keepdim=True)
            if torch.isnan(x).any() or torch.isnan(logj).any():
                import pdb;pdb.set_trace()
            return x, logj
        else:
            return x
        

class ElementWisePWQuadraticTransform:
    """Invertible piecewise-quadratic transformations over the unit hypercube

    Implements a batched bijective transformation `h` from the d-dimensional unit hypercube to itself,
    in an element-wise fashion (each coordinate transformed independently)

    In each direction, the bijection is a piecewise-quadratic transform with b bins
    where the forward transform has bins with adjustable width. The transformation in each bin is
    then a quadratic spline. The network predicts the bin width w_tilde and the vertex height v_tilde of the
    derivative of the transform for each direction and each point of the batch. They are normalized such that: 
    1. h(0) = 0
    2. h(1) = 1
    3. h is monotonous
    4. h is continuous

    Conditions 1. to 3. ensure the transformation is a bijection and therefore invertible.
    The inverse is also an element-wise, piece-wise quadratic transformation.
    """
    def flow_inv(self, x, wv_tilde, return_jacobian=True):
        logj = None

        # TODO do a bottom-up assesment of how we handle the differentiability of variables
        
        v_tilde=wv_tilde[:,:,:int(np.ceil(wv_tilde.shape[2]/2))]
        w_tilde=wv_tilde[:,:,v_tilde.shape[2]:]
        N, k, b = w_tilde.shape
        Nx, kx = x.shape
        assert N == Nx and k == kx, "Shape mismatch"
        
        w=torch.exp(w_tilde).clamp_min(1e-6)
        wsum = torch.cumsum(w, axis=-1) 
        wnorms = torch.unsqueeze(wsum[:, :, -1], axis=-1) 
        w = (w/wnorms).clamp_min(1e-6)
        wsum=wsum/wnorms
        wsum_shift=torch.cat((torch.zeros([wsum.shape[0],wsum.shape[1],1]).to(wsum.device, wsum.dtype),wsum),axis=-1)
        
        v=modified_softmax(v_tilde, w).clamp_min(1e-6)
        
        #tensor of shape (N,k,b+1) with 0 entry if x is smaller than the cumulated w and 1 if it is bigger
        #this is used to find the bin with the number mx in which x lies; for this, the sum of the bin 
        #widths w has to be smaller to the left and bigger to the right
        finder=torch.where(wsum>torch.unsqueeze(x,axis=-1),torch.zeros_like(wsum),torch.ones_like(wsum))
        eps = torch.finfo(wsum.dtype).eps
        #the bin number can be extracted by finding the last index for which finder is nonzero. As wsum 
        #is increasing, this can be found by searching for the maximum entry of finder*wsum. In order to 
        #get the right result when x is in the first bin and finder is everywhere zero, a small first entry 
        #is added
        mx=torch.unsqueeze(  #we need to unsqueeze for later operations
            torch.argmax( #we search for the maximum in order to find the last bin for which x was greater than wsum
                torch.cat((torch.ones([wsum.shape[0],wsum.shape[1],1]).to(wsum.device, wsum.dtype)*eps,finder*wsum),
                        axis=-1),  #we add an offset to ensure that if x is in the first bin, a maximal argument is found 
                axis=-1), 
            axis=-1)
    
        # x is in the mx-th bin: x \in [0,1],
        # mx \in [[0,b-1]], so we clamp away the case x == 1
        mx = torch.clamp(mx, 0, b - 1).to(torch.long)
        # Need special error handling because trying to index with mx
        # if it contains nans will lock the GPU. (device-side assert triggered)
        if torch.any(torch.isnan(mx)).item() or torch.any(mx < 0) or torch.any(mx >= b):
            import pdb;pdb.set_trace()
            raise ValueError("NaN detected in PWQuad bin indexing")
        
        # alpha (element of [0.,1], the position of x in its bin)
        # gather collects the cumulated with of all bins until the one in which x lies
        # alpha=(x- Sum_(k=0)^(i-1) w_k)/w_b for x in bin b
        alphas=torch.div((x-torch.squeeze(torch.gather(wsum_shift,-1,mx),axis=-1)),  torch.squeeze(torch.gather(w,-1,mx),axis=-1)).clamp(0, 1)
        
        #vw_i= (v_i+1 - v_i)w_i/2 where i is the bin index
        vw=torch.cat((torch.zeros([v.shape[0],v.shape[1],1]).to(wsum.device, wsum.dtype), torch.cumsum(torch.mul((v[:,:,:-1]+v[:,:,1:])/2,w),axis=-1)),axis=-1)
        
        #quadratic term
        out_1=torch.mul((alphas**2)/2,torch.squeeze(torch.mul(torch.gather(v,-1, mx+1)-torch.gather(v,-1, mx), torch.gather(w,-1,mx)),axis=-1))
        
        #linear term
        out_2=torch.mul(torch.mul(alphas,torch.squeeze(torch.gather(v,-1,mx),axis=-1)), torch.squeeze(torch.gather(w,-1,mx),axis=-1))
        
        #constant
        out_3= torch.squeeze(torch.gather(vw,-1,mx),axis=-1)
        
        
        out=out_1+out_2+out_3
        
        # Regularization: points must be strictly within the unit hypercube
        # Use the dtype information from pytorch
        eps = torch.finfo(out.dtype).eps
        out = out.clamp(
            min=eps,
            max=1. - eps
        )
        if return_jacobian:
            # the derivative of this transformation is the linear interpolation between v_i-1 and v_i at alpha
            # the jacobian is the product of all linear interpolations
            # linear extrapolation between alpha, mx and mx+1
            logj=torch.sum(torch.log(torch.lerp(torch.gather(v,-1,mx).squeeze(-1), torch.gather(v,-1,mx+1).squeeze(-1), alphas)), dim=-1, keepdim=True)
            if torch.isnan(out).any() or torch.isnan(logj).any():
                import pdb;pdb.set_trace()
            return out, logj
        else:
            return out

    def flow(self, y, wv_tilde, return_jacobian=True):
        logj = None

        # TODO do a bottom-up assesment of how we handle the differentiability of variables
        
        v_tilde=wv_tilde[:,:,:int(np.ceil(wv_tilde.shape[2]/2))]
        w_tilde=wv_tilde[:,:,v_tilde.shape[2]:]
        N, k, b = w_tilde.shape
        
        Nx, kx = y.shape
        assert N == Nx and k == kx, "Shape mismatch"
        
        w=torch.exp(w_tilde)
        wsum = torch.cumsum(w, axis=-1) 
        wnorms = torch.unsqueeze(wsum[:, :, -1], axis=-1) 
        w = w/wnorms
        wsum=wsum/wnorms
        wsum_shift=torch.cat((torch.zeros([wsum.shape[0],wsum.shape[1],1]).to(wsum.device, wsum.dtype),wsum),axis=-1)
        
        v=modified_softmax(v_tilde, w).clamp_min(1e-6)
        
        #need to find the bin number for each of the y/x
        #-> find the last bin such that y is greater than the constant of the quadratic equation
        
        #vw_i= (v_i+1 - v_i)w_i/2 where i is the bin index. VW is the constant of the quadratic equation
        vw=torch.cat((torch.zeros([v.shape[0],v.shape[1],1]).to(wsum.device, wsum.dtype),
                                    torch.cumsum(torch.mul((v[:,:,:-1]+v[:,:,1:])/2,w),axis=-1)),axis=-1)
        # finder is contains 1 where y is smaller then the constant and 0 if it is greater
        finder=torch.where(vw>torch.unsqueeze(y,axis=-1),torch.zeros_like(vw),torch.ones_like(vw))
        eps = torch.finfo(vw.dtype).eps
        #the bin number can be extracted by finding the last index for which finder is nonzero. As vw 
        #is increasing, this can be found by searching for the maximum entry of finder*vw. In order to 
        #get the right result when y is in the first bin and finder is everywhere zero, a small first entry 
        #is added and mx is reduced by one to account for the shift.
        mx=torch.unsqueeze(
            torch.argmax(#we search for the maximum in order to find the last bin for which y was greater than vw
                torch.cat((torch.ones([vw.shape[0],vw.shape[1],1]).to(vw.device, vw.dtype)*eps,finder*(vw+1)),axis=-1),
                axis=-1), #we add an offset to ensure that if x is in the first bin, a maximal argument is found
            axis=-1)-1 # we substract -1 to account for the offset
        
        # x is in the mx-th bin: x \in [0,1],
        # mx \in [[0,b-1]], so we clamp away the case x == 1
        edges = torch.clamp(mx, 0, b - 1).to(torch.long)
        
        # Need special error handling because trying to index with mx
        # if it contains nans will lock the GPU. (device-side assert triggered)
        if torch.any(torch.isnan(edges)).item() or torch.any(edges < 0) or torch.any(edges >= b):
            import pdb;pdb.set_trace()
            raise ValueError("NaN detected in PWQuad bin indexing")
        
        #solve quadratic equation
        
        #prefactor of quadratic term
        a=torch.squeeze(torch.mul(torch.gather(v,-1, edges+1)-torch.gather(v,-1, edges),
                                                            torch.gather(w,-1,edges)),axis=-1)
        #prefactor of linear term
        b=torch.mul(torch.squeeze(torch.gather(v,-1,edges),axis=-1),torch.squeeze(torch.gather(w,-1,edges),axis=-1))
        #constant - y
        c= torch.squeeze(torch.gather(vw,-1,edges),axis=-1)-y
        
        #ensure that division by zero is taken care of
        eps = torch.finfo(a.dtype).eps
        a=torch.where(torch.abs(a)<eps,eps*torch.ones_like(a),a)
        
        d = (b**2) - (2*a*c)
        
        d = d.clamp_min(0)
        # if torch.any(d<0) or torch.any(a==0):
        #     import pdb;pdb.set_trace()
        #     assert not torch.any(d<0), "Value error in PWQuad inversion"
        #     assert not torch.any(a==0), "Value error in PWQuad inversion, a==0"
            
        # find two solutions
        sol1 = (-b-torch.sqrt(d))/(a)
        sol2 = (-b+torch.sqrt(d))/(a)
        
        # choose solution which is in the allowed range
    
        sol=torch.where((sol1>=0)&(sol1<1), sol1, sol2)
        
        if torch.any(torch.isnan(sol)).item():
            import pdb;pdb.set_trace()
            raise ValueError("NaN detected in PWQuad inversion")
        
        eps = torch.finfo(sol.dtype).eps
        
        
        sol = sol.clamp(
            min=eps,
            max=1. - eps
        )
        
        #the solution is the relative position inside the bin. This can be
        #converted to the absolute position by adding the sum of the bin widths up to this bin
        x=torch.mul(torch.squeeze(torch.gather(w,-1,edges),axis=-1), sol)+torch.squeeze(torch.gather(wsum_shift,-1,edges),axis=-1)
        
        eps = torch.finfo(x.dtype).eps
        
        x = x.clamp(
            min=eps,
            max=1. - eps
        )
        
        if return_jacobian:
            # linear extrapolation between sol, edges and edges+1 gives the jacobian of the forward transformation. The prefactor of -1 is the log of the jacobian of the inverse
            logj = -torch.sum(torch.log(torch.lerp(torch.gather(v,-1,edges).squeeze(-1), torch.gather(v,-1,edges+1).squeeze(-1), sol)), dim=-1, keepdim=True)
            if torch.isnan(x).any() or torch.isnan(logj).any():
                import pdb;pdb.set_trace()
            return x, logj
        else:
            return x
        

class ElementWiseAffineTransform:
    def flow(self, x, st, return_jacobian=True):
        es = torch.exp(st[..., 0])
        t = st[..., 1]
        x = es * x + t
        if return_jacobian:
            logj = torch.sum(torch.log(es.clamp_min(1e-6)), dim=-1, keepdim=True)
            return x, logj
        else:
            return x
    
    def flow_inv(self, x, st, return_jacobian=True):
        es = torch.exp(-st[..., 0])
        t = st[..., 1]
        x = es * (x - t)
        if return_jacobian:
            logj = torch.sum(torch.log(es.clamp_min(1e-6)), dim=-1, keepdim=True)
            return x, logj
        else:
            return x

class Block(torch.nn.Module):
    def __init__(self, d, mask,
                 d_hidden=64,
                 n_hidden=3,
                 feature_dim=128,
                 multires=3,
                 hidden_activation=torch.nn.LeakyReLU,
                 transform=ElementWisePWQuadraticTransform,
                 n_bins=21,
                 input_activation=Reshift,
                 ):
        super().__init__()
        self.d = d

        self.mask = mask
        self.mask_complement = [not it for it in mask]
        self.transform = transform()
        
        d_in = sum(mask)
        d_out = d - d_in
        self.embed_fn = None
        if multires > 0:
            self.embed_fn, d_in = get_embedder(multires, input_dims=d_in)

        self.out_shape = (d_out, n_bins)

        d_out = np.prod(self.out_shape)

        layers = []
        if input_activation is not None:
            layers.append(input_activation())
        last_dim = d_in + feature_dim
        for i in range(n_hidden):
            lin = torch.nn.Linear(last_dim, d_hidden)
            # nn.init.zeros_(lin.bias)
            # nn.init.zeros_(lin.weight)
            # torch.nn.init.constant_(lin.bias, 0.0)
            # torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(d_hidden))
            layers.append(lin)
            layers.append(hidden_activation())
            last_dim = d_hidden
        
        lin = torch.nn.Linear(last_dim, d_out)
        # nn.init.zeros_(lin.bias)
        # nn.init.zeros_(lin.weight)
        # torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(d_hidden), std=0.0001)
        # torch.nn.init.constant_(lin.bias, 0.0)
        layers.append(lin)

        self.nn = torch.nn.Sequential(*layers)

    def flow(self, y, logj, feature, return_jacobian=False):
        y_n = y[..., self.mask]
        y_m = y[..., self.mask_complement]

        x = torch.zeros_like(y)
        x[..., self.mask] = y_n
        if self.embed_fn is not None:
            y_n = self.embed_fn(y_n)
        nn_in = torch.cat([y_n, feature], dim=-1)
        st = self.nn(nn_in).view(*(y_n.shape[:-1]), *self.out_shape)
        if return_jacobian:
            x[..., self.mask_complement], logjy = self.transform.flow(y_m, st, return_jacobian)
            logj = logj + logjy
            return x, logj
        else:
            x[..., self.mask_complement] = self.transform.flow(y_m, st, return_jacobian)
            return x

    def print_grad(self):
        for name, param in self.named_parameters():
            if param.grad is not None and param.grad.sum() != 0:
                print(name, param.grad.sum())
    
    def flow_inv(self, y, logj, feature, return_jacobian=False):
        y_n = y[..., self.mask]
        y_m = y[..., self.mask_complement]

        x = torch.zeros_like(y)
        x[..., self.mask] = y_n
        if self.embed_fn is not None:
            y_n = self.embed_fn(y_n)
        nn_in = torch.cat([y_n, feature], dim=-1)
        st = self.nn(nn_in).view(*(y_n.shape[:-1]), *self.out_shape)
        if return_jacobian:
            x[..., self.mask_complement], logjy = self.transform.flow_inv(y_m, st, return_jacobian)
            logj = logj + logjy
            if torch.isnan(logj).any():
                import pdb;pdb.set_trace()
            return x, logj
        else:
            x[..., self.mask_complement] = self.transform.flow_inv(y_m, st, return_jacobian)
            return x

class TensoFlow(torch.nn.Module):
    flow_kwargs = {
        "realnvp": (FactorizedGaussianSampler, ElementWiseAffineTransform, lambda x: 2, None, InvertibleAnalyticSigmoid),
        "pwlinear": (SphereSampler, ElementWisePWLinearTransform, lambda x: x, Reshift, None),
        "pwquad": (SphereSampler, ElementWisePWQuadraticTransform, lambda x: 2*x+1, Reshift, None)
    }
    def __init__(self, d, aabb, device='cuda', gridSize=[512, 512, 512], nis_n_comp=12, nis_dim = 64, nis_feature_dim=16, nis_multires=3, refl_multires=3, roughness_multires=3, angle_multires=3, flow='pwquad', n_bins=10, disable_tensorial=False, disable_reflected=False):
        super().__init__()
        
        self.nis_n_comp = nis_n_comp
        self.nis_dim = nis_dim
        self.nis_feature_dim = nis_feature_dim
        self.device = device
        self.matMode = [[0,1], [0,2], [1,2]]
        self.vecMode =  [2, 1, 0]
        self.comp_w = [1,1,1]
        self.nplane = len(self.vecMode)
        self.gridSize = torch.tensor(gridSize)
        self.aabb = aabb
        self.n_levels = 3
        self.init_svd_volume(device)
        self.init_mlp(device, nis_multires=nis_multires, refl_multires=refl_multires, roughness_multires=roughness_multires)

        flow_priors, transform, bin_fn, input_activation, output_cell = self.flow_kwargs[flow]
        
        masks = [] 
        for rep in range(1):
            for offset in range(2):
                masks.append([(i + offset) % 2 == 0 for i in range(d)])
        flows = []
        for mask in masks:
            flows.append(Block(d=d, mask=mask, feature_dim=nis_feature_dim + self.refl_input_ch + self.roughness_input_ch, multires=angle_multires, transform=transform, n_bins=bin_fn(n_bins), input_activation=input_activation))
        if output_cell is not None:
            flows.append(output_cell())
        self.flows = torch.nn.ModuleList(flows)
        self.latent_prior = flow_priors(d=d)

        self.disable_tensorial = disable_tensorial
        self.disable_reflected = disable_reflected
        
    def init_svd_volume(self, device):
        self.nis_plane, self.nis_line = self.init_one_svd(self.nis_n_comp, device)

    def init_mlp(self, device, nis_multires, refl_multires, roughness_multires):
        self.embed_fn = None
        nis_input_ch = 3
        if nis_multires > 0:
            self.embed_fn, nis_input_ch = get_embedder(nis_multires, input_dims=nis_input_ch)

        nis_feat_input_ch = self.nis_n_comp * self.nplane
        
        self.nis_mat = nn.Sequential(
            nn.Linear(nis_feat_input_ch + nis_input_ch, self.nis_dim), nn.Softplus(beta=100),
            nn.Linear(self.nis_dim , self.nis_feature_dim)
        ).to(device)
        
        self.embed_fn_refl = None
        self.refl_input_ch = 2
        if refl_multires > 0:
            self.embed_fn_refl, self.refl_input_ch = get_embedder(refl_multires, input_dims=self.refl_input_ch)
            
        self.embed_fn_roughness = None
        self.roughness_input_ch = 1
        if roughness_multires > 0:
            self.embed_fn_roughness, self.roughness_input_ch = get_embedder(roughness_multires, input_dims=self.roughness_input_ch)

    def tenso_feature(self, xyz_sampled, level_vol=None):
        # xyz_sampled : (rn*sn, 3)
        # plane + line basis
        inputs_xyz = xyz_sampled
        xyz_sampled = contraction(xyz_sampled, self.aabb).reshape(-1, 3)
        level = (torch.zeros([xyz_sampled.shape[0], 1], device=xyz_sampled.device) if level_vol is None else level_vol).view(-1, 1).unsqueeze(0).contiguous() # 1, N, 1
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)          # 3, rn * sn, 1, 2

        plane_coef_point,line_coef_point = [],[]
        planes, lines = self.nis_plane, self.nis_line
        for idx in range(self.nplane):
            plane_coef_point.append(
                dr.texture(planes[idx].permute(0, 2, 3, 1).contiguous(), 
                        coordinate_plane[[idx]], 
                        mip_level_bias=level, 
                        boundary_mode="clamp", 
                        max_mip_level=self.n_levels-1
                        ).permute(0, 3, 1, 2).contiguous().view(-1, *xyz_sampled.shape[:1]))
            line_coef_point.append(
                dr.texture(lines[idx].permute(0, 2, 3, 1).contiguous(), 
                        coordinate_line[[idx]], 
                        mip_level_bias=level, 
                        boundary_mode="clamp", 
                        max_mip_level=self.n_levels-1
                        ).permute(0, 3, 1, 2).contiguous().view(-1, *xyz_sampled.shape[:1]))
                
        plane_coef_point, line_coef_point = torch.cat(plane_coef_point), torch.cat(line_coef_point)
        sigma_feature = plane_coef_point * line_coef_point

        inputs_feat = sigma_feature.T
        if self.embed_fn is not None:
            inputs_xyz = self.embed_fn(inputs_xyz)
        out_feats = self.nis_mat(torch.cat([inputs_feat, inputs_xyz], dim=-1))
        return out_feats

    def get_optparam_groups(self, lr_init_spatialxyz = 0.01, lr_init_network = 0.001):
        grad_vars = [
            {'params': self.nis_line, 'lr': lr_init_spatialxyz}, 
            {'params': self.nis_plane, 'lr': lr_init_spatialxyz},
            {'params': self.nis_mat.parameters(), 'lr':lr_init_network},
            {'params': self.flows.parameters(), 'lr':lr_init_network},
        ]
        return grad_vars

    def init_one_svd(self, n_component, device):
        plane_coef, line_coef = [], []
        for i in range(self.nplane):
            planeSize = self.gridSize[self.matMode[i]]
            lineSize = self.gridSize[self.vecMode[i]]
            init_plane = 1e-4 * (2 * torch.rand(1, n_component, planeSize[0], planeSize[1]) - 1) # 1, n, grid, grid
            init_line = torch.ones((1, n_component, lineSize, 1)) * (1./(n_component * self.nplane)) # 1, n, grid, 1
            plane_coef.append(nn.Parameter(init_plane.clone()))
            line_coef.append(nn.Parameter(init_line.clone()))
        return nn.ParameterList(plane_coef).to(device), nn.ParameterList(line_coef).to(device)

    def flow(self, x, logj, feature, return_jacobian=False):
        feature = feature.unsqueeze(1).expand(-1, x.shape[1], -1)
        pre_shape = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
        logj = logj.reshape(-1, logj.shape[-1])
        feature = feature.reshape(-1, feature.shape[-1])
        for f in self.flows:
            if return_jacobian:
                x, logj = f.flow(x, logj, feature, return_jacobian)
            else:
                x = f.flow(x, logj, feature, return_jacobian)
        if return_jacobian:
            return x.reshape(*pre_shape, x.shape[-1]), logj.reshape(*pre_shape, logj.shape[-1])
        else:
            return x.reshape(*pre_shape, x.shape[-1])
    
    def flow_inv(self, x, logj, feature, return_jacobian=False, rays_id=None):
        if len(x.shape) == 3:
            feature = feature.unsqueeze(1).expand(-1, x.shape[1], -1)
        pre_shape = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
        logj = logj.reshape(-1, logj.shape[-1])
        feature = feature.reshape(-1, feature.shape[-1])
        for f in self.flows[::-1]:
            if return_jacobian:
                x, logj = f.flow_inv(x, logj, feature, return_jacobian)
                if torch.isnan(logj).any():
                    import pdb;pdb.set_trace()
            else:
                x = f.flow_inv(x, logj, feature, return_jacobian)
        if return_jacobian:
            return x.reshape(*pre_shape, x.shape[-1]), logj.reshape(*pre_shape, logj.shape[-1])
        else:
            return x.reshape(*pre_shape, x.shape[-1])
    
    def forward(self, pts, reflections, roughness, x, return_jacobian=False, rays_id=None):
        x = x.clamp(1e-6, 1-1e-6)
        feature = self.tenso_feature(pts)
        if self.disable_tensorial:
            feature = torch.zeros_like(feature)
        if self.embed_fn_refl is not None:
            reflections = self.embed_fn_refl(reflections)
        if self.disable_reflected:
            reflections = torch.zeros_like(reflections)
        if self.embed_fn_roughness is not None:
            roughness = self.embed_fn_roughness(roughness)
        # feature = torch.zeros_like(feature)
        # reflections = torch.zeros_like(reflections)
        roughness = torch.zeros_like(roughness)
        feature = torch.cat([feature, reflections, roughness], dim=-1)
        # feature = torch.cat([feature, reflections], dim=-1)
        
        if rays_id is not None:
            feature = feature[rays_id]
        
        logj = torch.zeros(*x.shape[:-1], 1)
        
        if return_jacobian:
            z, logj = self.flow_inv(x, logj, feature, return_jacobian=return_jacobian)
            logqx = logj + self.latent_prior.log_prob(z)
            if torch.isnan(logqx).any():
                import pdb;pdb.set_trace()
            return z, logqx
        else:
            z = self.flow_inv(x, logj, feature, return_jacobian=return_jacobian)
            return z
    
    def sample(self, pts, reflections, roughness, n_samples, return_jacobian=False):
        shape = (pts.shape[0], n_samples)
        x, logj = self.latent_prior(shape)
        feature = self.tenso_feature(pts)
        if self.disable_tensorial:
            feature = torch.zeros_like(feature)
        if self.embed_fn_refl is not None:
            reflections = self.embed_fn_refl(reflections)
        if self.disable_reflected:
            reflections = torch.zeros_like(reflections)
        if self.embed_fn_roughness is not None:
            roughness = self.embed_fn_roughness(roughness)
        # feature = torch.zeros_like(feature)
        # reflections = torch.zeros_like(reflections)
        roughness = torch.zeros_like(roughness)
        feature = torch.cat([feature, reflections, roughness], dim=-1)
        # feature = torch.cat([feature, reflections], dim=-1)
        
        # angle, logj1 = self.flow(x, logj, feature, return_jacobian=return_jacobian)
        # logj_ = torch.zeros(*angle.shape[:-1], 1)
        # z, logj2 = self.flow_inv(angle, logj_, feature, return_jacobian=return_jacobian)
        # logqx = logj2 + self.latent_prior.log_prob(z)
        return self.flow(x, logj, feature, return_jacobian=return_jacobian)
        
        
        feature = feature.unsqueeze(1).expand(-1, x.shape[1], -1)
        pre_shape = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
        logj = logj.reshape(-1, logj.shape[-1])
        feature = feature.reshape(-1, feature.shape[-1])
        
        # angle, logj1 = self.flow(x, logj, feature, return_jacobian=return_jacobian)
        x11, logj11 = self.flows[0].flow(x, logj, feature, return_jacobian)
        x12, logj12 = self.flows[1].flow(x11, logj11, feature, return_jacobian)
        
        nn_in = torch.cat([self.flows[0].embed_fn(x[:, [0]]), feature], dim=-1)
        st = self.flows[0].nn(nn_in).view(*(nn_in.shape[:-1]), *self.flows[0].out_shape)
        x_t, logjy = self.flows[0].transform.flow(x[:, [1]], st, return_jacobian)
        x_t2, logjy2 = self.flows[0].transform.flow_inv(x_t, st, return_jacobian)
        
        logj_ = torch.zeros(*angle.shape[:-1], 1)
        angle = angle.reshape(-1, angle.shape[-1])
        logj_ = logj_.reshape(-1, logj_.shape[-1])
        
        # logj_ = torch.zeros(*angle.shape[:-1], 1)
        # z, logj2 = self.flow_inv(angle, logj_, feature, return_jacobian=return_jacobian)
        z, logj2 = self.flows[0].flow_inv(x, logj, feature, return_jacobian)
        logqx = logj2 + self.latent_prior.log_prob(z)
        return self.flow(x, logj, feature, return_jacobian=return_jacobian)

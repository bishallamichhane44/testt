import torch
import torch.nn as nn

class CConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        
        self.real_conv = nn.Conv2d(in_channels=self.in_channels, 
                                   out_channels=self.out_channels, 
                                   kernel_size=self.kernel_size, 
                                   padding=self.padding, 
                                   stride=self.stride)
        
        self.im_conv = nn.Conv2d(in_channels=self.in_channels, 
                                 out_channels=self.out_channels, 
                                 kernel_size=self.kernel_size, 
                                 padding=self.padding, 
                                 stride=self.stride)
        
        nn.init.xavier_uniform_(self.real_conv.weight)
        nn.init.xavier_uniform_(self.im_conv.weight)
        
    def forward(self, x):
        if torch.is_complex(x):
            x_real = x.real
            x_im = x.imag
        else:
            x_real = x[..., 0]
            x_im = x[..., 1]
        
        x_real = x_real.to(self.real_conv.weight.device)
        x_im = x_im.to(self.im_conv.weight.device)
        
        c_real = self.real_conv(x_real) - self.im_conv(x_im)
        c_im = self.im_conv(x_real) + self.real_conv(x_im)
        
        return torch.complex(c_real, c_im)

class CConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, output_padding=0, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.output_padding = output_padding
        self.padding = padding
        self.stride = stride
        
        self.real_convt = nn.ConvTranspose2d(in_channels=self.in_channels, 
                                            out_channels=self.out_channels, 
                                            kernel_size=self.kernel_size, 
                                            output_padding=self.output_padding,
                                            padding=self.padding,
                                            stride=self.stride)
        
        self.im_convt = nn.ConvTranspose2d(in_channels=self.in_channels, 
                                            out_channels=self.out_channels, 
                                            kernel_size=self.kernel_size, 
                                            output_padding=self.output_padding, 
                                            padding=self.padding,
                                            stride=self.stride)
        
        nn.init.xavier_uniform_(self.real_convt.weight)
        nn.init.xavier_uniform_(self.im_convt.weight)
        
    def forward(self, x):
        if torch.is_complex(x):
            x_real = x.real
            x_im = x.imag
        else:
            x_real = x[..., 0]
            x_im = x[..., 1]
        
        x_real = x_real.to(self.real_convt.weight.device)
        x_im = x_im.to(self.im_convt.weight.device)
        
        ct_real = self.real_convt(x_real) - self.im_convt(x_im)
        ct_im = self.im_convt(x_real) + self.real_convt(x_im)
        
        if torch.is_complex(x):
            return torch.complex(ct_real, ct_im)
        else:
            return torch.stack([ct_real, ct_im], dim=-1)

class CBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        
        self.real_b = nn.BatchNorm2d(num_features=self.num_features, eps=self.eps, momentum=self.momentum,
                                      affine=self.affine, track_running_stats=self.track_running_stats)
        self.im_b = nn.BatchNorm2d(num_features=self.num_features, eps=self.eps, momentum=self.momentum,
                                    affine=self.affine, track_running_stats=self.track_running_stats) 
        
    def forward(self, x):
        if torch.is_complex(x):
            x_real = x.real
            x_im = x.imag
        else:
            x_real = x[..., 0]
            x_im = x[..., 1]
            
        if x_real.dim() == 3:
            x_real = x_real.unsqueeze(0)
            x_im = x_im.unsqueeze(0)
            
        x_real = x_real.to(self.real_b.weight.device if self.affine else self.real_b.running_mean.device)
        x_im = x_im.to(self.im_b.weight.device if self.affine else self.im_b.running_mean.device)
        
        n_real = self.real_b(x_real)
        n_im = self.im_b(x_im)
        
        if x.dim() == 3:
            n_real = n_real.squeeze(0)
            n_im = n_im.squeeze(0)
        
        if torch.is_complex(x):
            return torch.complex(n_real, n_im)
        else:
            return torch.stack([n_real, n_im], dim=-1)

class Encoder(nn.Module):
    def __init__(self, filter_size=(7,5), stride_size=(2,2), in_channels=1, out_channels=45, padding=(0,0)):
        super().__init__()
        self.filter_size = filter_size
        self.stride_size = stride_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding

        self.cconv = CConv2d(in_channels=self.in_channels, out_channels=self.out_channels, 
                             kernel_size=self.filter_size, stride=self.stride_size, padding=self.padding)
        
        self.cbn = CBatchNorm2d(num_features=self.out_channels) 
        self.leaky_relu = nn.LeakyReLU()
            
    def forward(self, x):
        conved = self.cconv(x)
        normed = self.cbn(conved)
        
        if torch.is_complex(normed):
            real_part = self.leaky_relu(normed.real)
            imag_part = self.leaky_relu(normed.imag)
            acted = torch.complex(real_part, imag_part)
        else:
            acted = self.leaky_relu(normed)
        
        return acted

class Decoder(nn.Module):
    def __init__(self, filter_size=(7,5), stride_size=(2,2), in_channels=1, out_channels=45,
                 output_padding=(0,0), padding=(0,0), last_layer=False):
        super().__init__()
        self.filter_size = filter_size
        self.stride_size = stride_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.output_padding = output_padding
        self.padding = padding
        self.last_layer = last_layer
        
        self.cconvt = CConvTranspose2d(in_channels=self.in_channels, out_channels=self.out_channels, 
                             kernel_size=self.filter_size, stride=self.stride_size, output_padding=self.output_padding, padding=self.padding)
        
        self.cbn = CBatchNorm2d(num_features=self.out_channels) 
        self.leaky_relu = nn.LeakyReLU()
            
    def forward(self, x):
        conved = self.cconvt(x)
        
        if not self.last_layer:
            normed = self.cbn(conved)
            if torch.is_complex(normed):
                real_part = self.leaky_relu(normed.real)
                imag_part = self.leaky_relu(normed.imag)
                output = torch.complex(real_part, imag_part)
            else:
                output = self.leaky_relu(normed)
        else:
            if torch.is_complex(conved):
                m_phase = conved / (torch.abs(conved) + 1e-8)
                m_mag = torch.tanh(torch.abs(conved))
                output = m_phase * m_mag  # Keep as complex tensor
            else:
                # Handle non-complex case (though unlikely with STFT input)
                m_phase = conved / (torch.abs(conved) + 1e-8)
                m_mag = torch.tanh(torch.abs(conved))
                output = torch.complex(m_phase, m_mag)  # Convert to complex explicitly
        
        return output

class DCUnet20(nn.Module):
    def __init__(self, n_fft=3072, hop_length=768):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        self.set_size(model_complexity=int(45//1.414), input_channels=1, model_depth=20)
        self.encoders = []
        self.model_length = 20 // 2
        
        for i in range(self.model_length):
            module = Encoder(in_channels=self.enc_channels[i], out_channels=self.enc_channels[i + 1],
                             filter_size=self.enc_kernel_sizes[i], stride_size=self.enc_strides[i], padding=self.enc_paddings[i])
            self.add_module("encoder{}".format(i), module)
            self.encoders.append(module)

        self.decoders = []
        for i in range(self.model_length):
            if i != self.model_length - 1:
                module = Decoder(in_channels=self.dec_channels[i] + self.enc_channels[self.model_length - i], out_channels=self.dec_channels[i + 1], 
                                 filter_size=self.dec_kernel_sizes[i], stride_size=self.dec_strides[i], padding=self.dec_paddings[i],
                                 output_padding=self.dec_output_padding[i])
            else:
                module = Decoder(in_channels=self.dec_channels[i] + self.enc_channels[self.model_length - i], out_channels=self.dec_channels[i + 1], 
                                 filter_size=self.dec_kernel_sizes[i], stride_size=self.dec_strides[i], padding=self.dec_paddings[i],
                                 output_padding=self.dec_output_padding[i], last_layer=True)
            self.add_module("decoder{}".format(i), module)
            self.decoders.append(module)
       
    def forward(self, x, is_istft=True, window=None, n_fft=None, hop_length=None):
        device = next(self.parameters()).device
        x = x.to(device)
        orig_x = x
        
        # Use provided parameters or fall back to instance defaults
        n_fft = n_fft if n_fft is not None else self.n_fft
        hop_length = hop_length if hop_length is not None else self.hop_length
        
        # Expect x to be [batch, channels, freq, time]
        xs = []
        for i, encoder in enumerate(self.encoders):
            xs.append(x)
            x = encoder(x)
        
        p = x
        for i, decoder in enumerate(self.decoders):
            p = decoder(p)
            if i == self.model_length - 1:
                break
            if torch.is_complex(p):
                skip = xs[self.model_length - 1 - i]
                p = torch.cat([p, skip], dim=1)
            else:
                p = torch.cat([p, xs[self.model_length - 1 - i]], dim=1)
        
        mask = p
        output = mask * orig_x  # Should be [batch, channels, freq, time]
        
        if is_istft:
            if window is None:
                window = torch.hann_window(n_fft).to(device)
            # Ensure output is [batch, freq, time] for istft by squeezing channels dimension
            if output.dim() == 4 and output.size(1) == 1:  # [batch, 1, freq, time]
                output = output.squeeze(1)  # -> [batch, freq, time]
            output = torch.istft(
                output,
                n_fft=n_fft,
                hop_length=hop_length,
                window=window,
                normalized=True,
                onesided=True,
                center=True,
                return_complex=False
            )
        
        return output
    
    def set_size(self, model_complexity, model_depth=20, input_channels=1):
        if model_depth == 20:
            self.enc_channels = [input_channels,
                                 model_complexity,
                                 model_complexity,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 128]

            self.enc_kernel_sizes = [(7, 1),
                                     (1, 7),
                                     (6, 4),
                                     (7, 5),
                                     (5, 3),
                                     (5, 3),
                                     (5, 3),
                                     (5, 3),
                                     (5, 3),
                                     (5, 3)]

            self.enc_strides = [(1, 1),
                                (1, 1),
                                (2, 2),
                                (2, 1),
                                (2, 2),
                                (2, 1),
                                (2, 2),
                                (2, 1),
                                (2, 2),
                                (2, 1)]

            self.enc_paddings = [(3, 0),
                                 (0, 3),
                                 (0, 0),
                                 (0, 0),
                                 (0, 0),
                                 (0, 0),
                                 (0, 0),
                                 (0, 0),
                                 (0, 0),
                                 (0, 0)]

            self.dec_channels = [0,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity,
                                 model_complexity,
                                 1]

            self.dec_kernel_sizes = [(6, 3), 
                                     (6, 3),
                                     (6, 3),
                                     (6, 4),
                                     (6, 3),
                                     (6, 4),
                                     (8, 5),
                                     (7, 5),
                                     (1, 7),
                                     (7, 1)]

            self.dec_strides = [(2, 1),
                                (2, 2),
                                (2, 1),
                                (2, 2),
                                (2, 1),
                                (2, 2),
                                (2, 1),
                                (2, 2),
                                (1, 1),
                                (1, 1)]

            self.dec_paddings = [(0, 0),
                                 (0, 0),
                                 (0, 0),
                                 (0, 0),
                                 (0, 0),
                                 (0, 0),
                                 (0, 0),
                                 (0, 0),
                                 (0, 3),
                                 (3, 0)]
            
            self.dec_output_padding = [(0,0),
                                       (0,0),
                                       (0,0),
                                       (0,0),
                                       (0,0),
                                       (0,0),
                                       (0,0),
                                       (0,0),
                                       (0,0),
                                       (0,0)]
        else:
            raise ValueError("Unknown model depth : {}".format(model_depth))
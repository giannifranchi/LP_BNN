import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import numpy as np

class EnsembleModule(nn.Module):
    """
    creates base class for BNN, in order to enable specific behavior
    """
    def init(self):
        super().__init__()


class Ensemble_FC(EnsembleModule):
    def __init__(self, in_features, out_features, first_layer, \
                 num_models, bias=True):
        super(Ensemble_FC, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_size = 32
        self.fc = nn.Linear(in_features, out_features, bias=False)
        #self.alpha = torch.Tensor(num_models, in_features).cuda()
        #self.gamma = torch.Tensor(num_models, out_features).cuda()
        self.alpha = nn.Parameter(torch.Tensor(num_models, in_features),requires_grad=False)
        self.gamma = nn.Parameter(torch.Tensor(num_models, out_features))
        self.encoder_fc1 = nn.Linear(in_features, self.hidden_size)
        self.encoder_fcmean = nn.Linear(self.hidden_size, self.hidden_size)
        self.encoder_fcvar = nn.Linear(self.hidden_size, self.hidden_size)
        #self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, 1, batch_first=True, dropout=0.2)
        self.decoder_fc1 = nn.Linear(self.hidden_size, in_features)
        self.loss_latent = 0
        #self.alpha_little = nn.Parameter(torch.Tensor(num_models, self.hidden_size))
        self.num_models = num_models
        if bias:
            #self.bias = nn.Parameter(torch.Tensor(out_features))
            self.bias = nn.Parameter(torch.Tensor(self.num_models, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.first_layer = first_layer

    def reset_parameters(self):
        #nn.init.constant_(self.alpha, 1.0)
        #nn.init.constant_(self.gamma, 1.0)
        nn.init.normal_(self.alpha, mean=1., std=0.1)
        nn.init.normal_(self.gamma, mean=1., std=0.1)
        #nn.init.normal_(self.alpha, mean=1., std=1)
        #nn.init.normal_(self.gamma, mean=1., std=1)
        #alpha_coeff = torch.randint_like(self.alpha, low=0, high=2)
        #alpha_coeff.mul_(2).add_(-1)
        #gamma_coeff = torch.randint_like(self.gamma, low=0, high=2)
        #gamma_coeff.mul_(2).add_(-1)
        #with torch.no_grad():
        #    self.alpha *= alpha_coeff
        #    self.gamma *= gamma_coeff
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.fc.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)


    def update_indices(self, indices):
        self.indices = indices

    def forward(self, x):
        embedded=F.relu(self.encoder_fc1(self.alpha))
        #embedded, _ = self.rnn(embedded.view(len(self.alpha), 1, -1))
        embedded_mean, embedded_logvar=self.encoder_fcmean(embedded),self.encoder_fcmean(embedded)
        z_embedded = self.reparameterize(embedded_mean, embedded_logvar)
        alpha_decoded = self.decoder_fc1(z_embedded)
        if self.training:
            MSE = F.mse_loss(alpha_decoded,self.alpha) #F.binary_cross_entropy(alpha_decoded, self.alpha, reduction='sum')
            KLD = -0.5 * torch.sum(1 + embedded_logvar - embedded_mean.pow(2) - embedded_logvar.exp())
            self.loss_latent = MSE + KLD


            curr_bs = x.size(0)
            makeup_bs = self.num_models - curr_bs
            if makeup_bs > 0:
                indices = torch.randint(
                    high=self.num_models,
                    size=(curr_bs,), device=self.alpha.device)
                alpha = torch.index_select(alpha_decoded, 0, indices)
                gamma = torch.index_select(self.gamma, 0, indices)
                bias = torch.index_select(self.bias, 0, indices)
                result = self.fc(x * alpha) * gamma + bias
            elif makeup_bs < 0:
                indices = torch.randint(
                    high=self.num_models,
                    size=(curr_bs,), device=self.alpha.device)
                alpha = torch.index_select(alpha_decoded, 0, indices)
                gamma = torch.index_select(self.gamma, 0, indices)
                bias = torch.index_select(self.bias, 0, indices)
                result = self.fc(x * alpha) * gamma + bias
            else:
                result = self.fc(x * alpha_decoded) * self.gamma + self.bias
            return result[:curr_bs]
        else:
            if self.first_layer:
                # Repeated pattern: [[A,B,C],[A,B,C]]
                x = torch.cat([x for i in range(self.num_models)], dim=0)
            # Repeated pattern: [[A,A],[B,B],[C,C]]
            batch_size = int(x.size(0) / self.num_models)
            alpha = torch.cat(
                [alpha_decoded for i in range(batch_size)],
                dim=1).view([-1, self.in_features])
            gamma = torch.cat(
                [self.gamma for i in range(batch_size)],
                dim=1).view([-1, self.out_features])
            bias = torch.cat(
                [self.bias for i in range(batch_size)],
                dim=1).view([-1, self.out_features])
            result = self.fc(x * alpha) * gamma + bias
            return result


class Ensemble_orderFC(EnsembleModule):
    def __init__(self, in_features, out_features, num_models, first_layer=False,
                 bias=True, constant_init=False, p=0.5, random_sign_init=False,):
        super(Ensemble_orderFC, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_size = 32
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.alpha = nn.Parameter(torch.Tensor(num_models, in_features))
        self.gamma = nn.Parameter(torch.Tensor(num_models, out_features))
        #self.alpha = torch.Tensor(num_models, in_features).cuda()
        #self.gamma = torch.Tensor(num_models, out_features).cuda()
        self.encoder_fc1 = nn.Linear(in_features, self.hidden_size)
        self.encoder_fcmean = nn.Linear(self.hidden_size, self.hidden_size)
        self.encoder_fcvar = nn.Linear(self.hidden_size, self.hidden_size)
        #self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, 1, batch_first=True, dropout=0.2)
        self.decoder_fc1 = nn.Linear(self.hidden_size, in_features)
        self.loss_latent = 0
        self.num_models = num_models
        self.random_sign_init = random_sign_init
        self.constant_init = constant_init
        self.probability = p
        if bias:
            #self.bias = nn.Parameter(torch.Tensor(out_features))
            self.bias = nn.Parameter(torch.Tensor(self.num_models, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.first_layer = first_layer

    def reset_parameters(self):
        if self.constant_init:
            nn.init.constant_(self.alpha, 1.)
            nn.init.constant_(self.gamma, 1.)
            if self.random_sign_init:
                if self.probability  == -1:
                    with torch.no_grad():
                        factor = torch.ones(
                            self.num_models, device=self.alpha.device).bernoulli_(0.5)
                        factor.mul_(2).add_(-1)
                        self.alpha.data = (self.alpha.t() * factor).t()
                        self.gamma.data = (self.gamma.t() * factor).t()
                elif self.probability  == -2:
                    with torch.no_grad():
                        positives_num = self.num_models // 2
                        factor1 = torch.Tensor([1 for i in range(positives_num)])
                        factor2 = torch.Tensor(
                            [-1 for i in range(self.num_models-positives_num)])
                        factor = torch.cat([factor1, factor2]).to(self.alpha.device)
                        self.alpha.data = (self.alpha.t() * factor).t()
                        self.gamma.data = (self.gamma.t() * factor).t()
                else:
                    with torch.no_grad():
                        self.alpha.bernoulli_(self.probability)
                        self.alpha.mul_(2).add_(-1)
                        self.gamma.bernoulli_(self.probability)
                        self.gamma.mul_(2).add_(-1)
        else:
            nn.init.normal_(self.alpha, mean=1., std=0.5)
            nn.init.normal_(self.gamma, mean=1., std=0.5)
            #nn.init.normal_(self.alpha, mean=1., std=1)
            #nn.init.normal_(self.gamma, mean=1., std=1)
            if self.random_sign_init:
                with torch.no_grad():
                    alpha_coeff = torch.randint_like(self.alpha, low=0, high=2)
                    alpha_coeff.mul_(2).add_(-1)
                    self.alpha *= alpha_coeff
                    gamma_coeff = torch.randint_like(self.gamma, low=0, high=2)
                    gamma_coeff.mul_(2).add_(-1)
                    self.gamma *= gamma_coeff
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.fc.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def update_indices(self, indices):
        self.indices = indices

    def reparameterize(self, mu, logvar):

        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)


    def forward(self, x):
        embedded=F.relu(self.encoder_fc1(self.alpha))
        #embedded, _ = self.rnn(embedded.view(len(self.alpha), 1, -1))
        embedded_mean, embedded_logvar = self.encoder_fcmean(embedded), self.encoder_fcmean(embedded)
        z_embedded = self.reparameterize(embedded_mean, embedded_logvar)
        #print(z_embedded,', mean =>',embedded_mean,', var =>',embedded_logvar.exp_())
        alpha_decoded = self.decoder_fc1(z_embedded)
        if self.training:
            MSE = F.mse_loss(alpha_decoded,self.alpha) #F.binary_cross_entropy(alpha_decoded, self.alpha, reduction='sum')
            KLD = -0.5 * torch.sum(1 + embedded_logvar - embedded_mean.pow(2) - embedded_logvar.exp())
            self.loss_latent = MSE + KLD

        if not self.training and self.first_layer:
            # Repeated pattern in test: [[A,B,C],[A,B,C]]
            x = torch.cat([x for i in range(self.num_models)], dim=0)

        num_examples_per_model = int(x.size(0) / self.num_models)
        extra = x.size(0) - (num_examples_per_model * self.num_models)
        # Repeated pattern: [[A,A],[B,B],[C,C]]
        if num_examples_per_model != 0:
            alpha = torch.cat(
                [alpha_decoded for i in range(num_examples_per_model)],
                dim=1).view([-1, self.in_features])
            gamma = torch.cat(
                [self.gamma for i in range(num_examples_per_model)],
                dim=1).view([-1, self.out_features])
            if self.bias is not None:
                bias = torch.cat(
                    [self.bias for i in range(num_examples_per_model)],
                    dim=1).view([-1, self.out_features])
        else:
            alpha = self.alpha.clone()
            gamma = self.gamma.clone()
            if self.bias is not None:
                bias = self.bias.clone()
        if extra != 0:
            alpha = torch.cat([alpha, alpha[:extra]], dim=0)
            gamma = torch.cat([gamma, gamma[:extra]], dim=0)
            bias = torch.cat([bias, bias[:extra]], dim=0)

        result = self.fc(x*alpha)*gamma
        return result + bias if self.bias is not None else result


class Ensemble_Conv2d(EnsembleModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 groups=1, first_layer=False, num_models=100, train_gamma=True,
                 bias=True, constant_init=False, p=0.5, random_sign_init=False):
        super(Ensemble_Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = 32
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, groups=groups, bias=False)
        self.alpha = nn.Parameter(torch.Tensor(num_models, in_channels), requires_grad=False)
        self.train_gamma = train_gamma
        self.random_sign_init = random_sign_init
        self.constant_init = constant_init
        self.probability = p

        self.encoder_fc1 = nn.Linear(in_channels, self.hidden_size)
        self.encoder_fcmean = nn.Linear(self.hidden_size, self.hidden_size)
        self.encoder_fcvar = nn.Linear(self.hidden_size, self.hidden_size)
        #self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, 1, batch_first=True, dropout=0.2)
        self.loss_latent =0
        self.decoder_fc1 = nn.Linear(self.hidden_size, in_channels)
        if train_gamma:
            self.gamma = nn.Parameter(torch.Tensor(num_models, out_channels))
        self.num_models = num_models
        if bias:
            #self.bias = nn.Parameter(torch.Tensor(self.out_channels))
            self.bias = nn.Parameter(torch.Tensor(self.num_models, out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.first_layer = first_layer

    def reset_parameters(self):
        if self.constant_init:
            nn.init.constant_(self.alpha, 1.)
            if self.random_sign_init:
                if self.probability  == -1:
                    with torch.no_grad():
                        factor = torch.ones(
                            self.num_models, device=self.alpha.device).bernoulli_(0.5)
                        factor.mul_(2).add_(-1)
                        self.alpha.data = (self.alpha.t() * factor).t()
                        if self.train_gamma:
                            self.gamma.fill_(1.)
                            self.gamma.data = (self.gamma.t() * factor).t()
                elif self.probability  == -2:
                    with torch.no_grad():
                        positives_num = self.num_models // 2
                        factor1 = torch.Tensor([1 for i in range(positives_num)])
                        factor2 = torch.Tensor(
                            [-1 for i in range(self.num_models-positives_num)])
                        factor = torch.cat([factor1, factor2]).to(self.alpha.device)
                        self.alpha.data = (self.alpha.t() * factor).t()
                        if self.train_gamma:
                            self.gamma.fill_(1.)
                            self.gamma.data = (self.gamma.t() * factor).t()
                else:
                    with torch.no_grad():
                        self.alpha.bernoulli_(self.probability)
                        self.alpha.mul_(2).add_(-1)
                        if self.train_gamma:
                            self.gamma.bernoulli_(self.probability)
                            self.gamma.mul_(2).add_(-1)
        else:
            nn.init.normal_(self.alpha, mean=1., std=0.5)
            #nn.init.normal_(self.alpha, mean=1., std=1)
            if self.train_gamma:
                nn.init.normal_(self.gamma, mean=1., std=0.5)
                #nn.init.normal_(self.gamma, mean=1., std=1)
            if self.random_sign_init:
                with torch.no_grad():
                    alpha_coeff = torch.randint_like(self.alpha, low=0, high=2)
                    alpha_coeff.mul_(2).add_(-1)
                    self.alpha *= alpha_coeff
                    if self.train_gamma:
                        gamma_coeff = torch.randint_like(self.gamma, low=0, high=2)
                        gamma_coeff.mul_(2).add_(-1)
                        self.gamma *= gamma_coeff
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.conv.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def reparameterize(self, mu, logvar):

        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)


    def update_indices(self, indices):
        self.indices = indices

    def forward(self, x):
        embedded=F.relu(self.encoder_fc1(self.alpha))
        #embedded, _ = self.rnn(embedded.view(len(self.alpha), 1, -1))
        embedded_mean, embedded_logvar = self.encoder_fcmean(embedded), self.encoder_fcmean(embedded)
        z_embedded = self.reparameterize(embedded_mean, embedded_logvar)
        alpha_decoded = self.decoder_fc1(z_embedded)

        alpha_decoded=self.decoder_fc1(embedded.view(len(self.alpha), -1))

        if not self.training and self.first_layer:
            # Repeated pattern in test: [[A,B,C],[A,B,C]]
            x = torch.cat([x for i in range(self.num_models)], dim=0)
        else:
            MSE = F.mse_loss(alpha_decoded,self.alpha)  # F.binary_cross_entropy(alpha_decoded, self.alpha, reduction='sum')
            KLD = -0.5 * torch.sum(1 + embedded_logvar - embedded_mean.pow(2) - embedded_logvar.exp())
            self.loss_latent = MSE + KLD
        if self.train_gamma:
            num_examples_per_model = int(x.size(0) / self.num_models)
            extra = x.size(0) - (num_examples_per_model * self.num_models)

            # Repeated pattern: [[A,A],[B,B],[C,C]]
            alpha = torch.cat(
                [alpha_decoded for i in range(num_examples_per_model)],
                dim=1).view([-1, self.in_channels])
            alpha.unsqueeze_(-1).unsqueeze_(-1)
            gamma = torch.cat(
                [self.gamma for i in range(num_examples_per_model)],
                dim=1).view([-1, self.out_channels])
            gamma.unsqueeze_(-1).unsqueeze_(-1)
            if self.bias is not None:
                bias = torch.cat(
                    [self.bias for i in range(num_examples_per_model)],
                    dim=1).view([-1, self.out_channels])
                bias.unsqueeze_(-1).unsqueeze_(-1)

            if extra != 0:
                alpha = torch.cat([alpha, alpha[:extra]], dim=0)
                gamma = torch.cat([gamma, gamma[:extra]], dim=0)
                if self.bias is not None:
                    bias = torch.cat([bias, bias[:extra]], dim=0)
            result = self.conv(x*alpha)*gamma


            return result + bias if self.bias is not None else result
        else:
            num_examples_per_model = int(x.size(0) / self.num_models)
            # Repeated pattern: [[A,A],[B,B],[C,C]]
            alpha = torch.cat(
                [alpha_decoded for i in range(num_examples_per_model)],
                dim=1).view([-1, self.in_channels])
            alpha.unsqueeze_(-1).unsqueeze_(-1)

            if self.bias is not None:
                bias = torch.cat(
                    [self.bias for i in range(num_examples_per_model)],
                    dim=1).view([-1, self.out_channels])
                bias.unsqueeze_(-1).unsqueeze_(-1)
            result = self.conv(x*alpha)
            return result + bias if self.bias is not None else result



class Ensemble_BatchNorm(nn.Module):
    def __init__(self, num_features, num_models=1, eps=1e-5,
                 momentum=0.1, affine=True, track_running_stats=True,
                 constant_init=False, p=0.5, random_sign_init=False):
        super(Ensemble_BatchNorm, self).__init__()
        self.num_models = num_models
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.constant_init = constant_init
        self.probability = p
        self.random_sign_init = random_sign_init
        self.batch_norms = nn.ModuleList(
            [nn.BatchNorm1d(num_features, affine=affine)
            for i in range(num_models)])
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            for l in self.batch_norms:
                nn.init.constant_(l.bias, 0.)
                if self.constant_init:
                    nn.init.constant_(l.weight, 1.)
                    if self.random_sign_init:
                        return
                        #if self.probability  == -1:
                        #    with torch.no_grad():
                        #        factor = torch.ones(self.num_models, device=
                        #                            self.l.weight.device).bernoulli_(0.5)
                        #        factor.mul_(2).add_(-1)
                        #        l.weight = (l.weight.t() * factor).t()
                        #else:
                        #    with torch.no_grad():
                        #        l.weight.bernoulli_(self.probability)
                        #        l.weight.mul_(2).add_(-1)
                else:
                    nn.init.normal_(l.weight, mean=1., std=0.1)
                    #nn.init.normal_(l.weight, mean=1., std=1.)
                    if self.random_sign_init:
                        with torch.no_grad():
                            weight_coeff = torch.randint_like(
                                l.weight, low=0, high=2)
                            weight_coeff.mul_(2).add_(-1)
                            l.weight *= weight_coeff

    def forward(self, input):
        inputs = torch.chunk(input, self.num_models, dim=0)
        res = torch.cat(
            [l(inputs[i]) for i,l in enumerate(self.batch_norms)], dim=0)
        return res


class Ensemble_BatchNorm2d(nn.Module):
    def __init__(self, num_features, num_models=1, eps=1e-5,
                 momentum=0.1, affine=True, track_running_stats=True,
                 constant_init=False, p=0.5, random_sign_init=False):
        super(Ensemble_BatchNorm2d, self).__init__()
        self.num_models = num_models
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.constant_init = constant_init
        self.probability = p
        self.random_sign_init = random_sign_init
        self.batch_norms = nn.ModuleList(
            [nn.BatchNorm2d(num_features, affine=affine)
            for i in range(num_models)])
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            for l in self.batch_norms:
                nn.init.constant_(l.bias, 0.)
                if self.constant_init:
                    nn.init.constant_(l.weight, 1.)
                    if self.random_sign_init:
                        return
                        #if self.probability  == -1:
                        #    with torch.no_grad():
                        #        factor = torch.ones(self.num_models, device=
                        #                            self.l.weight.device).bernoulli_(0.5)
                        #        factor.mul_(2).add_(-1)
                        #        l.weight = (l.weight.t() * factor).t()
                        #else:
                        #    with torch.no_grad():
                        #        l.weight.bernoulli_(self.probability)
                        #        l.weight.mul_(2).add_(-1)
                else:
                    nn.init.normal_(l.weight, mean=1., std=0.1)
                    #nn.init.normal_(l.weight, mean=1., std=1.)
                    if self.random_sign_init:
                        with torch.no_grad():
                            weight_coeff = torch.randint_like(
                                l.weight, low=0, high=2)
                            weight_coeff.mul_(2).add_(-1)
                            l.weight *= weight_coeff

    def forward(self, input):
        inputs = torch.chunk(input, self.num_models, dim=0)
        res = torch.cat(
            [l(inputs[i]) for i,l in enumerate(self.batch_norms)], dim=0)
        return res

class Ensemble_BatchNorm_fast(nn.Module):
    def __init__(self, num_features, num_models=1, eps=1e-5,
                 momentum=0.1, affine=True, track_running_stats=True,
                 constant_init=False, random_sign_init=False):
        super(Ensemble_BatchNorm, self).__init__()
        self.num_models = num_models
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.constant_init = constant_init
        self.track_running_stats = track_running_stats
        self.random_sign_init = random_sign_init
        if self.track_running_stats:
            self.register_buffer(
                'running_mean', torch.zeros(num_models, num_features))
            self.register_buffer(
                'running_var', torch.ones(num_models, num_features))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_models, num_features))
            self.bias = nn.Parameter(torch.Tensor(num_models, num_features))
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.constant_(self.bias, 0.)
            if self.constant_init:
                nn.init.constant_(self.weight, 1.)
                if self.random_sign_init:
                    with torch.no_grad():
                        weight_coeff = torch.randint_like(
                            self.weight, low=0, high=2)
                        weight_coeff.mul_(2).add_(-1)
                        self.weight *= weight_coeff
            else:
                nn.init.normal_(self.weight, mean=1., std=0.1)
                #nn.init.normal_(self.weight, mean=1., std=1.)
                if self.random_sign_init:
                    with torch.no_grad():
                        weight_coeff = torch.randint_like(
                            self.weight, low=0, high=2)
                        weight_coeff.mul_(2).add_(-1)
                        self.weight *= weight_coeff

    def forward(self, input):
        if self.training:
            # update running estimates
            input_size = list(input.size())
            input = torch.stack(
                torch.chunk(input, self.num_models, dim=0), dim=0)
            input = input.transpose(1, 2).reshape(
                self.num_models, self.num_features, -1)
            batch_mean = torch.mean(input, dim=-1)
            batch_var = torch.var(input, dim=-1)
            input = input.view(
                [self.num_models, self.num_features,
                 input_size[0] // self.num_models,
                 *input_size[2:]]).transpose(1,2).reshape(*input_size)
            self.running_mean -= self.momentum * (
                self.running_mean - batch_mean)
            self.running_var -= self.momentum * (
                self.running_var - batch_var)

            # Forward pass.
            input = input.permute(1, 0, 2, 3)
            input = F.group_norm(
                input, self.num_models, None, None, self.eps)
            input = input.permute(1, 0, 2, 3)
            if self.affine:
                num_examples_per_model = input.size(0) // self.num_models
                weight = torch.cat(
                    [self.weight for i in range(num_examples_per_model)],
                    dim=1).view([-1, self.num_features])
                weight = weight.unsqueeze(-1).unsqueeze(-1)
                bias = torch.cat(
                    [self.bias for i in range(num_examples_per_model)],
                    dim=1).view([-1, self.num_features])
                bias = bias.unsqueeze(-1).unsqueeze(-1)
                return input * weight + bias
            else:
                return input
        else:
            inputs = torch.chunk(input, self.num_models, dim=0)
            res = torch.cat([F.batch_norm(
                inputs[i], self.running_mean[i], self.running_var[i],
                self.weights[i], self.bias[i], False, 0., self.eps)
                for i in range(self.num_models)], dim=0)
            return res


#class Ensemble_BatchNorm(nn.Module):
#    def __init__(self, num_features, num_models=1, eps=1e-5, momentum=0.1,
#                 affine=True, track_running_stats=True):
#        super(Ensemble_BatchNorm, self).__init__()
#        self.num_models = num_models
#        self.num_features = num_features
#        self.eps = eps
#        self.momentum = momentum
#        self.affine = affine
#        self.track_running_stats = track_running_stats
#        if self.affine:
#            self.weight = nn.Parameter(torch.Tensor(num_models, num_features))
#            self.bias = nn.Parameter(torch.Tensor(num_models, num_features))
#        else:
#            self.register_parameter('weight', None)
#            self.register_parameter('bias', None)
#        if self.track_running_stats:
#            self.register_buffer(
#                'running_mean', torch.zeros(num_models, num_features))
#            self.register_buffer(
#                'running_std', torch.ones(num_models, num_features))
#            self.register_buffer(
#                'num_batches_tracked', torch.tensor(0, dtype=torch.long))
#        else:
#            self.register_parameter('running_mean', None)
#            self.register_parameter('running_var', None)
#            self.register_parameter('num_batches_tracked', None)
#        self.reset_parameters()
#
#    def reset_running_stats(self):
#        if self.track_running_stats:
#            self.running_mean.zero_()
#            self.running_std.fill_(1)
#            self.num_batches_tracked.zero_()
#
#    def reset_parameters(self):
#        self.reset_running_stats
#        if self.affine:
#            nn.init.constant_(self.weight, 1.)
#            nn.init.constant_(self.bias, 0.)
#
#    def forward(self, input):
#        if self.training and self.track_running_stats:
#            if self.num_batches_tracked is not None:
#                self.num_batches_tracked += 1
#                exponential_average_factor = self.momentum
#        input_shape = list(input.size())
#        self.num_models=4
#        input = input.view(self.num_models, -1, *input_shape[1:]).transpose(1,2)
#        new_mean, new_std = torch.mean(input, dim=[2,3,4]), \
#            torch.std(input, dim=[2,3,4])
#        import pdb
#        pdb.set_trace()
#        inputs = torch.chunk(input, self.num_models, dim=0)
#
#        input = input.permute(1, 0, 2, 3)
#        input = F.group_norm(
#            input, self.num_groups, None, None, self.eps)
#        input = input.permute(1, 0, 2, 3)
#        if self.affine:
#            weight = self.weight.unsqueeze(-1).unsqueeze(-1)
#            bias = self.bias.unsqueeze(-1).unsqueeze(-1)
#            return input * weight + bias
#        else:
#            return input


class Ensemble_GroupNorm(nn.GroupNorm):
    def __init__(self, num_features, num_groups=4, eps=1e-5, momentum=0.1,
                 affine=False, track_running_stats=True):
        self.num_groups = num_groups
        super(Ensemble_GroupNorm, self).__init__(
            num_groups, num_features, eps, affine=False)
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_groups, num_features))
            #self.register_parameter('bias', None)
            self.bias = nn.Parameter(torch.Tensor(num_groups, num_features))

    def reset_parameters(self):
        if self.affine:
            nn.init.constant_(self.weight, 1.)
            nn.init.constant_(self.bias, 0.)

    def forward(self, input):
        input = input.permute(1, 0, 2, 3)
        input = F.group_norm(
            input, self.num_groups, None, None, self.eps)
        input = input.permute(1, 0, 2, 3)
        if self.affine:
            weight = self.weight.unsqueeze(-1).unsqueeze(-1)
            bias = self.bias.unsqueeze(-1).unsqueeze(-1)
            return input * weight + bias
        else:
            return input


class Ensemble_Conv2dBatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, first_layer=False, num_models=100, bias=True):
        super(Ensemble_Conv2dBatchNorm, self).__init__()
        self.conv = Ensemble_Conv2d(
            in_channels, out_channels,
            kernel_size, stride=stride,
            padding=padding, first_layer=first_layer,
            num_models=num_models, bias=True, train_gamma=True)
        #self.conv = Ensemble_Conv2d(
        #    in_channels, out_channels,
        #    kernel_size, stride=stride,
        #    padding=padding, first_layer=first_layer,
        #    num_models=num_models, bias=False, train_gamma=False)
        self.out_channels = out_channels
        #self.norm_layer = Ensemble_GroupNorm(
        #    out_channels, num_groups=num_models, affine=False)
        self.norm_layer = nn.BatchNorm2d(out_channels, affine=True)
        #self.batch_norm = Ensemble_BatchNorm2d(
        #    out_channels, affine=False, num_models=num_models)
        #self.gamma = nn.Parameter(torch.Tensor(num_models, out_channels))
        self.num_models = num_models

        self.bias = False
        #self.bias = bias

        if self.bias:
            #self.bias = nn.Parameter(torch.Tensor(self.out_channels))
            self.bias = nn.Parameter(torch.Tensor(self.num_models, out_channels))
        else:
            self.bias = None
            #self.register_parameter('bias', None)
        self.reset_parameters()
        self.first_layer = first_layer
        self.train_gamma = 'gamma' in self.__dict__

    def reset_parameters(self):
        #nn.init.constant_(self.alpha, 1.0)
        if 'gamma' in self.__dict__:
            nn.init.constant_(self.gamma, 1.0)
        #nn.init.normal_(self.gamma, mean=1., std=1)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(
                self.conv.conv.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        inter_conv = self.norm_layer(self.conv(x))
        num_examples_per_model = int(x.size(0) / self.num_models)
        if self.train_gamma:
            gamma = torch.cat(
                [self.gamma for i in range(num_examples_per_model)],
                dim=1).view([-1, self.out_channels])
            gamma.unsqueeze_(-1).unsqueeze_(-1)
        if self.bias is not None:
            bias = torch.cat(
                [self.bias for i in range(num_examples_per_model)],
                dim=1).view([-1, self.out_channels])
            bias.unsqueeze_(-1).unsqueeze_(-1)
            return (inter_conv*gamma + bias) \
                if self.train_gamma else inter_conv + bias
        else:
            return inter_conv*gamma if self.train_gamma else inter_conv



class Ensemble_Conv2dBatchNorm_pre(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, first_layer=False, num_models=100, bias=True):
        super(Ensemble_Conv2dBatchNorm_pre, self).__init__()
        self.conv = Ensemble_Conv2d(
            in_channels, out_channels,
            kernel_size, stride=stride,
            padding=padding, first_layer=first_layer,
            num_models=num_models, bias=True, train_gamma=True)
        #self.conv = Ensemble_Conv2d(
        #    in_channels, out_channels,
        #    kernel_size, stride=stride,
        #    padding=padding, first_layer=first_layer,
        #    num_models=num_models, bias=False, train_gamma=False)
        self.out_channels = out_channels
        #self.norm_layer = Ensemble_GroupNorm(
        #    out_channels, num_groups=num_models, affine=False)
        self.norm_layer = nn.BatchNorm2d(in_channels, affine=True)
        #self.batch_norm = Ensemble_BatchNorm2d(
        #    out_channels, affine=False, num_models=num_models)
        #self.gamma = nn.Parameter(torch.Tensor(num_models, out_channels))
        self.num_models = num_models

        self.bias = False
        #self.bias = bias

        if self.bias:
            #self.bias = nn.Parameter(torch.Tensor(self.out_channels))
            self.bias = nn.Parameter(torch.Tensor(self.num_models, out_channels))
        else:
            self.bias = None
            #self.register_parameter('bias', None)
        self.reset_parameters()
        self.first_layer = first_layer
        self.train_gamma = 'gamma' in self.__dict__

    def reset_parameters(self):
        #nn.init.constant_(self.alpha, 1.0)
        if 'gamma' in self.__dict__:
            nn.init.constant_(self.gamma, 1.0)
        #nn.init.normal_(self.gamma, mean=1., std=1)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(
                self.conv.conv.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        inter_conv = self.conv(self.norm_layer(x))
        num_examples_per_model = int(x.size(0) / self.num_models)
        if self.train_gamma:
            gamma = torch.cat(
                [self.gamma for i in range(num_examples_per_model)],
                dim=1).view([-1, self.out_channels])
            gamma.unsqueeze_(-1).unsqueeze_(-1)
        if self.bias is not None:
            bias = torch.cat(
                [self.bias for i in range(num_examples_per_model)],
                dim=1).view([-1, self.out_channels])
            bias.unsqueeze_(-1).unsqueeze_(-1)
            return (inter_conv*gamma + bias) \
                if self.train_gamma else inter_conv + bias
        else:
            return inter_conv*gamma if self.train_gamma else inter_conv


class Ensemble_Net(nn.Module):
    def __init__(self, num_models):
        super(Ensemble_Net, self).__init__()
        self.fc1 = Ensemble_FC(784, 500, True, num_models)
        self.fc2 = Ensemble_FC(500, 500, False, num_models)
        self.fc3 = Ensemble_FC(500, 10, False, num_models)
        self.fcs = [self.fc1, self.fc2, self.fc3]
        self.num_models = num_models
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        curr_bs = x.size(0)
        makeup_bs = abs(self.num_models - curr_bs)
        indices = torch.randint(
            high=self.num_models,
            size=(curr_bs,), device=self.fc1.fc.weight.device)
        for m_fc in self.fcs:
            m_fc.update_indices(indices)
        x = self.fc1(x.view(-1, 784))
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        if not self.training:
            return F.log_softmax(
                x.view([self.num_models, -1, 10]).mean(dim=0), dim=1)
            #return F.log_softmax(x.mean(dim=0), dim=1)
            #return F.log_softmax(x, dim=1).mean(dim=0)
        return F.log_softmax(x, dim=1)


class Ensemble_convNet(nn.Module):
    def __init__(self, num_models):
        super(Ensemble_convNet, self).__init__()
        self.conv1 = Ensemble_Conv2d(
            1, 20, 5, 1, first_layer=True, num_models=num_models)
        self.conv2 = Ensemble_Conv2d(
            20, 50, 5, 1, first_layer=False, num_models=num_models)
        self.fc1 = Ensemble_FC(4*4*50, 500, False, num_models)
        self.fc2 = Ensemble_FC(500, 10, False, num_models)
        self.convs = [self.conv1, self.conv2]
        self.fcs = [self.fc1, self.fc2]
        self.num_models = num_models
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x, visualize=False):
        curr_bs = x.size(0)
        makeup_bs = abs(self.num_models - curr_bs)
        indices = torch.randint(
            high=self.num_models,
            size=(curr_bs,), device=self.conv1.conv.weight.device)
        for m_conv in self.convs:
            m_conv.update_indices(indices)
        for m_fc in self.fcs:
            m_fc.update_indices(indices)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        if not self.training:
            if not visualize:
                return F.log_softmax(
                    x.view([self.num_models, -1, 10]).mean(dim=0), dim=1)
            else:
                x = x.view([self.num_models, -1, 10])
                return F.log_softmax(x, dim=-1).view([self.num_models, -1, 10])
            #return F.log_softmax(x.mean(dim=0), dim=1)
            #return F.log_softmax(x, dim=1).mean(dim=0)
        return F.log_softmax(x, dim=1)


class ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=20):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels, histogram=False):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        accs = []

        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                accs.append(accuracy_in_bin)
            else:
                accs.append(0.)

        if histogram:
            return ece, accs
        else:
            return ece


class NaiveEnsembleResnet(nn.Module):
    def __init__(self, n_models=4, models=[]):
        super(NaiveEnsembleResnet, self).__init__()
        self.resnets = nn.ModuleList()
        for model in models:
            self.resnets.append(model)

    def forward(self, input, visualize=False):
        outputs = [model(input) for model in self.resnets]
        outputs = torch.stack(outputs)
        if visualize:
            return outputs
        else:
            return outputs.mean(dim=0)


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class GNet(nn.Module):
    def __init__(self, opt):
        super(GNet, self).__init__()
        self.use_g_encode = opt.use_g_encode
        if self.use_g_encode:
            G = np.zeros((opt.num_domain, opt.nt))
            for i in range(opt.num_domain):
                G[i] = opt.g_encode[str(i)]
            self.G = torch.from_numpy(G).float().to(device=opt.device)
        else:
            self.fc1 = nn.Linear(opt.num_domain, opt.nh)
            self.fc_final = nn.Linear(opt.nh, opt.nt)

    def forward(self, x):
        re = x.dim() == 3
        if re:
            T, B, C = x.shape
            x = x.reshape(T * B, -1)

        if self.use_g_encode:
            x = torch.matmul(x.float(), self.G)
        else:
            x = F.relu(self.fc1(x.float()))
            # x = nn.Dropout(p=p)(x)
            x = self.fc_final(x)
        return x


class FeatureNet(nn.Module):
    def __init__(self, opt):
        super(FeatureNet, self).__init__()

        nx, nh, nt, p = opt.nx, opt.nh, opt.nt, opt.p
        self.p = p

        self.fc1 = nn.Linear(nx, nh)
        self.fc2 = nn.Linear(nh * 2, nh * 2)
        self.fc3 = nn.Linear(nh * 2, nh * 2)
        self.fc4 = nn.Linear(nh * 2, nh * 2)
        self.fc_final = nn.Linear(nh * 2, nh)

        # here I change the input to fit the change dimension
        self.fc1_var = nn.Linear(nt, nh)
        self.fc2_var = nn.Linear(nh, nh)

    def forward(self, x, t):
        re = x.dim() == 3
        if re:
            T, B, C = x.shape
            x = x.reshape(T * B, -1)
            t = t.reshape(T * B, -1)

        x = F.relu(self.fc1(x))
        t = F.relu(self.fc1_var(t))
        t = F.relu(self.fc2_var(t))

        # combine feature in the middle
        x = torch.cat((x, t), dim=1)

        # main
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc_final(x)

        if re:
            return x.reshape(T, B, -1)
        else:
            return x


class GraphDNet(nn.Module):
    """
    Generate z' for connection loss
    """

    def __init__(self, opt):
        super(GraphDNet, self).__init__()
        nh = opt.nh
        nin = nh
        self.fc3 = nn.Linear(nin, nh)
        self.bn3 = nn.BatchNorm1d(nh)

        self.fc4 = nn.Linear(nh, nh)
        self.bn4 = nn.BatchNorm1d(nh)

        self.fc5 = nn.Linear(nh, nh)
        self.bn5 = nn.BatchNorm1d(nh)

        self.fc6 = nn.Linear(nh, nh)
        self.bn6 = nn.BatchNorm1d(nh)

        self.fc7 = nn.Linear(nh, nh)
        self.bn7 = nn.BatchNorm1d(nh)

        self.fc_final = nn.Linear(nh, opt.nd_out)

        if opt.no_bn:
            self.bn3 = Identity()
            self.bn4 = Identity()
            self.bn5 = Identity()
            self.bn6 = Identity()
            self.bn7 = Identity()

    def forward(self, x):
        re = x.dim() == 3

        if re:
            T, B, C = x.shape
            x = x.reshape(T * B, -1)

        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))
        x = F.relu(self.bn5(self.fc5(x)))
        x = F.relu(self.bn6(self.fc6(x)))
        x = F.relu(self.bn7(self.fc7(x)))

        x = self.fc_final(x)

        if re:
            return x.reshape(T, B, -1)
        else:
            return x


class ResGraphDNet(nn.Module):
    """
    Generate z' for connection loss
    """

    def __init__(self, opt):
        super(ResGraphDNet, self).__init__()
        nh = opt.nh
        nin = nh
        self.fc3 = nn.Linear(nin, nh)
        self.bn3 = nn.BatchNorm1d(nh)

        self.fc4 = nn.Linear(nh, nh)
        self.bn4 = nn.BatchNorm1d(nh)

        self.fc5 = nn.Linear(nh, nh)
        self.bn5 = nn.BatchNorm1d(nh)

        self.fc6 = nn.Linear(nh, nh)
        self.bn6 = nn.BatchNorm1d(nh)

        self.fc7 = nn.Linear(nh, nh)
        self.bn7 = nn.BatchNorm1d(nh)

        self.fc8 = nn.Linear(nh, nh)
        self.bn8 = nn.BatchNorm1d(nh)

        self.fc9 = nn.Linear(nh, nh)
        self.bn9 = nn.BatchNorm1d(nh)

        self.fc10 = nn.Linear(nh, nh)
        self.bn10 = nn.BatchNorm1d(nh)

        self.fc11 = nn.Linear(nh, nh)
        self.bn11 = nn.BatchNorm1d(nh)

        self.fc_final = nn.Linear(nh, opt.nd_out)

        if opt.no_bn:
            self.bn3 = Identity()
            self.bn4 = Identity()
            self.bn5 = Identity()
            self.bn6 = Identity()
            self.bn7 = Identity()
            self.bn8 = Identity()
            self.bn9 = Identity()
            self.bn10 = Identity()
            self.bn11 = Identity()

    def forward(self, x):
        re = x.dim() == 3

        if re:
            T, B, C = x.shape
            x = x.reshape(T * B, -1)

        x = F.relu(self.bn3(self.fc3(x)))
        id1 = x
        out = F.relu(self.bn4(self.fc4(x)))
        out = self.bn5(self.fc5(out))
        x = F.relu(out + id1)

        id2 = x
        out = F.relu(self.bn6(self.fc6(x)))
        out = self.bn7(self.fc7(out))
        x = F.relu(out + id2)

        id3 = x
        out = F.relu(self.bn8(self.fc8(x)))
        out = self.bn9(self.fc9(out))
        x = F.relu(out + id3)

        id4 = x
        out = F.relu(self.bn10(self.fc10(x)))
        out = self.bn11(self.fc11(out))
        x = F.relu(out + id4)

        x = self.fc_final(x)

        if re:
            return x.reshape(T, B, -1)
        else:
            return x


class DiscNet(nn.Module):
    # Discriminator doing binary classification: source v.s. target

    def __init__(self, opt):
        super(DiscNet, self).__init__()
        nh = opt.nh

        nin = nh
        self.fc3 = nn.Linear(nin, nh)
        self.bn3 = nn.BatchNorm1d(nh)

        self.fc4 = nn.Linear(nh, nh)
        self.bn4 = nn.BatchNorm1d(nh)

        self.fc5 = nn.Linear(nh, nh)
        self.bn5 = nn.BatchNorm1d(nh)

        self.fc6 = nn.Linear(nh, nh)
        self.bn6 = nn.BatchNorm1d(nh)

        self.fc7 = nn.Linear(nh, nh)
        self.bn7 = nn.BatchNorm1d(nh)

        if opt.no_bn:
            self.bn3 = Identity()
            self.bn4 = Identity()
            self.bn5 = Identity()
            self.bn6 = Identity()
            self.bn7 = Identity()

        self.fc_final = nn.Linear(nh, 1)
        if opt.model in ["ADDA", "CUA"]:
            print("===> Discrinimator Output Activation: sigmoid")
            self.output = lambda x: torch.sigmoid(x)
        else:
            print("===> Discrinimator Output Activation: identity")
            self.output = lambda x: x

    def forward(self, x):
        re = x.dim() == 3

        if re:
            T, B, C = x.shape
            x = x.reshape(T * B, -1)

        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))
        x = F.relu(self.bn5(self.fc5(x)))
        x = F.relu(self.bn6(self.fc6(x)))
        x = F.relu(self.bn7(self.fc7(x)))
        x = self.output(self.fc_final(x))

        if re:
            return x.reshape(T, B, -1)
        else:
            return x


class ClassDiscNet(nn.Module):
    """
    Discriminator doing multi-class classification on the domain
    """

    def __init__(self, opt):
        super(ClassDiscNet, self).__init__()
        nh = opt.nh
        nc = opt.nc
        nin = nh
        nout = opt.num_domain

        if opt.cond_disc:
            print("===> Conditioned Discriminator")
            nmid = nh * 2
            self.cond = nn.Sequential(
                nn.Linear(nc, nh),
                nn.ReLU(True),
                nn.Linear(nh, nh),
                nn.ReLU(True),
            )
        else:
            print("===> Unconditioned Discriminator")
            nmid = nh
            self.cond = None

        print(f"===> Discriminator will distinguish {nout} domains")

        self.fc3 = nn.Linear(nin, nh)
        self.bn3 = nn.BatchNorm1d(nh)

        self.fc4 = nn.Linear(nmid, nh)
        self.bn4 = nn.BatchNorm1d(nh)

        self.fc5 = nn.Linear(nh, nh)
        self.bn5 = nn.BatchNorm1d(nh)

        self.fc6 = nn.Linear(nh, nh)
        self.bn6 = nn.BatchNorm1d(nh)

        self.fc7 = nn.Linear(nh, nh)
        self.bn7 = nn.BatchNorm1d(nh)

        if opt.no_bn:
            self.bn3 = Identity()
            self.bn4 = Identity()
            self.bn5 = Identity()
            self.bn6 = Identity()
            self.bn7 = Identity()

        self.fc_final = nn.Linear(nh, nout)

    def forward(self, x):
        re = x.dim() == 3

        if re:
            T, B, C = x.shape
            x = x.reshape(T * B, -1)
            # f_exp = f_exp.reshape(T * B, -1)

        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))
        x = F.relu(self.bn5(self.fc5(x)))
        x = F.relu(self.bn6(self.fc6(x)))
        x = F.relu(self.bn7(self.fc7(x)))
        # x = self.fc_final(x)
        x = F.relu(self.fc_final(x))
        x = torch.log_softmax(x, dim=1)
        if re:
            return x.reshape(T, B, -1)
        else:
            return x


class CondClassDiscNet(nn.Module):
    """
    Discriminator doing multi-class classification on the domain
    """

    def __init__(self, opt):
        super(CondClassDiscNet, self).__init__()
        nh = opt.nh
        nc = opt.nc
        nin = nh
        nout = opt.num_domain

        if opt.cond_disc:
            print("===> Conditioned Discriminator")
            nmid = nh * 2
            self.cond = nn.Sequential(
                nn.Linear(nc, nh),
                nn.ReLU(True),
                nn.Linear(nh, nh),
                nn.ReLU(True),
            )
        else:
            print("===> Unconditioned Discriminator")
            nmid = nh
            self.cond = None

        print(f"===> Discriminator will distinguish {nout} domains")

        self.fc3 = nn.Linear(nin, nh)
        self.bn3 = nn.BatchNorm1d(nh)

        self.fc4 = nn.Linear(nmid, nh)
        self.bn4 = nn.BatchNorm1d(nh)

        self.fc5 = nn.Linear(nh, nh)
        self.bn5 = nn.BatchNorm1d(nh)

        self.fc6 = nn.Linear(nh, nh)
        self.bn6 = nn.BatchNorm1d(nh)

        self.fc7 = nn.Linear(nh, nh)
        self.bn7 = nn.BatchNorm1d(nh)

        if opt.no_bn:
            self.bn3 = Identity()
            self.bn4 = Identity()
            self.bn5 = Identity()
            self.bn6 = Identity()
            self.bn7 = Identity()

        self.fc_final = nn.Linear(nh, nout)

    def forward(self, x, f_exp):
        re = x.dim() == 3

        if re:
            T, B, C = x.shape
            x = x.reshape(T * B, -1)
            f_exp = f_exp.reshape(T * B, -1)

        x = F.relu(self.bn3(self.fc3(x)))
        if self.cond is not None:
            f = self.cond(f_exp)
            x = torch.cat([x, f], dim=1)
        x = F.relu(self.bn4(self.fc4(x)))
        x = F.relu(self.bn5(self.fc5(x)))
        x = F.relu(self.bn6(self.fc6(x)))
        x = F.relu(self.bn7(self.fc7(x)))
        x = self.fc_final(x)
        x = torch.log_softmax(x, dim=1)
        if re:
            return x.reshape(T, B, -1)
        else:
            return x


class ProbDiscNet(nn.Module):
    def __init__(self, opt):
        super(ProbDiscNet, self).__init__()

        nmix = opt.nmix

        nh = opt.nh

        nin = nh
        nout = opt.dim_domain * nmix * 3

        self.fc3 = nn.Linear(nin, nh)
        self.bn3 = nn.BatchNorm1d(nh)

        self.fc4 = nn.Linear(nh, nh)
        self.bn4 = nn.BatchNorm1d(nh)

        self.fc5 = nn.Linear(nh, nh)
        self.bn5 = nn.BatchNorm1d(nh)

        self.fc6 = nn.Linear(nh, nh)
        self.bn6 = nn.BatchNorm1d(nh)

        if opt.no_bn:
            self.bn3 = Identity()
            self.bn4 = Identity()
            self.bn5 = Identity()
            self.bn6 = Identity()

        self.fc_final = nn.Linear(nh, nout)

        self.ndomain = opt.dim_domain
        self.nmix = nmix

    def forward(self, x):
        re = x.dim() == 3
        if re:
            T, B, C = x.shape
            x = x.reshape(T * B, -1)

        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))
        x = F.relu(self.bn5(self.fc5(x)))
        x = F.relu(self.bn6(self.fc6(x)))

        x = self.fc_final(x).reshape(-1, 3, self.ndomain, self.nmix)
        x_mean, x_std, x_weight = x[:, 0], x[:, 1], x[:, 2]
        x_std = torch.sigmoid(x_std) * 2 + 0.1
        x_weight = torch.softmax(x_weight, dim=1)

        if re:
            return (
                x_mean.reshape(T, B, -1),
                x_std.reshape(T, B, -1),
                x_weight.reshape(T, B, -1),
            )
        else:
            return x_mean, x_std, x_weight


class PredNet(nn.Module):
    def __init__(self, opt):
        super(PredNet, self).__init__()

        nh, nc = opt.nh, opt.nc
        nin = nh
        self.fc3 = nn.Linear(nin, nh)
        self.bn3 = nn.BatchNorm1d(nh)
        self.fc4 = nn.Linear(nh, nh)
        self.bn4 = nn.BatchNorm1d(nh)
        self.fc_final = nn.Linear(nh, nc)
        if opt.no_bn:
            self.bn3 = Identity()
            self.bn4 = Identity()

    def forward(self, x, return_softmax=False):
        re = x.dim() == 3
        if re:
            T, B, C = x.shape
            x = x.reshape(T * B, -1)

        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))
        x = self.fc_final(x)
        x_softmax = F.softmax(x, dim=1)

        # x = F.log_softmax(x, dim=1)
        # x = torch.clamp_max(x_softmax + 1e-4, 1)
        # x = torch.log(x)
        x = torch.log(x_softmax + 1e-4)

        if re:
            x = x.reshape(T, B, -1)
            x_softmax = x_softmax.reshape(T, B, -1)

        if return_softmax:
            return x, x_softmax
        else:
            return x


# ======================================================================================================================

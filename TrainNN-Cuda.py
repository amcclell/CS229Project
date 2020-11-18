import torch
import numpy
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib import rc
import copy
from timeit import default_timer as timer
import subprocess

rc('text', usetex=True)

dim = 3
reg_factor = 1.0e-8
th = 0.000076
stress_scale, strain_scale = 1.0e3, 1.0e-1

"""
Visualize the data 
"""


def MyScatter(xx, yy, yy_pred, name, stress_scale=1.0e3, strain_scale=1.0e-1, Test_or_Train="Train"):
    s, ls = 35, 33
    mke = 20
    plt.figure(figsize=(12, 10))
    ax = plt.axes()
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.scatter(xx[0:-1:mke, 0] * strain_scale, yy[0:-1:mke, 0] * stress_scale / th, label="Reference",
                facecolors='none', edgecolors='black')
    plt.scatter(xx[0:-1:mke, 0] * strain_scale, yy_pred[0:-1:mke, 0] * stress_scale / th, label=name + " model",
                facecolors='none', edgecolors='red')
    plt.xlabel(r"$E_{0}^{(11)}$", size=s, labelpad=21)
    plt.ylabel(r"$S_{0}^{(11)}$", size=s, labelpad=21)
    plt.tick_params(axis='both', labelsize=ls)
    plt.tick_params(axis='both', labelsize=ls)
    ax.yaxis.get_offset_text().set_fontsize(ls)
    plt.legend(prop={'size': ls})
    plt.tight_layout()
    plt.savefig(Test_or_Train + "_sigma_xx_" + name + ".png")
    plt.close("all")

    plt.figure(figsize=(12, 10))
    ax = plt.axes()
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.scatter(xx[0:-1:mke, 1] * strain_scale, yy[0:-1:mke, 1] * stress_scale / th, label="Reference",
                facecolors='none', edgecolors='black')
    plt.scatter(xx[0:-1:mke, 1] * strain_scale, yy_pred[0:-1:mke, 1] * stress_scale / th, label=name + " model",
                facecolors='none', edgecolors='red')
    plt.xlabel(r"$E_{0}^{(22)}$", size=s, labelpad=21)
    plt.ylabel(r"$S_{0}^{(22)}$", size=s, labelpad=21)
    plt.tick_params(axis='both', labelsize=ls)
    plt.tick_params(axis='both', labelsize=ls)
    ax.yaxis.get_offset_text().set_fontsize(ls)
    plt.legend(prop={'size': ls})
    plt.tight_layout()
    plt.savefig(Test_or_Train + "_sigma_yy_" + name + ".png")
    plt.close("all")

    plt.figure(figsize=(12, 10))
    ax = plt.axes()
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.scatter(xx[0:-1:mke, 2] * strain_scale, yy[0:-1:mke, 2] * stress_scale / th, label="Reference",
                facecolors='none', edgecolors='black')
    plt.scatter(xx[0:-1:mke, 2] * strain_scale, yy_pred[0:-1:mke, 2] * stress_scale / th, label=name + " model",
                facecolors='none', edgecolors='red')
    plt.xlabel(r"$2E_{0}^{(12)}$", size=s, labelpad=21)
    plt.ylabel(r"$S_{0}^{(12)}$", size=s, labelpad=21)
    plt.tick_params(axis='both', labelsize=ls)
    plt.tick_params(axis='both', labelsize=ls)
    ax.yaxis.get_offset_text().set_fontsize(ls)
    plt.legend(prop={'size': ls})
    plt.tight_layout()
    plt.savefig(Test_or_Train + "_sigma_xy_" + name + ".png")
    plt.close("all")


def ReadData(stress_scale=1.0e3, strain_scale=1.0e-1, SYM=False, DIR="Training_Data/"):
    dummy = numpy.loadtxt(DIR + "macro.strainxx.1", usecols=(0, 1))
    dim = 3
    ntp = dummy.shape[0]
    print("(ntp, dim) = ", ntp, " , ", dim)

    sym_n = 0
    if SYM:
        sym_n = 3
    xx = numpy.zeros((ntp * (sym_n + 1), dim))
    yy = numpy.zeros((ntp * (sym_n + 1), dim))

    xx_ori = numpy.zeros((ntp, dim))
    yy_ori = numpy.zeros((ntp, dim))

    xx_ori[:, 0] = numpy.loadtxt(DIR + "macro.strainxx.1", usecols=(1)) / strain_scale
    xx_ori[:, 1] = numpy.loadtxt(DIR + "macro.strainyy.1", usecols=(1)) / strain_scale
    xx_ori[:, 2] = numpy.loadtxt(DIR + "macro.strainxy.1", usecols=(1)) / strain_scale

    yy_ori[:, 0] = numpy.loadtxt(DIR + "macro.stressxx.1", usecols=(1)) / stress_scale
    yy_ori[:, 1] = numpy.loadtxt(DIR + "macro.stressyy.1", usecols=(1)) / stress_scale
    yy_ori[:, 2] = numpy.loadtxt(DIR + "macro.stressxy.1", usecols=(1)) / stress_scale

    print("Strain_Ranges are ", xx_ori[:, 0].min(), " ", xx_ori[:, 1].min(), " ", xx_ori[:, 2].min(), " ",
          xx_ori[:, 0].max(), " ", xx_ori[:, 1].max(), " ", xx_ori[:, 2].max())

    print("Abs_Strain_Ranges are ", numpy.fabs(xx_ori[:, 0]).min(), " ", numpy.fabs(xx_ori[:, 1]).min(), " ",
          numpy.fabs(xx_ori[:, 2]).min(), " ",
          numpy.fabs(xx_ori[:, 0]).max(), " ", numpy.fabs(xx_ori[:, 1]).max(), " ", numpy.fabs(xx_ori[:, 2]).max())

    print("Stress_Ranges are ", yy_ori[:, 0].min(), " ", yy_ori[:, 1].min(), " ", yy_ori[:, 2].min(), " ",
          yy_ori[:, 0].max(), " ", yy_ori[:, 1].max(), " ", yy_ori[:, 2].max())

    print("Abs_Stress_Ranges are ", numpy.fabs(yy_ori[:, 0]).min(), " ", numpy.fabs(yy_ori[:, 1]).min(), " ",
          numpy.fabs(yy_ori[:, 2]).min(), " ",
          numpy.fabs(yy_ori[:, 0]).max(), " ", numpy.fabs(yy_ori[:, 1]).max(), " ", numpy.fabs(yy_ori[:, 2]).max())

    xx[0:ntp, :], yy[0:ntp, :] = xx_ori, yy_ori

    if SYM:
        i = 1
        xx[i * (ntp):(i + 1) * (ntp), 0] = xx_ori[:, 0]
        xx[i * (ntp):(i + 1) * (ntp), 1] = xx_ori[:, 1]
        xx[i * (ntp):(i + 1) * (ntp), 2] = -xx_ori[:, 2]

        yy[i * (ntp):(i + 1) * (ntp), 0] = yy_ori[:, 0]
        yy[i * (ntp):(i + 1) * (ntp), 1] = yy_ori[:, 1]
        yy[i * (ntp):(i + 1) * (ntp), 2] = -yy_ori[:, 2]

        i = 2
        xx[i * (ntp):(i + 1) * (ntp), 0] = xx_ori[:, 1]
        xx[i * (ntp):(i + 1) * (ntp), 1] = xx_ori[:, 0]
        xx[i * (ntp):(i + 1) * (ntp), 2] = xx_ori[:, 2]

        yy[i * (ntp):(i + 1) * (ntp), 0] = yy_ori[:, 1]
        yy[i * (ntp):(i + 1) * (ntp), 1] = yy_ori[:, 0]
        yy[i * (ntp):(i + 1) * (ntp), 2] = yy_ori[:, 2]

        i = 3
        xx[i * (ntp):(i + 1) * (ntp), 0] = xx_ori[:, 1]
        xx[i * (ntp):(i + 1) * (ntp), 1] = xx_ori[:, 0]
        xx[i * (ntp):(i + 1) * (ntp), 2] = -xx_ori[:, 2]

        yy[i * (ntp):(i + 1) * (ntp), 0] = yy_ori[:, 1]
        yy[i * (ntp):(i + 1) * (ntp), 1] = yy_ori[:, 0]
        yy[i * (ntp):(i + 1) * (ntp), 2] = -yy_ori[:, 2]

    return xx, yy


def LinearReg(xx, yy, type, stress_scale, strain_scale):
    ntp, dim = xx.shape
    assert (dim == 3)
    if type == "Sym":
        # H * xx == yy
        #
        # h0 h1 h3
        # h1 h2 h4
        # h3 h4 h5
        #
        # X h - Y
        X = numpy.zeros((dim * ntp, 6), dtype=float)
        for i in range(ntp):
            X[3 * i, :] = xx[i, 0], xx[i, 1], 0.0, xx[i, 2], 0.0, 0.0
            X[3 * i + 1, :] = 0.0, xx[i, 0], xx[i, 1], 0.0, xx[i, 2], 0.0
            X[3 * i + 2, :] = 0.0, 0.0, 0.0, xx[i, 0], xx[i, 1], xx[i, 2]
        Y = numpy.reshape(yy, (-1))

        h = numpy.linalg.lstsq(X, Y, rcond=None)[0]

        H = numpy.array([[h[0], h[1], h[3]], [h[1], h[2], h[4]], [h[3], h[4], h[5]]])





    elif type == "Orth":
        # H * xx == yy
        #
        # h0 h1 0
        # h1 h2 0
        # 0  0 h3
        #
        # X h - Y
        X = numpy.zeros((dim * ntp, 4), dtype=float)
        for i in range(ntp):
            X[3 * i, :] = xx[i, 0], xx[i, 1], 0.0, 0.0
            X[3 * i + 1, :] = 0.0, xx[i, 0], xx[i, 1], 0.0
            X[3 * i + 2, :] = 0.0, 0.0, 0.0, xx[i, 2]
        Y = numpy.reshape(yy, (-1))

        h = numpy.linalg.lstsq(X, Y, rcond=None)[0]

        H = numpy.array([[h[0], h[1], 0.0], [h[1], h[2], 0.0], [0.0, 0.0, h[3]]])

    else:
        print("error! type= ", type)

    print("Linear H = ", H)
    res = numpy.dot(xx, H) - yy
    print("LinearReg-", type, " ||res||_fro = ", numpy.linalg.norm(res, 'fro'))
    print("LinearReg-", type, " ||res||_fro/||yy||_fro = ",
          numpy.linalg.norm(res, 'fro') / numpy.linalg.norm(yy, 'fro'), " ",
          numpy.linalg.norm(res[:, 0:1], 'fro') / numpy.linalg.norm(yy[:, 0:1], 'fro'), " ",
          numpy.linalg.norm(res[:, 1:2], 'fro') / numpy.linalg.norm(yy[:, 1:2], 'fro'), " ",
          numpy.linalg.norm(res[:, 2:3], 'fro') / numpy.linalg.norm(yy[:, 2:3], 'fro'))

    yy_pred = numpy.dot(xx, H)
    MyScatter(xx, yy, yy_pred, "Linear", stress_scale, strain_scale, "figs/Train")

    return H, h


def QuadReg(xx, yy, type, stress_scale, strain_scale):
    ntp, dim = xx.shape
    assert (dim == 3)

    if type == "Orth":
        # H * xx == yy
        #
        # h0 h1 0 h2 h3 h4 0  0  h5
        # h1 h0 0 h3 h2 h4 0  0  h5
        # 0  0 h6 0  0   0 h7 h8 0
        #
        # xx = exx eyy gxy exx*exx eyy*eyy gxy*gxy eyy*gxy exx*gxy exx*eyy
        # X h - Y
        X = numpy.zeros((dim * ntp, 9), dtype=float)
        xx_extend = numpy.zeros((ntp, 9), dtype=float)
        for i in range(ntp):
            exx, eyy, gxy = xx[i, :]
            X[3 * i, :] = exx, eyy, exx * exx, eyy * eyy, gxy * gxy, exx * eyy, 0.0, 0.0, 0.0
            X[3 * i + 1, :] = eyy, exx, eyy * eyy, exx * exx, gxy * gxy, exx * eyy, 0.0, 0.0, 0.0
            X[3 * i + 2, :] = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gxy, eyy * gxy, exx * gxy
            xx_extend[i, :] = exx, eyy, gxy, exx * exx, eyy * eyy, gxy * gxy, eyy * gxy, exx * gxy, exx * eyy
        Y = numpy.reshape(yy, (-1))

        h = numpy.linalg.lstsq(X, Y, rcond=None)[0]

        H = numpy.array([[h[0], h[1], 0.0, h[2], h[3], h[4], 0.0, 0.0, h[5]],
                         [h[1], h[0], 0.0, h[3], h[2], h[4], 0.0, 0.0, h[5]],
                         [0.0, 0.0, h[6], 0.0, 0.0, 0.0, h[7], h[8], 0.0]])

    else:
        print("error! type= ", type)

    print("Quad H = ", H)

    res = numpy.dot(xx_extend, H.T) - yy
    print("QuadReg-", type, " ||res||_fro = ", numpy.linalg.norm(res, 'fro'))
    print("QuadReg-", type, " ||res||_fro/||yy||_fro = ", numpy.linalg.norm(res, 'fro') / numpy.linalg.norm(yy, 'fro'),
          " ",
          numpy.linalg.norm(res[:, 0:1], 'fro') / numpy.linalg.norm(yy[:, 0:1], 'fro'), " ",
          numpy.linalg.norm(res[:, 1:2], 'fro') / numpy.linalg.norm(yy[:, 1:2], 'fro'), " ",
          numpy.linalg.norm(res[:, 2:3], 'fro') / numpy.linalg.norm(yy[:, 2:3], 'fro'))

    yy_pred = numpy.dot(xx_extend, H.T)
    MyScatter(xx, yy, yy_pred, "Quadratic", stress_scale, strain_scale, "figs/Train")

    return H, h


class Net_Map(torch.nn.Module):
    # constitutive relation
    # strain -> H0*strain + NN(stress)

    def __init__(self):
        super(Net_Map, self).__init__()
        self.fc1 = torch.nn.Linear(3, 6).double()
        # True/False : with/without bias
        self.fc_out = torch.nn.Linear(6, 3, True).double()

    def forward(self, x_in):
        x = x_in
        x = torch.relu(self.fc1(x))
        x = self.fc_out(x)
        return x


class Net_Modified(torch.nn.Module):
    # constitutive relation
    # strain -> H0*strain + NN(stress)

    def __init__(self):
        super(Net_Modified, self).__init__()
        # Try 1
        # self.fc1 = torch.nn.Linear(3, 6).double()
        # True/False : with/without bias
        # self.fc2 = torch.nn.Linear(6, 12, True).double()
        # self.fc3 = torch.nn.Linear(12, 24, True).double()
        # self.fc4 = torch.nn.Linear(24, 3, True).double()

        # Try 2
        # self.fc1 = torch.nn.Linear(3, 20).double()
        # self.fc2 = torch.nn.Linear(20, 40).double()
        # self.fc3 = torch.nn.Linear(40, 3).double()

        # Try 3
        # self.fc1 = torch.nn.Linear(3, 20).double()
        # self.fc2 = torch.nn.Linear(20, 10).double()
        # self.fc3 = torch.nn.Linear(10, 3).double()

        # Try 4
        # self.fc1 = torch.nn.Linear(3, 80).double()
        # self.fc2 = torch.nn.Linear(80, 160).double()
        # self.fc3 = torch.nn.Linear(160, 320).double()
        # self.fc4 = torch.nn.Linear(320, 320).double()
        # self.fc5 = torch.nn.Linear(320, 3).double()

        # Try 5
        val = 200
        self.fc1 = torch.nn.Linear(3, val).double()
        self.fc2 = torch.nn.Linear(val, val * 2).double()
        self.fc3 = torch.nn.Linear(val * 2, val).double()
        self.fc4 = torch.nn.Linear(val, val * 2).double()
        # self.fc5 = torch.nn.Linear(val*2, val).double()
        # self.fc6 = torch.nn.Linear(val, val*2).double()
        # self.fc7 = torch.nn.Linear(val*2, val).double()
        # self.fc8 = torch.nn.Linear(val, val*2).double()
        # self.fc9 = torch.nn.Linear(val*2, val).double()
        # self.fc10 = torch.nn.Linear(val, val*2).double()
        self.fc11 = torch.nn.Linear(val * 2, 3).double()

    def forward(self, x_in):
        x = x_in
        # Try 1
        # x = torch.relu(self.fc1(x))
        # x = torch.relu(self.fc2(x))
        # x = torch.relu(self.fc3(x))
        # x = self.fc4(x)

        # Try 2
        # x = torch.relu(self.fc1(x))
        # x = torch.relu(self.fc2(x))
        # x = self.fc3(x)

        # Try 3
        # x = torch.pow(self.fc1(x), 3)
        # x = torch.pow(self.fc2(x), 3)
        # x = self.fc3(x)

        # Try 4
        # x = torch.relu(self.fc1(x))
        # x = torch.relu(self.fc2(x))
        # x = torch.relu(self.fc3(x))
        # x = torch.relu(self.fc4(x))
        # x = self.fc5(x)

        # Try 5
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        # x = torch.relu(self.fc5(x))
        # x = torch.relu(self.fc6(x))
        # x = torch.relu(self.fc7(x))
        # x = torch.relu(self.fc8(x))
        # x = torch.relu(self.fc9(x))
        # x = torch.relu(self.fc10(x))
        x = self.fc11(x)
        return x


# Linear/Quadratic fit with plot
# Scale strain or stress to similar magnitude
def FitH():
    xx, yy = ReadData(stress_scale, strain_scale, True, "Training_Data/")
    xx_test, yy_test = ReadData(stress_scale, strain_scale, True, "Test_Data/")

    H, h = LinearReg(xx, yy, "Orth", stress_scale, strain_scale)
    yy_test_pred = numpy.dot(xx_test, H)
    res_test = yy_test_pred - yy_test
    print("LinearReg-Test-Error ||res||_fro/||yy||_fro = ",
          numpy.linalg.norm(res_test, 'fro') / numpy.linalg.norm(yy_test, 'fro'), " ",
          numpy.linalg.norm(res_test[:, 0:1], 'fro') / numpy.linalg.norm(yy_test[:, 0:1], 'fro'), " ",
          numpy.linalg.norm(res_test[:, 1:2], 'fro') / numpy.linalg.norm(yy_test[:, 1:2], 'fro'), " ",
          numpy.linalg.norm(res_test[:, 2:3], 'fro') / numpy.linalg.norm(yy_test[:, 2:3], 'fro'))
    MyScatter(xx_test, yy_test, yy_test_pred, "Linear", stress_scale, strain_scale, "figs/Test")

    H, h = QuadReg(xx, yy, "Orth", stress_scale, strain_scale)
    ntp = xx_test.shape[0]
    xx_test_extend = numpy.zeros((ntp, 9), dtype=float)
    for i in range(ntp):
        exx, eyy, gxy = xx_test[i, :]
        xx_test_extend[i, :] = exx, eyy, gxy, exx * exx, eyy * eyy, gxy * gxy, eyy * gxy, exx * gxy, exx * eyy

    yy_test_pred = numpy.dot(xx_test_extend, H.T)
    res_test = yy_test_pred - yy_test
    print("QuadReg-Test-Error ||res||_fro/||yy||_fro = ",
          numpy.linalg.norm(res_test, 'fro') / numpy.linalg.norm(yy_test, 'fro'), " ",
          numpy.linalg.norm(res_test[:, 0:1], 'fro') / numpy.linalg.norm(yy_test[:, 0:1], 'fro'), " ",
          numpy.linalg.norm(res_test[:, 1:2], 'fro') / numpy.linalg.norm(yy_test[:, 1:2], 'fro'), " ",
          numpy.linalg.norm(res_test[:, 2:3], 'fro') / numpy.linalg.norm(yy_test[:, 2:3], 'fro'))

    MyScatter(xx_test, yy_test, yy_test_pred, "Quadratic", stress_scale, strain_scale, "figs/Test")


def AdditiveNNModified_Train():
    name = "Net_Modified"
    model = Net_Modified()

    # Read data
    xx, yy = ReadData(stress_scale, strain_scale, True, "Training_Data/")
    xx_test, yy_test = ReadData(stress_scale, strain_scale, True, "Test_Data/")

    # Linear fit to get H, h
    H, h = LinearReg(xx, yy, "Orth", stress_scale, strain_scale)

    yy_linear_fit = numpy.dot(xx, H)
    yy_test_linear_fit = numpy.dot(xx_test, H)

    ntp, dim = xx.shape
    ntp_test, dim_test = xx_test.shape

    inputs = torch.from_numpy(xx).view(ntp, dim)
    outputs = torch.from_numpy(yy - yy_linear_fit).view(ntp, dim)
    so, muo = torch.std_mean(outputs, 0)
    outputs = (outputs - muo) / so

    device = torch.device("cuda:0")
    cpu = torch.device("cpu")
    inputs = inputs.to(device)
    outputs = outputs.to(device)
    model.to(device)

    factor = torch.tensor(reg_factor * ntp, device=device)
    loss = torch.tensor(0.).double()

    optimizer = optim.LBFGS(model.parameters(), lr=0.8, max_iter=1000, line_search_fn='strong_wolfe')

    Nite = 1000
    start = timer()
    for i in range(Nite):
        print("Iteration : ", i, end='')

        def closure():
            # L2 regularization
            l2_loss = torch.tensor(0., device=device)
            for param in model.parameters():
                l2_loss += param.norm()

            optimizer.zero_grad()
            sigma = model(inputs)

            loss1 = (torch.sum((sigma - outputs) ** 2))  # * stress_scale * stress_scale
            loss2 = (factor * l2_loss)  # * stress_scale * stress_scale
            loss = loss1 + loss2

            loss.backward(retain_graph=True)
            # print("loss , loss1, loss2 = ", loss.item(), loss1.item(), loss2.item())
            return loss

        loss_prev = loss
        optimizer.step(closure)
        loss = closure().to(cpu).double()
        print(', Loss = ', loss.item(), end='')
        end = timer()
        print(', Elapsed Time = ', end - start)
        diff = torch.abs(loss - loss_prev) / torch.abs(loss)
        if diff < 1e-6:
            break

    model.to(cpu)
    yy_pred = model(torch.from_numpy(xx).view(ntp, dim))
    yy_pred = so * yy_pred + muo
    res_train = yy - yy_linear_fit - yy_pred.data.numpy()
    print("Train fro error =",
          numpy.linalg.norm(yy - yy_linear_fit - yy_pred.data.numpy(), ord='fro') * stress_scale * stress_scale)
    print("Train fro relative error = ",
          numpy.linalg.norm(res_train, ord='fro') / numpy.linalg.norm(yy, ord='fro'), " ",
          numpy.linalg.norm(res_train[:, 0:1], ord='fro') / numpy.linalg.norm(yy[:, 0:1], ord='fro'), " ",
          numpy.linalg.norm(res_train[:, 1:2], ord='fro') / numpy.linalg.norm(yy[:, 1:2], ord='fro'), " ",
          numpy.linalg.norm(res_train[:, 2:3], ord='fro') / numpy.linalg.norm(yy[:, 2:3], ord='fro'))

    name = "NN-Modified-CUDA"
    yy_pred = yy_linear_fit + yy_pred.data.numpy()
    MyScatter(xx, yy, yy_pred, name, stress_scale, strain_scale, "figs/Train")

    ############ Test
    yy_test_pred = model(torch.from_numpy(xx_test).view(ntp_test, dim))
    yy_test_pred = so * yy_test_pred + muo
    res_test = yy_test - yy_test_linear_fit - yy_test_pred.data.numpy()
    print("Test fro error =",
          numpy.linalg.norm(res_test, ord='fro') * stress_scale * stress_scale)
    print("Test fro relative error = ",
          numpy.linalg.norm(res_test, ord='fro') / numpy.linalg.norm(yy_test, ord='fro'), " ",
          numpy.linalg.norm(res_test[:, 0:1], ord='fro') / numpy.linalg.norm(yy_test[:, 0:1], ord='fro'), " ",
          numpy.linalg.norm(res_test[:, 1:2], ord='fro') / numpy.linalg.norm(yy_test[:, 1:2], ord='fro'), " ",
          numpy.linalg.norm(res_test[:, 2:3], ord='fro') / numpy.linalg.norm(yy_test[:, 2:3], ord='fro'))

    yy_test_pred = yy_test_linear_fit + yy_test_pred.data.numpy()
    MyScatter(xx_test, yy_test, yy_test_pred, name, stress_scale, strain_scale, "figs/Test")

    ###########  Save to cpp file
    example = torch.rand([1, dim]).double()
    traced_script_module = torch.jit.trace(model, example)
    output = traced_script_module(torch.ones([1, dim]).double())
    name = "NN-Modified"
    traced_script_module.save("models/model" + name + "AdditiveCUDA.pt")


if __name__ == "__main__":
    subprocess.run(["mkdir", "-p", "figs"])
    subprocess.run(["mkdir", "-p", "models"])
    if torch.cuda.is_available():
        AdditiveNNModified_Train()
    else:
        print('Error: Execution requires CUDA')





import torch
import numpy as np
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
def MyScatter(xx, yy, yy_pred, name, stress_scale = 1.0e3, strain_scale = 1.0e-1, Test_or_Train = "Train"):
    s, ls = 35, 33
    mke=20
    plt.figure(figsize=(12, 10))
    ax = plt.axes()
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.scatter(xx[0:-1:mke, 0]*strain_scale, yy[0:-1:mke, 0] * stress_scale/th, label="Reference", facecolors='none', edgecolors='black')
    plt.scatter(xx[0:-1:mke, 0]*strain_scale, yy_pred[0:-1:mke, 0] * stress_scale/th, label= name + " model", facecolors='none', edgecolors='red')
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
    plt.scatter(xx[0:-1:mke, 1]*strain_scale, yy[0:-1:mke, 1] * stress_scale/th, label="Reference", facecolors='none', edgecolors='black')
    plt.scatter(xx[0:-1:mke, 1]*strain_scale, yy_pred[0:-1:mke, 1] * stress_scale/th, label= name + " model", facecolors='none', edgecolors='red')
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
    plt.scatter(xx[0:-1:mke, 2]*strain_scale, yy[0:-1:mke, 2] * stress_scale/th, label="Reference", facecolors='none', edgecolors='black')
    plt.scatter(xx[0:-1:mke, 2]*strain_scale, yy_pred[0:-1:mke, 2] * stress_scale/th, label= name + " model", facecolors='none', edgecolors='red')
    plt.xlabel(r"$2E_{0}^{(12)}$", size=s, labelpad=21)
    plt.ylabel(r"$S_{0}^{(12)}$", size=s, labelpad=21)
    plt.tick_params(axis='both', labelsize=ls)
    plt.tick_params(axis='both', labelsize=ls)
    ax.yaxis.get_offset_text().set_fontsize(ls)
    plt.legend(prop={'size': ls})
    plt.tight_layout()
    plt.savefig(Test_or_Train + "_sigma_xy_" + name + ".png")
    plt.close("all")


def ReadData(stress_scale = 1.0e3, strain_scale=1.0e-1, SYM = False, DIR = "Training_Data/"):

    dummy = np.loadtxt(DIR + "macro.strainxx.1", usecols = (0,1))
    dim = 3
    ntp = dummy.shape[0]
    print("(ntp, dim) = ", ntp, " , ", dim)



    sym_n = 0
    if SYM:
        sym_n = 3
    xx = np.zeros((ntp * (sym_n + 1), dim))
    yy = np.zeros((ntp * (sym_n + 1), dim))

    xx_ori = np.zeros((ntp , dim))
    yy_ori = np.zeros((ntp , dim))

    xx_ori[:, 0] = np.loadtxt(DIR + "macro.strainxx.1", usecols=(1)) / strain_scale
    xx_ori[:, 1] = np.loadtxt(DIR + "macro.strainyy.1", usecols=(1)) / strain_scale
    xx_ori[:, 2] = np.loadtxt(DIR + "macro.strainxy.1", usecols=(1)) / strain_scale

    yy_ori[:, 0] = np.loadtxt(DIR + "macro.stressxx.1", usecols=(1)) / stress_scale
    yy_ori[:, 1] = np.loadtxt(DIR + "macro.stressyy.1", usecols=(1)) / stress_scale
    yy_ori[:, 2] = np.loadtxt(DIR + "macro.stressxy.1", usecols=(1)) / stress_scale


    print("Strain_Ranges are ", xx_ori[:,0].min(), " ", xx_ori[:,1].min(), " ", xx_ori[:,2].min(), " ",
                              xx_ori[:,0].max(), " ", xx_ori[:,1].max(), " ", xx_ori[:,2].max())

    print("Abs_Strain_Ranges are ", np.fabs(xx_ori[:, 0]).min(), " ", np.fabs(xx_ori[:, 1]).min(), " ", np.fabs(xx_ori[:, 2]).min(), " ",
          np.fabs(xx_ori[:, 0]).max(), " ", np.fabs(xx_ori[:, 1]).max(), " ", np.fabs(xx_ori[:, 2]).max())

    print("Stress_Ranges are ", yy_ori[:, 0].min(), " ", yy_ori[:, 1].min(), " ", yy_ori[:, 2].min(), " ",
          yy_ori[:, 0].max(), " ", yy_ori[:, 1].max(), " ", yy_ori[:, 2].max())

    print("Abs_Stress_Ranges are ", np.fabs(yy_ori[:, 0]).min(), " ", np.fabs(yy_ori[:, 1]).min(), " ", np.fabs(yy_ori[:, 2]).min(), " ",
          np.fabs(yy_ori[:, 0]).max(), " ", np.fabs(yy_ori[:, 1]).max(), " ", np.fabs(yy_ori[:, 2]).max())

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
    assert(dim == 3)
    if type == "Sym":
        # H * xx == yy
        #
        # h0 h1 h3
        # h1 h2 h4
        # h3 h4 h5
        #
        # X h - Y
        X = np.zeros((dim*ntp, 6), dtype=float)
        for i in range(ntp):
            X[3 * i,     :] = xx[i,0],xx[i,1],0.0,    xx[i,2], 0.0,    0.0
            X[3 * i + 1, :] = 0.0,    xx[i,0],xx[i,1],0.0,    xx[i,2], 0.0
            X[3 * i + 2, :] = 0.0,    0.0,    0.0,    xx[i,0],xx[i,1], xx[i,2]
        Y = np.reshape(yy, (-1))

        h = np.linalg.lstsq(X, Y, rcond=None)[0]

        H = np.array([[h[0], h[1], h[3]],[h[1], h[2], h[4]], [h[3], h[4], h[5]]])





    elif type == "Orth":
    # H * xx == yy
    #
    # h0 h1 0
    # h1 h2 0
    # 0  0 h3
    #
    # X h - Y
        X = np.zeros((dim * ntp, 4), dtype=float)
        for i in range(ntp):
            X[3 * i, :]     = xx[i, 0], xx[i, 1], 0.0,        0.0
            X[3 * i + 1, :] = 0.0,      xx[i, 0], xx[i, 1],   0.0
            X[3 * i + 2, :] = 0.0,      0.0,      0.0,        xx[i, 2]
        Y = np.reshape(yy, (-1))

        h = np.linalg.lstsq(X, Y, rcond=None)[0]

        H = np.array([[h[0], h[1], 0.0], [h[1], h[2], 0.0], [0.0, 0.0, h[3]]])

    else:
        print("error! type= ", type)

    print("Linear H = ", H)
    res = np.dot(xx, H) - yy
    print("LinearReg-", type, " ||res||_fro = ", np.linalg.norm(res, 'fro'))
    print("LinearReg-", type, " ||res||_fro/||yy||_fro = ", np.linalg.norm(res, 'fro')/np.linalg.norm(yy, 'fro')," ", 
                                                            np.linalg.norm(res[:,0:1], 'fro')/np.linalg.norm(yy[:,0:1], 'fro')," ",
                                                            np.linalg.norm(res[:,1:2], 'fro')/np.linalg.norm(yy[:,1:2], 'fro')," ",
                                                            np.linalg.norm(res[:,2:3], 'fro')/np.linalg.norm(yy[:,2:3], 'fro'))

    yy_pred = np.dot(xx, H)
    MyScatter(xx, yy, yy_pred,"Linear", stress_scale, strain_scale, "figs/Train")

    return H, h


def QuadReg(xx, yy, type, stress_scale, strain_scale):
    ntp, dim = xx.shape
    assert(dim == 3)

    if type == "Orth":
    # H * xx == yy
    #
    # h0 h1 0 h2 h3 h4 0  0  h5
    # h1 h0 0 h3 h2 h4 0  0  h5
    # 0  0 h6 0  0   0 h7 h8 0
    #
    # xx = exx eyy gxy exx*exx eyy*eyy gxy*gxy eyy*gxy exx*gxy exx*eyy
    # X h - Y
        X = np.zeros((dim * ntp, 9), dtype=float)
        xx_extend = np.zeros((ntp, 9), dtype=float)
        for i in range(ntp):
            exx, eyy, gxy = xx[i,:]
            X[3 * i, :]     = exx, eyy, exx*exx,  eyy*eyy,   gxy*gxy, exx*eyy, 0.0, 0.0,  0.0
            X[3 * i + 1, :] = eyy, exx, eyy*eyy,  exx*exx,   gxy*gxy, exx*eyy, 0.0, 0.0,  0.0
            X[3 * i + 2, :] = 0.0, 0.0, 0.0,      0.0,       0.0,     0.0,     gxy, eyy*gxy, exx*gxy
            xx_extend[i,:] = exx, eyy, gxy, exx*exx, eyy*eyy, gxy*gxy, eyy*gxy, exx*gxy, exx*eyy
        Y = np.reshape(yy, (-1))

        h = np.linalg.lstsq(X, Y, rcond=None)[0]

        H = np.array([[h[0], h[1], 0.0, h[2], h[3], h[4], 0.0,  0.0,  h[5]],
                         [h[1], h[0], 0.0, h[3], h[2], h[4], 0.0,  0.0,  h[5]],
                         [0.0,  0.0,  h[6], 0.0,  0.0,  0.0, h[7], h[8], 0.0]])

    else:
        print("error! type= ", type)

    print("Quad H = ", H)

    res = np.dot(xx_extend, H.T) - yy
    print("QuadReg-", type, " ||res||_fro = ", np.linalg.norm(res, 'fro'))
    print("QuadReg-", type, " ||res||_fro/||yy||_fro = ", np.linalg.norm(res, 'fro')/np.linalg.norm(yy, 'fro')," ",
                                                            np.linalg.norm(res[:,0:1], 'fro')/np.linalg.norm(yy[:,0:1], 'fro')," ",
                                                            np.linalg.norm(res[:,1:2], 'fro')/np.linalg.norm(yy[:,1:2], 'fro')," ",
                                                            np.linalg.norm(res[:,2:3], 'fro')/np.linalg.norm(yy[:,2:3], 'fro'))

    yy_pred = np.dot(xx_extend, H.T)
    MyScatter(xx, yy, yy_pred, "Quadratic", stress_scale, strain_scale, "figs/Train")

    return H, h






class Net_Map(torch.nn.Module):
    # constitutive relation
    # strain -> H0*strain + NN(stress)
   
    def __init__(self):
        super(Net_Map, self).__init__()
        self.fc1 = torch.nn.Linear(3, 6).double()
        #True/False : with/without bias
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

        self.fc1 = torch.nn.Linear(3, 20).double()
        self.fc2 = torch.nn.Linear(20, 40).double()
        self.fc3 = torch.nn.Linear(40, 3).double()


    def forward(self, x_in):
        x = x_in
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Net_ModifiedNoSym(torch.nn.Module):
    # constitutive relation
    # strain -> H0*strain + NN(stress)

    def __init__(self):
        super(Net_ModifiedNoSym, self).__init__()

        self.fc1 = torch.nn.Linear(3, 6).double()
        self.fc2 = torch.nn.Linear(6, 15).double()
        # self.fc3 = torch.nn.Linear(20, 40).double()
        self.fc4 = torch.nn.Linear(15, 3).double()


    def forward(self, x_in):
        x = x_in
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        # x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class Net_Normal(torch.nn.Module):
    def __init__(self):
        super(Net_Normal, self).__init__()
        self.fc1 = torch.nn.Linear(3, 20).double()
        self.fc2 = torch.nn.Linear(20, 40).double()
        self.fc3 = torch.nn.Linear(40, 2).double()


    def forward(self, x_in):
        x = x_in
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Net_Shear(torch.nn.Module):
    def __init__(self):
        super(Net_Shear, self).__init__()
        self.fc1 = torch.nn.Linear(3, 20).double()
        self.fc2 = torch.nn.Linear(20, 40).double()
        self.fc3 = torch.nn.Linear(40, 1).double()

    def forward(self, x_in):
        x = x_in
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Linear/Quadratic fit with plot
# Scale strain or stress to similar magnitude
def FitH():
    xx, yy = ReadData(stress_scale, strain_scale, True, "Training_Data/")
    xx_test, yy_test = ReadData(stress_scale, strain_scale, True, "Test_Data/")

    H, h = LinearReg(xx, yy, "Orth", stress_scale, strain_scale)
    yy_test_pred = np.dot(xx_test, H)
    res_test = yy_test_pred - yy_test
    print("LinearReg-Test-Error ||res||_fro/||yy||_fro = ",
          np.linalg.norm(res_test, 'fro') / np.linalg.norm(yy_test, 'fro')," ",
                                                            np.linalg.norm(res_test[:,0:1], 'fro')/np.linalg.norm(yy_test[:,0:1], 'fro')," ",
                                                            np.linalg.norm(res_test[:,1:2], 'fro')/np.linalg.norm(yy_test[:,1:2], 'fro')," ",
                                                            np.linalg.norm(res_test[:,2:3], 'fro')/np.linalg.norm(yy_test[:,2:3], 'fro'))
    MyScatter(xx_test, yy_test, yy_test_pred, "Linear", stress_scale, strain_scale, "figs/Test")



    H, h = QuadReg(xx, yy, "Orth", stress_scale, strain_scale)
    ntp = xx_test.shape[0]
    xx_test_extend = np.zeros((ntp, 9), dtype=float)
    for i in range(ntp):
        exx, eyy, gxy = xx_test[i, :]
        xx_test_extend[i, :] = exx, eyy, gxy, exx * exx, eyy * eyy, gxy * gxy, eyy * gxy, exx * gxy, exx * eyy

    yy_test_pred = np.dot(xx_test_extend, H.T)
    res_test= yy_test_pred - yy_test
    print("QuadReg-Test-Error ||res||_fro/||yy||_fro = ",
          np.linalg.norm(res_test, 'fro') / np.linalg.norm(yy_test, 'fro')," ",
                                                            np.linalg.norm(res_test[:,0:1], 'fro')/np.linalg.norm(yy_test[:,0:1], 'fro')," ",
                                                            np.linalg.norm(res_test[:,1:2], 'fro')/np.linalg.norm(yy_test[:,1:2], 'fro')," ",
                                                            np.linalg.norm(res_test[:,2:3], 'fro')/np.linalg.norm(yy_test[:,2:3], 'fro'))

    MyScatter(xx_test, yy_test, yy_test_pred, "Quadratic", stress_scale, strain_scale, "figs/Test")



def AdditiveNN_Train():

    name = "Net_Map"
    model = Net_Map()
       
  
    #Read data
    xx, yy = ReadData(stress_scale, strain_scale, True, "Training_Data/")
    xx_test, yy_test = ReadData(stress_scale, strain_scale, True, "Test_Data/")
    
    #Linear fit to get H, h
    H, h = LinearReg(xx, yy, "Orth", stress_scale, strain_scale)

    yy_linear_fit = np.dot(xx, H)
    yy_test_linear_fit = np.dot(xx_test, H)

    ntp, dim = xx.shape
    ntp_test, dim_test = xx_test.shape



    inputs = torch.from_numpy(xx).view(ntp, dim)
    outputs = torch.from_numpy(yy - yy_linear_fit).view(ntp, dim)

    optimizer = optim.LBFGS(model.parameters(), lr=0.8, max_iter=1000, line_search_fn='strong_wolfe')

    Nite = 50
    for i in range(Nite):
        print("Iteration : ", i)
        def closure():
            # L2 regularization
            factor = torch.tensor(reg_factor * ntp)
            l2_loss = torch.tensor(0.)
            for param in model.parameters():
                l2_loss += param.norm()

            optimizer.zero_grad()
            sigma = model(inputs)

            loss1 = (torch.sum((sigma - outputs) ** 2)) * stress_scale * stress_scale
            loss2 = (factor * l2_loss) * stress_scale * stress_scale
            loss = loss1 + loss2
           

            loss.backward(retain_graph=True)
            print("loss , loss1, loss2 = ", loss.item(), loss1.item(), loss2.item())
            return loss
        optimizer.step(closure)

    yy_pred = model(torch.from_numpy(xx).view(ntp, dim))
    res_train = yy - yy_linear_fit - yy_pred.data.numpy()
    print("Train fro error =", np.linalg.norm(yy - yy_linear_fit - yy_pred.data.numpy(), ord='fro') * stress_scale * stress_scale)
    print("Train fro relative error = ",
          np.linalg.norm(res_train, ord='fro') / np.linalg.norm(yy, ord = 'fro'), " ", 
          np.linalg.norm(res_train[:,0:1], ord='fro') / np.linalg.norm(yy[:,0:1], ord = 'fro'), " ", 
          np.linalg.norm(res_train[:,1:2], ord='fro') / np.linalg.norm(yy[:,1:2], ord = 'fro'), " ",
          np.linalg.norm(res_train[:,2:3], ord='fro') / np.linalg.norm(yy[:,2:3], ord = 'fro'))

    
    name = "NN-ReLU"
    yy_pred = yy_linear_fit + yy_pred.data.numpy()
    MyScatter(xx, yy, yy_pred, name, stress_scale, strain_scale, "figs/Train")


    ############ Test
    yy_test_pred = model(torch.from_numpy(xx_test).view(ntp_test, dim))
    res_test = yy_test - yy_test_linear_fit - yy_test_pred.data.numpy()
    print("Test fro error =",
          np.linalg.norm(res_test, ord='fro') * stress_scale * stress_scale)
    print("Test fro relative error = ",
          np.linalg.norm(res_test, ord='fro') / np.linalg.norm(yy_test, ord='fro'), " ",
          np.linalg.norm(res_test[:,0:1], ord='fro') / np.linalg.norm(yy_test[:,0:1], ord='fro'), " ",
          np.linalg.norm(res_test[:,1:2], ord='fro') / np.linalg.norm(yy_test[:,1:2], ord='fro'), " ",
          np.linalg.norm(res_test[:,2:3], ord='fro') / np.linalg.norm(yy_test[:,2:3], ord='fro'))


    yy_test_pred =  yy_test_linear_fit + yy_test_pred.data.numpy()
    MyScatter(xx_test, yy_test, yy_test_pred, name, stress_scale, strain_scale, "figs/Test")


    ###########  Save to cpp file
    example = torch.rand([1, dim]).double()
    traced_script_module = torch.jit.trace(model, example)
    output = traced_script_module(torch.ones([1, dim]).double())
    traced_script_module.save("models/model" + name + "Additive.pt")


def AdditiveNNModified_Train():
    name = "Net_Modified"
    model = Net_Modified()

    # Read data
    xx, yy = ReadData(stress_scale, strain_scale, True, "Training_Data/")
    xx_test, yy_test = ReadData(stress_scale, strain_scale, True, "Test_Data/")

    # Linear fit to get H, h
    H, h = LinearReg(xx, yy, "Orth", stress_scale, strain_scale)

    yy_linear_fit = np.dot(xx, H)
    yy_test_linear_fit = np.dot(xx_test, H)

    ntp, dim = xx.shape
    ntp_test, dim_test = xx_test.shape

    inputs = torch.from_numpy(xx).view(ntp, dim)
    outputs = torch.from_numpy(yy - yy_linear_fit).view(ntp, dim)
    # si, mui = torch.std_mean(inputs, 0)
    so, muo = torch.std_mean(outputs, 0)
    outputs = (outputs - muo) / so

    optimizer = optim.LBFGS(model.parameters(), lr=0.8, max_iter=1000, line_search_fn='strong_wolfe')

    loss = torch.tensor(0.).double()

    Nite = 1000
    start = timer()
    for i in range(Nite):
        print("Iteration : ", i, end='')

        def closure():
            # L2 regularization
            factor = torch.tensor(reg_factor * ntp)
            l2_loss = torch.tensor(0.)
            for param in model.parameters():
                l2_loss += param.norm()

            optimizer.zero_grad()
            sigma = model(inputs)

            loss1 = (torch.sum((sigma - outputs) ** 2))  # * stress_scale * stress_scale
            loss2 = (factor * l2_loss)# * stress_scale * stress_scale
            loss = loss1 + loss2

            loss.backward(retain_graph=True)
            # print("loss , loss1, loss2 = ", loss.item(), loss1.item(), loss2.item())
            return loss

        loss_prev = loss
        optimizer.step(closure)
        loss = closure().double()

        diff = torch.abs(loss - loss_prev) / torch.abs(loss)
        end = timer()
        print(', Loss : {}, Rel. Change : {}, Elapsed Time : {}'.format(loss, diff, end - start))

        if diff < 1e-3:
            break


    yy_pred = model(torch.from_numpy(xx).view(ntp, dim))
    yy_pred = so * yy_pred + muo
    res_train = yy - yy_linear_fit - yy_pred.data.numpy()
    print("Train fro error =",
          np.linalg.norm(yy - yy_linear_fit - yy_pred.data.numpy(), ord='fro') * stress_scale * stress_scale)
    print("Train fro relative error = ",
          np.linalg.norm(res_train, ord='fro') / np.linalg.norm(yy, ord='fro'), " ",
          np.linalg.norm(res_train[:, 0:1], ord='fro') / np.linalg.norm(yy[:, 0:1], ord='fro'), " ",
          np.linalg.norm(res_train[:, 1:2], ord='fro') / np.linalg.norm(yy[:, 1:2], ord='fro'), " ",
          np.linalg.norm(res_train[:, 2:3], ord='fro') / np.linalg.norm(yy[:, 2:3], ord='fro'))

    name = "NN-Modified"
    yy_pred = yy_linear_fit + yy_pred.data.numpy()
    MyScatter(xx, yy, yy_pred, name, stress_scale, strain_scale, "figs/Train")

    ############ Test
    yy_test_pred = model(torch.from_numpy(xx_test).view(ntp_test, dim))
    yy_test_pred = so * yy_test_pred + muo
    res_test = yy_test - yy_test_linear_fit - yy_test_pred.data.numpy()
    print("Test fro error =",
          np.linalg.norm(res_test, ord='fro') * stress_scale * stress_scale)
    print("Test fro relative error = ",
          np.linalg.norm(res_test, ord='fro') / np.linalg.norm(yy_test, ord='fro'), " ",
          np.linalg.norm(res_test[:, 0:1], ord='fro') / np.linalg.norm(yy_test[:, 0:1], ord='fro'), " ",
          np.linalg.norm(res_test[:, 1:2], ord='fro') / np.linalg.norm(yy_test[:, 1:2], ord='fro'), " ",
          np.linalg.norm(res_test[:, 2:3], ord='fro') / np.linalg.norm(yy_test[:, 2:3], ord='fro'))

    yy_test_pred = yy_test_linear_fit + yy_test_pred.data.numpy()
    MyScatter(xx_test, yy_test, yy_test_pred, name, stress_scale, strain_scale, "figs/Test")

    ###########  Save to cpp file
    example = torch.rand([1, dim]).double()
    traced_script_module = torch.jit.trace(model, example)
    output = traced_script_module(torch.ones([1, dim]).double())
    traced_script_module.save("models/model" + name + "Additive.pt")

    return H, model, so, muo


def AdditiveNNSeparate_Train():
    model1 = Net_Normal()
    model2 = Net_Shear()

    model = torch.jit.load('models/modelNN-ModifiedAdditive.pt')
    model.eval()

    # m1f1w = torch.zeros(model.fc1.weight.shape).double()
    # m1f1b = torch.zeros(model.fc1.bias.shape).double()
    # m1f2w = torch.zeros(model.fc2.weight.shape).double()
    # m1f2b = torch.zeros(model.fc2.bias.shape).double()
    # m1f3w = torch.zeros((model.fc3.weight.shape[0] - 1, model.fc3.weight.shape[1])).double()
    # m1f3b = torch.zeros((model.fc3.bias.shape[0] - 1, )).double()
    #
    # m2f1w = torch.zeros(model.fc1.weight.shape).double()
    # m2f1b = torch.zeros(model.fc1.bias.shape).double()
    # m2f2w = torch.zeros(model.fc2.weight.shape).double()
    # m2f2b = torch.zeros(model.fc2.bias.shape).double()
    # m2f3w = torch.zeros((model.fc3.weight.shape[0] - 2, model.fc3.weight.shape[1])).double()
    # m2f3b = torch.zeros((model.fc3.bias.shape[0] - 2, )).double()
    #
    # for i in range(model.fc1.weight.shape[0]):
    #     for j in range(model.fc1.weight.shape[1]):
    #         m1f1w[i, j] = model.fc1.weight[i, j]
    #         m2f1w[i, j] = model.fc1.weight[i, j]
    #     m1f1b[i] = model.fc1.bias[i]
    #     m2f1b[i] = model.fc1.bias[i]
    #
    # for i in range(model.fc2.weight.shape[0]):
    #     for j in range(model.fc2.weight.shape[1]):
    #         m1f2w[i, j] = model.fc2.weight[i, j]
    #         m2f2w[i, j] = model.fc2.weight[i, j]
    #     m1f2b[i] = model.fc2.bias[i]
    #     m2f2b[i] = model.fc2.bias[i]
    #
    # for j in range(model.fc1.weight.shape[1]):
    #     m1f3w[0, j] = model.fc3.weight[0, j]
    #     m1f3w[1, j] = model.fc3.weight[1, j]
    #     m2f3w[0, j] = model.fc3.weight[2, j]
    # m1f3b[0] = model.fc3.bias[0]
    # m1f3b[1] = model.fc3.bias[1]
    # m2f3b[0] = model.fc3.bias[2]
    #
    # model1.fc1.weight = torch.nn.Parameter(m1f1w, requires_grad=True)
    # model1.fc1.bias = torch.nn.Parameter(m1f1b, requires_grad=True)
    # model1.fc2.weight = torch.nn.Parameter(m1f2w, requires_grad=True)
    # model1.fc2.bias = torch.nn.Parameter(m1f2b, requires_grad=True)
    # model1.fc3.weight = torch.nn.Parameter(m1f3w, requires_grad=True)
    # model1.fc3.bias = torch.nn.Parameter(m1f3b, requires_grad=True)
    #
    # model2.fc1.weight = torch.nn.Parameter(m2f1w, requires_grad=True)
    # model2.fc1.bias = torch.nn.Parameter(m2f1b, requires_grad=True)
    # model2.fc2.weight = torch.nn.Parameter(m2f2w, requires_grad=True)
    # model2.fc2.bias = torch.nn.Parameter(m2f2b, requires_grad=True)
    # model2.fc3.weight = torch.nn.Parameter(m2f3w, requires_grad=True)
    # model2.fc3.bias = torch.nn.Parameter(m2f3b, requires_grad=True)

    # Read data
    xx, yy = ReadData(stress_scale, strain_scale, True, "Training_Data/")
    xx_test, yy_test = ReadData(stress_scale, strain_scale, True, "Test_Data/")

    # Linear fit to get H, h
    H, h = LinearReg(xx, yy, "Orth", stress_scale, strain_scale)

    yy_linear_fit = np.dot(xx, H)
    yy_test_linear_fit = np.dot(xx_test, H)

    ntp, dim = xx.shape
    ntp_test, dim_test = xx_test.shape

    inputs = torch.from_numpy(xx).view(ntp, dim)
    outputs = torch.from_numpy(yy - yy_linear_fit).view(ntp, dim)
    # si, mui = torch.std_mean(inputs, 0)
    so, muo = torch.std_mean(outputs, 0)
    outputs = (outputs - muo) / so
    outputs1 = outputs[:, 0:2]
    outputs2 = outputs[:, 2].reshape((ntp, 1))

    optimizer = optim.LBFGS(model1.parameters(), lr=0.8, max_iter=1000, line_search_fn='strong_wolfe')

    loss = torch.tensor(0.).double()

    Nite = 1000
    print('Training Normal Stresses')
    start = timer()
    for i in range(Nite):
        print("Iteration : ", i, end='')

        def closure():
            # L2 regularization
            factor = torch.tensor(reg_factor * ntp)
            l2_loss = torch.tensor(0.)
            for param in model1.parameters():
                l2_loss += param.norm()

            optimizer.zero_grad()
            sigma = model1(inputs)

            loss1 = (torch.sum((sigma - outputs1) ** 2))  # * stress_scale * stress_scale
            loss2 = (factor * l2_loss)# * stress_scale * stress_scale
            loss = loss1 + loss2

            loss.backward(retain_graph=True)
            # print("loss , loss1, loss2 = ", loss.item(), loss1.item(), loss2.item())
            return loss

        loss_prev = loss
        optimizer.step(closure)
        loss = closure().double()

        diff = torch.abs(loss - loss_prev) / torch.abs(loss)
        end = timer()
        print(', Loss : {}, Rel. Change : {}, Elapsed Time : {}'.format(loss, diff, end - start))

        if diff < 1e-3:
            break

    optimizer = optim.LBFGS(model2.parameters(), lr=0.8, max_iter=1000, line_search_fn='strong_wolfe')
    # optimizer = optim.SGD(model2.parameters(), lr=1e-10)  # , momentum=0.9)
    loss = torch.tensor(0.).double()
    print('Training Shear Stress')
    for i in range(Nite):
        print("Iteration : ", i, end='')

        def closure():
            # L2 regularization
            factor = torch.tensor(reg_factor * ntp)
            l2_loss = torch.tensor(0.)
            for param in model2.parameters():
                l2_loss += param.norm()

            optimizer.zero_grad()
            sigma = model2(inputs)

            loss1 = (torch.sum((sigma - outputs2) ** 2))  # * stress_scale * stress_scale
            loss2 = (factor * l2_loss)  # * stress_scale * stress_scale
            loss = loss1 + loss2

            loss.backward(retain_graph=True)
            # print("loss , loss1, loss2 = ", loss.item(), loss1.item(), loss2.item())
            return loss

        loss_prev = loss
        optimizer.step(closure)
        loss = closure().double()

        diff = torch.abs(loss - loss_prev) / torch.abs(loss)
        end = timer()
        print(', Loss : {}, Rel. Change : {}, Elapsed Time : {}'.format(loss, diff, end - start))

        if diff < 1e-3:
            break

    yy_pred = torch.cat((model1(torch.from_numpy(xx).view(ntp, dim)), model2(torch.from_numpy(xx).view(ntp, dim))), 1)
    yy_pred = so * yy_pred + muo
    res_train = yy - yy_linear_fit - yy_pred.data.numpy()
    print("Train fro error =",
          np.linalg.norm(yy - yy_linear_fit - yy_pred.data.numpy(), ord='fro') * stress_scale * stress_scale)
    print("Train fro relative error = ",
          np.linalg.norm(res_train, ord='fro') / np.linalg.norm(yy, ord='fro'), " ",
          np.linalg.norm(res_train[:, 0:1], ord='fro') / np.linalg.norm(yy[:, 0:1], ord='fro'), " ",
          np.linalg.norm(res_train[:, 1:2], ord='fro') / np.linalg.norm(yy[:, 1:2], ord='fro'), " ",
          np.linalg.norm(res_train[:, 2:3], ord='fro') / np.linalg.norm(yy[:, 2:3], ord='fro'))

    name = "NN-Separate"
    yy_pred = yy_linear_fit + yy_pred.data.numpy()
    MyScatter(xx, yy, yy_pred, name, stress_scale, strain_scale, 'figs/Train')

    ############ Test
    yy_test_pred = torch.cat((model1(torch.from_numpy(xx_test).view(ntp_test, dim)), model2(torch.from_numpy(xx_test).view(ntp_test, dim))), 1)
    yy_test_pred = so * yy_test_pred + muo
    res_test = yy_test - yy_test_linear_fit - yy_test_pred.data.numpy()
    print("Test fro error =",
          np.linalg.norm(res_test, ord='fro') * stress_scale * stress_scale)
    print("Test fro relative error = ",
          np.linalg.norm(res_test, ord='fro') / np.linalg.norm(yy_test, ord='fro'), " ",
          np.linalg.norm(res_test[:, 0:1], ord='fro') / np.linalg.norm(yy_test[:, 0:1], ord='fro'), " ",
          np.linalg.norm(res_test[:, 1:2], ord='fro') / np.linalg.norm(yy_test[:, 1:2], ord='fro'), " ",
          np.linalg.norm(res_test[:, 2:3], ord='fro') / np.linalg.norm(yy_test[:, 2:3], ord='fro'))

    yy_test_pred = yy_test_linear_fit + yy_test_pred.data.numpy()
    MyScatter(xx_test, yy_test, yy_test_pred, name, stress_scale, strain_scale, "figs/Test")

    ###########  Save to cpp file
    example = torch.rand([1, dim]).double()
    traced_script_module = torch.jit.trace(model1, example)
    output = traced_script_module(torch.ones([1, dim]).double())
    traced_script_module.save("models/model" + name + "-Normal-Additive.pt")
    traced_script_module = torch.jit.trace(model2, example)
    output = traced_script_module(torch.ones([1, dim]).double())
    traced_script_module.save("models/model" + name + "-Shear-Additive.pt")


def AdditiveNNModifiedNoSym_Train():
    name = "Net_Modified-NoSym"
    model = Net_ModifiedNoSym()

    # Read data
    xx, yy = ReadData(stress_scale, strain_scale, False, "Training_Data/")
    xx_test, yy_test = ReadData(stress_scale, strain_scale, False, "Test_Data/")

    # Linear fit to get H, h
    H, h = LinearReg(xx, yy, "Orth", stress_scale, strain_scale)

    yy_linear_fit = np.dot(xx, H)
    yy_test_linear_fit = np.dot(xx_test, H)

    ntp, dim = xx.shape
    ntp_test, dim_test = xx_test.shape

    inputs = torch.from_numpy(xx).view(ntp, dim)
    outputs = torch.from_numpy(yy - yy_linear_fit).view(ntp, dim)
    # si, mui = torch.std_mean(inputs, 0)
    so, muo = torch.std_mean(outputs, 0)
    outputs = (outputs - muo) / so

    optimizer = optim.LBFGS(model.parameters(), lr=0.8, max_iter=1000, line_search_fn='strong_wolfe')

    loss = torch.tensor(0.).double()

    Nite = 1000
    start = timer()
    for i in range(Nite):
        print("Iteration : ", i, end='')

        def closure():
            # L2 regularization
            factor = torch.tensor(reg_factor * ntp)
            l2_loss = torch.tensor(0.)
            for param in model.parameters():
                l2_loss += param.norm()

            optimizer.zero_grad()
            sigma = model(inputs)

            loss1 = (torch.sum((sigma - outputs) ** 2))  # * stress_scale * stress_scale
            loss2 = (factor * l2_loss)# * stress_scale * stress_scale
            loss = loss1 + loss2

            loss.backward(retain_graph=True)
            # print("loss , loss1, loss2 = ", loss.item(), loss1.item(), loss2.item())
            return loss

        loss_prev = loss
        optimizer.step(closure)
        loss = closure().double()

        diff = torch.abs(loss - loss_prev) / torch.abs(loss)
        end = timer()
        print(', Loss : {}, Rel. Change : {}, Elapsed Time : {}'.format(loss, diff, end - start))

        if diff < 1e-3:
            break


    yy_pred = model(torch.from_numpy(xx).view(ntp, dim))
    yy_pred = so * yy_pred + muo
    res_train = yy - yy_linear_fit - yy_pred.data.numpy()
    print("Train fro error =",
          np.linalg.norm(yy - yy_linear_fit - yy_pred.data.numpy(), ord='fro') * stress_scale * stress_scale)
    print("Train fro relative error = ",
          np.linalg.norm(res_train, ord='fro') / np.linalg.norm(yy, ord='fro'), " ",
          np.linalg.norm(res_train[:, 0:1], ord='fro') / np.linalg.norm(yy[:, 0:1], ord='fro'), " ",
          np.linalg.norm(res_train[:, 1:2], ord='fro') / np.linalg.norm(yy[:, 1:2], ord='fro'), " ",
          np.linalg.norm(res_train[:, 2:3], ord='fro') / np.linalg.norm(yy[:, 2:3], ord='fro'))

    name = "NN-Modified-NoSym"
    yy_pred = yy_linear_fit + yy_pred.data.numpy()
    MyScatter(xx, yy, yy_pred, name, stress_scale, strain_scale, "figs/Train")

    ############ Test
    yy_test_pred = model(torch.from_numpy(xx_test).view(ntp_test, dim))
    yy_test_pred = so * yy_test_pred + muo
    res_test = yy_test - yy_test_linear_fit - yy_test_pred.data.numpy()
    print("Test fro error =",
          np.linalg.norm(res_test, ord='fro') * stress_scale * stress_scale)
    print("Test fro relative error = ",
          np.linalg.norm(res_test, ord='fro') / np.linalg.norm(yy_test, ord='fro'), " ",
          np.linalg.norm(res_test[:, 0:1], ord='fro') / np.linalg.norm(yy_test[:, 0:1], ord='fro'), " ",
          np.linalg.norm(res_test[:, 1:2], ord='fro') / np.linalg.norm(yy_test[:, 1:2], ord='fro'), " ",
          np.linalg.norm(res_test[:, 2:3], ord='fro') / np.linalg.norm(yy_test[:, 2:3], ord='fro'))

    yy_test_pred = yy_test_linear_fit + yy_test_pred.data.numpy()
    MyScatter(xx_test, yy_test, yy_test_pred, name, stress_scale, strain_scale, "figs/Test")

    ###########  Save to cpp file
    example = torch.rand([1, dim]).double()
    traced_script_module = torch.jit.trace(model, example)
    output = traced_script_module(torch.ones([1, dim]).double())
    traced_script_module.save("models/model" + name + "Additive.pt")


def AdditiveNN_ModifiedTrain():
    name = "Net_Modified-Training"
    model = Net_ModifiedNoSym()

    # Read data
    xx_train, yy_train = ReadData(stress_scale, strain_scale, False, "Training_Data/")
    xx_test0, yy_test0 = ReadData(stress_scale, strain_scale, False, "Test_Data/")

    # Symmetrize data without duplicating points
    inds = (xx_train[:, 2] > np.fabs(xx_train[:, 2].min()))
    tmp = np.vstack((xx_train[inds, 0], xx_train[inds, 1], -xx_train[inds, 2])).T
    xx = np.vstack((xx_train, tmp))
    tmp = np.vstack((yy_train[inds, 0], yy_train[inds, 1], -yy_train[inds, 2])).T
    yy = np.vstack((yy_train, tmp))

    inds = (xx_test0[:, 2] > np.fabs(xx_test0[:, 2].min()))
    tmp = np.vstack((xx_test0[inds, 0], xx_test0[inds, 1], -xx_test0[inds, 2])).T
    xx_test = np.vstack((xx_test0, tmp))
    tmp = np.vstack((yy_test0[inds, 0], yy_test0[inds, 1], -yy_test0[inds, 2])).T
    yy_test = np.vstack((yy_test0, tmp))

    # Linear fit to get H, h
    H, h = LinearReg(xx, yy, "Orth", stress_scale, strain_scale)

    yy_linear_fit = np.dot(xx, H)
    yy_test_linear_fit = np.dot(xx_test, H)

    ntp, dim = xx.shape
    ntp_test, dim_test = xx_test.shape

    inputs = torch.from_numpy(xx).view(ntp, dim)
    outputs = torch.from_numpy(yy - yy_linear_fit).view(ntp, dim)
    # si, mui = torch.std_mean(inputs, 0)
    so, muo = torch.std_mean(outputs, 0)
    outputs = (outputs - muo) / so

    optimizer = optim.LBFGS(model.parameters(), lr=0.8, max_iter=1000, line_search_fn='strong_wolfe')

    loss = torch.tensor(0.).double()

    Nite = 1000
    start = timer()
    for i in range(Nite):
        print("Iteration : ", i, end='')

        def closure():
            # L2 regularization
            factor = torch.tensor(reg_factor * ntp)
            l2_loss = torch.tensor(0.)
            for param in model.parameters():
                l2_loss += param.norm()

            optimizer.zero_grad()
            sigma = model(inputs)

            loss1 = (torch.sum((sigma - outputs) ** 2))  # * stress_scale * stress_scale
            loss2 = (factor * l2_loss)# * stress_scale * stress_scale
            loss = loss1 + loss2

            loss.backward(retain_graph=True)
            # print("loss , loss1, loss2 = ", loss.item(), loss1.item(), loss2.item())
            return loss

        loss_prev = loss
        optimizer.step(closure)
        loss = closure().double()

        diff = torch.abs(loss - loss_prev) / torch.abs(loss)
        end = timer()
        print(', Loss : {}, Rel. Change : {}, Elapsed Time : {}'.format(loss, diff, end - start))

        if diff < 1e-3:
            break


    yy_pred = model(torch.from_numpy(xx).view(ntp, dim))
    yy_pred = so * yy_pred + muo
    res_train = yy - yy_linear_fit - yy_pred.data.numpy()
    print("Train fro error =",
          np.linalg.norm(yy - yy_linear_fit - yy_pred.data.numpy(), ord='fro') * stress_scale * stress_scale)
    print("Train fro relative error = ",
          np.linalg.norm(res_train, ord='fro') / np.linalg.norm(yy, ord='fro'), " ",
          np.linalg.norm(res_train[:, 0:1], ord='fro') / np.linalg.norm(yy[:, 0:1], ord='fro'), " ",
          np.linalg.norm(res_train[:, 1:2], ord='fro') / np.linalg.norm(yy[:, 1:2], ord='fro'), " ",
          np.linalg.norm(res_train[:, 2:3], ord='fro') / np.linalg.norm(yy[:, 2:3], ord='fro'))

    name = "NN-Modified-Training"
    yy_pred = yy_linear_fit + yy_pred.data.numpy()
    MyScatter(xx, yy, yy_pred, name, stress_scale, strain_scale, "figs/Train")

    ############ Test
    yy_test_pred = model(torch.from_numpy(xx_test).view(ntp_test, dim))
    yy_test_pred = so * yy_test_pred + muo
    res_test = yy_test - yy_test_linear_fit - yy_test_pred.data.numpy()
    print("Test fro error =",
          np.linalg.norm(res_test, ord='fro') * stress_scale * stress_scale)
    print("Test fro relative error = ",
          np.linalg.norm(res_test, ord='fro') / np.linalg.norm(yy_test, ord='fro'), " ",
          np.linalg.norm(res_test[:, 0:1], ord='fro') / np.linalg.norm(yy_test[:, 0:1], ord='fro'), " ",
          np.linalg.norm(res_test[:, 1:2], ord='fro') / np.linalg.norm(yy_test[:, 1:2], ord='fro'), " ",
          np.linalg.norm(res_test[:, 2:3], ord='fro') / np.linalg.norm(yy_test[:, 2:3], ord='fro'))

    yy_test_pred = yy_test_linear_fit + yy_test_pred.data.numpy()
    MyScatter(xx_test, yy_test, yy_test_pred, name, stress_scale, strain_scale, "figs/Test")

    ###########  Save to cpp file
    example = torch.rand([1, dim]).double()
    traced_script_module = torch.jit.trace(model, example)
    output = traced_script_module(torch.ones([1, dim]).double())
    traced_script_module.save("models/model" + name + "Additive.pt")

def AdditiveNNModified2_Train():
    name = "Net_Modified2"
    model = Net_ModifiedNoSym()

    # Read data
    xx, yy = ReadData(stress_scale, strain_scale, True, "Training_Data/")
    xx_test, yy_test = ReadData(stress_scale, strain_scale, True, "Test_Data/")

    # Linear fit to get H, h
    H, h = LinearReg(xx, yy, "Orth", stress_scale, strain_scale)

    yy_linear_fit = np.dot(xx, H)
    yy_test_linear_fit = np.dot(xx_test, H)

    ntp, dim = xx.shape
    ntp_test, dim_test = xx_test.shape

    inputs = torch.from_numpy(xx).view(ntp, dim)
    outputs = torch.from_numpy(yy - yy_linear_fit).view(ntp, dim)
    # si, mui = torch.std_mean(inputs, 0)
    so, muo = torch.std_mean(outputs, 0)
    outputs = (outputs - muo) / so

    optimizer = optim.LBFGS(model.parameters(), lr=0.8, max_iter=1000, line_search_fn='strong_wolfe')

    loss = torch.tensor(0.).double()

    Nite = 1000
    start = timer()
    for i in range(Nite):
        print("Iteration : ", i, end='')

        def closure():
            # L2 regularization
            factor = torch.tensor(reg_factor * ntp)
            l2_loss = torch.tensor(0.)
            for param in model.parameters():
                l2_loss += param.norm()

            optimizer.zero_grad()
            sigma = model(inputs)

            loss1 = (torch.sum((sigma - outputs) ** 2))  # * stress_scale * stress_scale
            loss2 = (factor * l2_loss)# * stress_scale * stress_scale
            loss = loss1 + loss2

            loss.backward(retain_graph=True)
            # print("loss , loss1, loss2 = ", loss.item(), loss1.item(), loss2.item())
            return loss

        loss_prev = loss
        optimizer.step(closure)
        loss = closure().double()

        diff = torch.abs(loss - loss_prev) / torch.abs(loss)
        end = timer()
        print(', Loss : {}, Rel. Change : {}, Elapsed Time : {}'.format(loss, diff, end - start))

        if diff < 1e-3:
            break


    yy_pred = model(torch.from_numpy(xx).view(ntp, dim))
    yy_pred = so * yy_pred + muo
    res_train = yy - yy_linear_fit - yy_pred.data.numpy()
    print("Train fro error =",
          np.linalg.norm(yy - yy_linear_fit - yy_pred.data.numpy(), ord='fro') * stress_scale * stress_scale)
    print("Train fro relative error = ",
          np.linalg.norm(res_train, ord='fro') / np.linalg.norm(yy, ord='fro'), " ",
          np.linalg.norm(res_train[:, 0:1], ord='fro') / np.linalg.norm(yy[:, 0:1], ord='fro'), " ",
          np.linalg.norm(res_train[:, 1:2], ord='fro') / np.linalg.norm(yy[:, 1:2], ord='fro'), " ",
          np.linalg.norm(res_train[:, 2:3], ord='fro') / np.linalg.norm(yy[:, 2:3], ord='fro'))

    name = "NN-Modified2"
    yy_pred = yy_linear_fit + yy_pred.data.numpy()
    MyScatter(xx, yy, yy_pred, name, stress_scale, strain_scale, "figs/Train")

    ############ Test
    yy_test_pred = model(torch.from_numpy(xx_test).view(ntp_test, dim))
    yy_test_pred = so * yy_test_pred + muo
    res_test = yy_test - yy_test_linear_fit - yy_test_pred.data.numpy()
    print("Test fro error =",
          np.linalg.norm(res_test, ord='fro') * stress_scale * stress_scale)
    print("Test fro relative error = ",
          np.linalg.norm(res_test, ord='fro') / np.linalg.norm(yy_test, ord='fro'), " ",
          np.linalg.norm(res_test[:, 0:1], ord='fro') / np.linalg.norm(yy_test[:, 0:1], ord='fro'), " ",
          np.linalg.norm(res_test[:, 1:2], ord='fro') / np.linalg.norm(yy_test[:, 1:2], ord='fro'), " ",
          np.linalg.norm(res_test[:, 2:3], ord='fro') / np.linalg.norm(yy_test[:, 2:3], ord='fro'))

    yy_test_pred = yy_test_linear_fit + yy_test_pred.data.numpy()
    MyScatter(xx_test, yy_test, yy_test_pred, name, stress_scale, strain_scale, "figs/Test")

    ###########  Save to cpp file
    example = torch.rand([1, dim]).double()
    traced_script_module = torch.jit.trace(model, example)
    output = traced_script_module(torch.ones([1, dim]).double())
    traced_script_module.save("models/model" + name + "Additive.pt")

if __name__ == "__main__":

    # FitH()

    subprocess.run(["mkdir", "-p", "figs"])
    subprocess.run(["mkdir", "-p", "models"])

    AdditiveNN_Train()
    AdditiveNNModified_Train()
    AdditiveNNSeparate_Train()
    AdditiveNNModifiedNoSym_Train()
    AdditiveNNModified2_Train()
    AdditiveNN_ModifiedTrain()





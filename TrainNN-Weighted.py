import torch
import numpy
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)



dim = 3
reg_factor = 1.0e-8
th = 0.000076
neuron = 20
def MyScatter(xx, yy, yy_pred, name, stress_scale = 1.0e3, strain_scale = 1.0e-1, Test_or_Train = "Train"):
    s, ls = 35, 33
    mke=20
    plt.figure(figsize=(12, 10))
    ax = plt.axes()
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.scatter(xx[0:-1:mke, 0]*strain_scale, yy[0:-1:mke, 0] * stress_scale/th, label="Reference", facecolors='none', edgecolors='black')
    plt.scatter(xx[0:-1:mke, 0]*strain_scale, yy_pred[0:-1:mke, 0] * stress_scale/th, label= name + " model (weighted " + str(neuron)+ ")", facecolors='none', edgecolors='red')
    plt.xlabel(r"$E_{0}^{(11)}$", size=s, labelpad=21)
    plt.ylabel(r"$S_{0}^{(11)}$", size=s, labelpad=21)
    plt.tick_params(axis='both', labelsize=ls)
    plt.tick_params(axis='both', labelsize=ls)
    ax.yaxis.get_offset_text().set_fontsize(ls)
    plt.legend(prop={'size': ls})
    plt.tight_layout()
    plt.savefig(Test_or_Train + "_sigma_xx_" + name +  "weighted_" + str(neuron) + ".png")
    plt.close("all")


    plt.figure(figsize=(12, 10))
    ax = plt.axes()
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.scatter(xx[0:-1:mke, 1]*strain_scale, yy[0:-1:mke, 1] * stress_scale/th, label="Reference", facecolors='none', edgecolors='black')
    plt.scatter(xx[0:-1:mke, 1]*strain_scale, yy_pred[0:-1:mke, 1] * stress_scale/th, label= name + " model (weighted " + str(neuron)+ ")", facecolors='none', edgecolors='red')
    plt.xlabel(r"$E_{0}^{(22)}$", size=s, labelpad=21)
    plt.ylabel(r"$S_{0}^{(22)}$", size=s, labelpad=21)
    plt.tick_params(axis='both', labelsize=ls)
    plt.tick_params(axis='both', labelsize=ls)
    ax.yaxis.get_offset_text().set_fontsize(ls)
    plt.legend(prop={'size': ls})
    plt.tight_layout()
    plt.savefig(Test_or_Train + "_sigma_yy_" + name +  "weighted_" + str(neuron) + ".png")
    plt.close("all")

    plt.figure(figsize=(12, 10))
    ax = plt.axes()
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.scatter(xx[0:-1:mke, 2]*strain_scale, yy[0:-1:mke, 2] * stress_scale/th, label="Reference", facecolors='none', edgecolors='black')
    plt.scatter(xx[0:-1:mke, 2]*strain_scale, yy_pred[0:-1:mke, 2] * stress_scale/th, label= name + " model (weighted " + str(neuron)+ ")", facecolors='none', edgecolors='red')
    plt.xlabel(r"$2E_{0}^{(12)}$", size=s, labelpad=21)
    plt.ylabel(r"$S_{0}^{(12)}$", size=s, labelpad=21)
    plt.tick_params(axis='both', labelsize=ls)
    plt.tick_params(axis='both', labelsize=ls)
    ax.yaxis.get_offset_text().set_fontsize(ls)
    plt.legend(prop={'size': ls})
    plt.tight_layout()
    plt.savefig(Test_or_Train + "_sigma_xy_" + name +  "weighted_" + str(neuron) + ".png")
    plt.close("all")


def ArtificialData(box_low, box_up, strain_scale, ntp=20000):
    dim = 3
    print("(ntp, dim) = ", ntp, " , ", dim)

    xx = numpy.zeros((ntp, dim))
    yy = numpy.zeros((ntp, dim))

    for i in range(dim):
        xx[:, i] = numpy.random.uniform(box_low[i], box_up[i], ntp)

    return xx / strain_scale, yy


def TrainingData(stress_scale = 1.0e3, strain_scale=1.0e-1, SYM = False, DIR = "NN_data_2/"):

    dummy = numpy.loadtxt(DIR + "macro.strainxx.1", usecols = (0,1))
    dim = 3
    ntp = dummy.shape[0]
    print("(ntp, dim) = ", ntp, " , ", dim)



    sym_n = 0
    if SYM:
        sym_n = 3
    xx = numpy.zeros((ntp * (sym_n + 1), dim))
    yy = numpy.zeros((ntp * (sym_n + 1), dim))

    xx_ori = numpy.zeros((ntp , dim))
    yy_ori = numpy.zeros((ntp , dim))

    xx_ori[:, 0] = numpy.loadtxt(DIR + "macro.strainxx.1", usecols=(1)) / strain_scale
    xx_ori[:, 1] = numpy.loadtxt(DIR + "macro.strainyy.1", usecols=(1)) / strain_scale
    xx_ori[:, 2] = numpy.loadtxt(DIR + "macro.strainxy.1", usecols=(1)) / strain_scale

    yy_ori[:, 0] = numpy.loadtxt(DIR + "macro.stressxx.1", usecols=(1)) / stress_scale
    yy_ori[:, 1] = numpy.loadtxt(DIR + "macro.stressyy.1", usecols=(1)) / stress_scale
    yy_ori[:, 2] = numpy.loadtxt(DIR + "macro.stressxy.1", usecols=(1)) / stress_scale


    print("Strain_Ranges are ", xx_ori[:,0].min(), " ", xx_ori[:,1].min(), " ", xx_ori[:,2].min(), " ",
                              xx_ori[:,0].max(), " ", xx_ori[:,1].max(), " ", xx_ori[:,2].max())

    print("Abs_Strain_Ranges are ", numpy.fabs(xx_ori[:, 0]).min(), " ", numpy.fabs(xx_ori[:, 1]).min(), " ", numpy.fabs(xx_ori[:, 2]).min(), " ",
          numpy.fabs(xx_ori[:, 0]).max(), " ", numpy.fabs(xx_ori[:, 1]).max(), " ", numpy.fabs(xx_ori[:, 2]).max())

    print("Stress_Ranges are ", yy_ori[:, 0].min(), " ", yy_ori[:, 1].min(), " ", yy_ori[:, 2].min(), " ",
          yy_ori[:, 0].max(), " ", yy_ori[:, 1].max(), " ", yy_ori[:, 2].max())

    print("Abs_Stress_Ranges are ", numpy.fabs(yy_ori[:, 0]).min(), " ", numpy.fabs(yy_ori[:, 1]).min(), " ", numpy.fabs(yy_ori[:, 2]).min(), " ",
          numpy.fabs(yy_ori[:, 0]).max(), " ", numpy.fabs(yy_ori[:, 1]).max(), " ", numpy.fabs(yy_ori[:, 2]).max())

    print("Abs_Stress_Ave are ",  numpy.average(numpy.fabs(yy_ori[:, 0])), " ",  numpy.average(numpy.fabs(yy_ori[:, 1])), " ",  numpy.average(numpy.fabs(yy_ori[:, 2])))


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


    # Generate artificial data with yy_a=0 outside the inner box
    # number of fake data, and repeat the real data
    ntp_a = 0#100000

    box_low, box_up = [-0.002, -0.002, -0.004], [0.002, 0.002, 0.004]
    xx_a, yy_a = ArtificialData(box_low, box_up, strain_scale, ntp_a)

    return xx, yy, xx_a, yy_a




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
        X = numpy.zeros((dim*ntp, 6), dtype=float)
        for i in range(ntp):
            X[3 * i,     :] = xx[i,0],xx[i,1],0.0,    xx[i,2], 0.0,    0.0
            X[3 * i + 1, :] = 0.0,    xx[i,0],xx[i,1],0.0,    xx[i,2], 0.0
            X[3 * i + 2, :] = 0.0,    0.0,    0.0,    xx[i,0],xx[i,1], xx[i,2]
        Y = numpy.reshape(yy, (-1))

        h = numpy.linalg.lstsq(X, Y, rcond=None)[0]

        H = numpy.array([[h[0], h[1], h[3]],[h[1], h[2], h[4]], [h[3], h[4], h[5]]])





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
            X[3 * i, :]     = xx[i, 0], xx[i, 1], 0.0,        0.0
            X[3 * i + 1, :] = 0.0,      xx[i, 0], xx[i, 1],   0.0
            X[3 * i + 2, :] = 0.0,      0.0,      0.0,        xx[i, 2]
        Y = numpy.reshape(yy, (-1))

        h = numpy.linalg.lstsq(X, Y, rcond=None)[0]

        H = numpy.array([[h[0], h[1], 0.0], [h[1], h[2], 0.0], [0.0, 0.0, h[3]]])

    else:
        print("error! type= ", type)

    print("Linear H = ", H)
    res = numpy.dot(xx, H) - yy
    print("LinearReg-", type, " ||res||_fro = ", numpy.linalg.norm(res, 'fro'))
    print("LinearReg-", type, " ||res||_fro/||yy||_fro = ", numpy.linalg.norm(res, 'fro')/numpy.linalg.norm(yy, 'fro')," ", 
                                                            numpy.linalg.norm(res[:,0:1], 'fro')/numpy.linalg.norm(yy[:,0:1], 'fro')," ",
                                                            numpy.linalg.norm(res[:,1:2], 'fro')/numpy.linalg.norm(yy[:,1:2], 'fro')," ",
                                                            numpy.linalg.norm(res[:,2:3], 'fro')/numpy.linalg.norm(yy[:,2:3], 'fro'))

    yy_pred = numpy.dot(xx, H)
    MyScatter(xx, yy, yy_pred,"Linear", stress_scale, strain_scale)

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
        X = numpy.zeros((dim * ntp, 9), dtype=float)
        xx_extend = numpy.zeros((ntp, 9), dtype=float)
        for i in range(ntp):
            exx, eyy, gxy = xx[i,:]
            X[3 * i, :]     = exx, eyy, exx*exx,  eyy*eyy,   gxy*gxy, exx*eyy, 0.0, 0.0,  0.0
            X[3 * i + 1, :] = eyy, exx, eyy*eyy,  exx*exx,   gxy*gxy, exx*eyy, 0.0, 0.0,  0.0
            X[3 * i + 2, :] = 0.0, 0.0, 0.0,      0.0,       0.0,     0.0,     gxy, eyy*gxy, exx*gxy
            xx_extend[i,:] = exx, eyy, gxy, exx*exx, eyy*eyy, gxy*gxy, eyy*gxy, exx*gxy, exx*eyy
        Y = numpy.reshape(yy, (-1))

        h = numpy.linalg.lstsq(X, Y, rcond=None)[0]

        H = numpy.array([[h[0], h[1], 0.0, h[2], h[3], h[4], 0.0,  0.0,  h[5]],
                         [h[1], h[0], 0.0, h[3], h[2], h[4], 0.0,  0.0,  h[5]],
                         [0.0,  0.0,  h[6], 0.0,  0.0,  0.0, h[7], h[8], 0.0]])

    else:
        print("error! type= ", type)

    print("Quad H = ", H)

    res = numpy.dot(xx_extend, H.T) - yy
    print("QuadReg-", type, " ||res||_fro = ", numpy.linalg.norm(res, 'fro'))
    print("QuadReg-", type, " ||res||_fro/||yy||_fro = ", numpy.linalg.norm(res, 'fro')/numpy.linalg.norm(yy, 'fro')," ",
                                                            numpy.linalg.norm(res[:,0:1], 'fro')/numpy.linalg.norm(yy[:,0:1], 'fro')," ",
                                                            numpy.linalg.norm(res[:,1:2], 'fro')/numpy.linalg.norm(yy[:,1:2], 'fro')," ",
                                                            numpy.linalg.norm(res[:,2:3], 'fro')/numpy.linalg.norm(yy[:,2:3], 'fro'))

    yy_pred = numpy.dot(xx_extend, H.T)
    MyScatter(xx, yy, yy_pred, "Quadratic", stress_scale, strain_scale)

    return H, h



class Net_Map(torch.nn.Module):
    # constitutive relation
    # strain -> H0*strain + NN(stress)
   
    def __init__(self):
        super(Net_Map, self).__init__()
        self.fc1 = torch.nn.Linear(3, neuron).double()
        #self.fc2 = torch.nn.Linear(neuron, neuron).double()
        self.fc_out = torch.nn.Linear(neuron, 3, True).double()

    def forward(self, x_in):
        x = x_in
        x = torch.relu(self.fc1(x))
        #x = torch.relu(self.fc2(x))
        x = self.fc_out(x)

        return x


def FitH(stress_scale, strain_scale):
    xx, yy, _, _ = TrainingData(stress_scale, strain_scale, True, "NN_data_2/")
    xx_test, yy_test, _, _ = TrainingData(stress_scale, strain_scale, True, "NN_data_2_test/")



    H, h = LinearReg(xx, yy, "Orth", stress_scale, strain_scale)
    yy_test_pred = numpy.dot(xx_test, H)
    res_test = yy_test_pred - yy_test
    print("LinearReg-Test-Error ||res||_fro/||yy||_fro = ",
          numpy.linalg.norm(res_test, 'fro') / numpy.linalg.norm(yy_test, 'fro')," ",
                                                            numpy.linalg.norm(res_test[:,0:1], 'fro')/numpy.linalg.norm(yy_test[:,0:1], 'fro')," ",
                                                            numpy.linalg.norm(res_test[:,1:2], 'fro')/numpy.linalg.norm(yy_test[:,1:2], 'fro')," ",
                                                            numpy.linalg.norm(res_test[:,2:3], 'fro')/numpy.linalg.norm(yy_test[:,2:3], 'fro'))
    MyScatter(xx_test, yy_test, yy_test_pred, "Linear", stress_scale, strain_scale, "Test")



    H, h = QuadReg(xx, yy, "Orth", stress_scale, strain_scale)
    ntp = xx_test.shape[0]
    xx_test_extend = numpy.zeros((ntp, 9), dtype=float)
    for i in range(ntp):
        exx, eyy, gxy = xx_test[i, :]
        xx_test_extend[i, :] = exx, eyy, gxy, exx * exx, eyy * eyy, gxy * gxy, eyy * gxy, exx * gxy, exx * eyy

    yy_test_pred = numpy.dot(xx_test_extend, H.T)
    res_test= yy_test_pred - yy_test
    print("QuadReg-Test-Error ||res||_fro/||yy||_fro = ",
          numpy.linalg.norm(res_test, 'fro') / numpy.linalg.norm(yy_test, 'fro')," ",
                                                            numpy.linalg.norm(res_test[:,0:1], 'fro')/numpy.linalg.norm(yy_test[:,0:1], 'fro')," ",
                                                            numpy.linalg.norm(res_test[:,1:2], 'fro')/numpy.linalg.norm(yy_test[:,1:2], 'fro')," ",
                                                            numpy.linalg.norm(res_test[:,2:3], 'fro')/numpy.linalg.norm(yy_test[:,2:3], 'fro'))

    MyScatter(xx_test, yy_test, yy_test_pred, "Quadratic", stress_scale, strain_scale, "Test")



def AdditiveNN_Train(name):


    if name == "Net_Map":
        model = Net_Map()
    elif name == "Net_OrthMat":
        model = Net_OrthMat()
    elif name == "Net_OrthMatS":
        model = Net_OrthMatS()

    else:
        print("NN has not implemented yet: ", name)
   

    stress_scale, strain_scale = 1.0e3, 1.0e-1

    #Read data
    xx, yy, xx_a, yy_a = TrainingData(stress_scale, strain_scale, True, "NN_data_2/")
    xx_test, yy_test, xx_test_a, yy_test_a = TrainingData(stress_scale, strain_scale, True, "NN_data_2_test/")
    
    #Linear fit to get H, h
    H, h = LinearReg(xx, yy, "Orth", stress_scale, strain_scale)


    yy_linear_fit = numpy.dot(xx, H)
    yy_test_linear_fit = numpy.dot(xx_test, H)


    ntp, dim = xx.shape
    ntp_a, dim = xx_a.shape
    ntp_test, dim_test = xx_test.shape



    inputs = torch.from_numpy(numpy.concatenate((xx, xx_a))).view(ntp + ntp_a, dim)
    outputs = torch.from_numpy(numpy.concatenate((yy - yy_linear_fit, yy_a))).view(ntp + ntp_a, dim)

    print(outputs)

    optimizer = optim.LBFGS(model.parameters(), lr=0.8, max_iter=1000, line_search_fn='strong_wolfe')

    Nite = 50
    for i in range(Nite):
        print("Iteration : ", i)
        def closure():
            # L2 regularization
            factor = torch.tensor(reg_factor * (ntp + ntp_a))
            l2_loss = torch.tensor(0.)
            for param in model.parameters():
                l2_loss += param.norm()

            optimizer.zero_grad()
            sigma = model(inputs)

            loss1 = (torch.sum((sigma - outputs) ** 2 * torch.Tensor([1.0,1.0,4.0e4]))) * stress_scale * stress_scale
            loss2 = (factor * l2_loss) * stress_scale * stress_scale
            loss = loss1 + loss2
           

            loss.backward(retain_graph=True)
            print("loss , loss1, loss2 = ", loss.item(), loss1.item(), loss2.item())
            return loss
        optimizer.step(closure)

    yy_pred = model(torch.from_numpy(xx).view(ntp, dim))
    res_train = yy - yy_linear_fit - yy_pred.data.numpy()
    print("Train fro error =", numpy.linalg.norm(yy - yy_linear_fit - yy_pred.data.numpy(), ord='fro') * stress_scale * stress_scale)
    print("Train fro relative error = ",
          numpy.linalg.norm(res_train, ord='fro') / numpy.linalg.norm(yy, ord = 'fro'), " ", 
          numpy.linalg.norm(res_train[:,0:1], ord='fro') / numpy.linalg.norm(yy[:,0:1], ord = 'fro'), " ", 
          numpy.linalg.norm(res_train[:,1:2], ord='fro') / numpy.linalg.norm(yy[:,1:2], ord = 'fro'), " ",
          numpy.linalg.norm(res_train[:,2:3], ord='fro') / numpy.linalg.norm(yy[:,2:3], ord = 'fro'))

    #name = "NN-tanh"
    name = "NN-ReLU"
    yy_pred = yy_linear_fit + yy_pred.data.numpy()
    MyScatter(xx, yy, yy_pred, name, stress_scale, strain_scale)


    ############ Test
    yy_test_pred = model(torch.from_numpy(xx_test).view(ntp_test, dim))
    res_test = yy_test - yy_test_linear_fit - yy_test_pred.data.numpy()
    print("Test fro error =",
          numpy.linalg.norm(res_test, ord='fro') * stress_scale * stress_scale)
    print("Test fro relative error = ",
          numpy.linalg.norm(res_test, ord='fro') / numpy.linalg.norm(yy_test, ord='fro'), " ",
          numpy.linalg.norm(res_test[:,0:1], ord='fro') / numpy.linalg.norm(yy_test[:,0:1], ord='fro'), " ",
          numpy.linalg.norm(res_test[:,1:2], ord='fro') / numpy.linalg.norm(yy_test[:,1:2], ord='fro'), " ",
          numpy.linalg.norm(res_test[:,2:3], ord='fro') / numpy.linalg.norm(yy_test[:,2:3], ord='fro'))


    yy_test_pred =  yy_test_linear_fit + yy_test_pred.data.numpy()
    MyScatter(xx_test, yy_test, yy_test_pred, name, stress_scale, strain_scale, "Test")



    ###########################

    zeros_pred = model(torch.from_numpy(numpy.array([[0.0,0.0,0.0]])).view(1, dim))
    print("sigma(zero) = ", zeros_pred)

    for param in model.parameters():
        print(param.data)

    print("H = ", H * stress_scale / strain_scale)


    example = torch.rand([1, dim]).double()
    traced_script_module = torch.jit.trace(model, example)
    output = traced_script_module(torch.ones([1, dim]).double())
    traced_script_module.save("model" + name + "Additive.pt")




if __name__ == "__main__":

    FitH(1.0e3, 1.0e-1)
    AdditiveNN_Train("Net_Map")
    # #names = [ "Net_Map", "Net_OrthMat"]
    # names = [ "Net_Map"]
    # #names = ["Net_OrthMat"]
    # for name in names:
    #     print("=========", name, "Additive NN")
    #     AdditiveNN_Train(name)
    #     # print("=========", name, "Direct NN")
    #     DirectNN_Train(name)

    #DirectIncrNN_Train("IncrNet_CholOrthMat")




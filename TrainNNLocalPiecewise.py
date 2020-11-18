import torch
import numpy
import torch.optim as optim
import matplotlib.pyplot as plt
import time
from matplotlib import rc
rc('text', usetex=True)
torch.autograd.set_detect_anomaly(True)

iidx = 0
N_CATS = 2
classify = False 
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
    """
    f, ax = plt.subplots(3,3)
    for i in range(3):
      for j in range(3):
        ax[j,i].scatter(xx[0:-1:mke, i]*strain_scale, (yy[0:-1:mke, j] - yy_pred[0:-1:mke,j]) * stress_scale/th, label= name + " model", facecolors='none', edgecolors='red')
        #ax[j,i].scatter(xx[0:-1:mke, i]*strain_scale, (yy[0:-1:mke, j]) * stress_scale/th, label= name + " model", facecolors='none', edgecolors='red')
    plt.show()
    exit()
    """
    plt.figure(figsize=(12, 10))
    ax = plt.axes()
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.scatter(xx[0:-1:mke, 0]*strain_scale, yy[0:-1:mke, 0] * stress_scale/th, label="Reference", facecolors='none', edgecolors='black')
    plt.scatter(xx[0:-1:mke, 0]*strain_scale, yy_pred[0:-1:mke, 0] * stress_scale/th, label= name + " model", facecolors='none', edgecolors='red')
    cat_score = xx[0:-1:mke, 0]*strain_scale
    yyy = (yy[0:-1:mke, 0] - yy_pred[0:-1:mke,0]) * stress_scale/th
    numpy.savetxt(Test_or_Train+'blah_xx.txt',numpy.transpose(numpy.array([cat_score,yyy])))
#    plt.scatter(xx[0:-1:mke, 0]*strain_scale, (yy[0:-1:mke, 0] - yy_pred[0:-1:mke,0]) * stress_scale/th, label= name + " model", facecolors='none', edgecolors='red')

    plt.xlabel(r"$E_{0}^{(11)}$", size=s, labelpad=21)
    #plt.ylabel(r"$S_{0}^{(11)}$", size=s, labelpad=21)
    plt.ylabel(r"$S_{0}^{(11)} - \hat{S_{0}^{(11)}}$", size=s, labelpad=21)
    plt.tick_params(axis='both', labelsize=ls)
    plt.tick_params(axis='both', labelsize=ls)
    ax.yaxis.get_offset_text().set_fontsize(ls)
    #plt.legend(prop={'size': ls})
    plt.tight_layout()
    plt.savefig(Test_or_Train + "sigma_xx_" + name + ".png")
    plt.close("all")


    plt.figure(figsize=(12, 10))
    ax = plt.axes()
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.scatter(xx[0:-1:mke, 1]*strain_scale, yy[0:-1:mke, 1] * stress_scale/th, label="Reference", facecolors='none', edgecolors='black')
    plt.scatter(xx[0:-1:mke, 1]*strain_scale, yy_pred[0:-1:mke, 1] * stress_scale/th, label= name + " model", facecolors='none', edgecolors='red')
    cat_score = xx[0:-1:mke, 0]*strain_scale
    yyy = (yy[0:-1:mke, 1] - yy_pred[0:-1:mke,1]) * stress_scale/th
    numpy.savetxt(Test_or_Train+'blah_yy.txt',numpy.transpose(numpy.array([cat_score,yyy])))
#    plt.scatter(xx[0:-1:mke, 1]*strain_scale, (yy[0:-1:mke, 1]  - yy_pred[0:-1:mke,1])* stress_scale/th, label= name + " model", facecolors='none', edgecolors='red')
    plt.xlabel(r"$E_{0}^{(22)}$", size=s, labelpad=21)
    plt.ylabel(r"$S_{0}^{(22)}$", size=s, labelpad=21)
    plt.tick_params(axis='both', labelsize=ls)
    plt.tick_params(axis='both', labelsize=ls)
    ax.yaxis.get_offset_text().set_fontsize(ls)
    #plt.legend(prop={'size': ls})
    plt.tight_layout()
    plt.savefig(Test_or_Train + "sigma_yy_" + name + ".png")
    plt.close("all")

    plt.figure(figsize=(12, 10))
    ax = plt.axes()
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.scatter(xx[0:-1:mke, 2]*strain_scale, yy[0:-1:mke, 2] * stress_scale/th, label="Reference", facecolors='none', edgecolors='black')
    plt.scatter(xx[0:-1:mke, 2]*strain_scale, yy_pred[0:-1:mke, 2] * stress_scale/th, label= name + " model", facecolors='none', edgecolors='red')
    cat_score = xx[0:-1:mke, 0]*strain_scale
    yyy = (yy[0:-1:mke, 2] - yy_pred[0:-1:mke,2]) * stress_scale/th
    numpy.savetxt(Test_or_Train+'blah_xy.txt',numpy.transpose(numpy.array([cat_score,yyy])))
    #plt.scatter(xx[0:-1:mke, 2]*strain_scale, (yy[0:-1:mke,2] - yy_pred[0:-1:mke, 2]) * stress_scale/th, label="Reference", facecolors='none', edgecolors='black')
    plt.xlabel(r"$2E_{0}^{(12)}$", size=s, labelpad=21)
    plt.ylabel(r"$S_{0}^{(12)}$", size=s, labelpad=21)
    plt.tick_params(axis='both', labelsize=ls)
    plt.tick_params(axis='both', labelsize=ls)
    ax.yaxis.get_offset_text().set_fontsize(ls)
    plt.legend(prop={'size': ls})
    plt.tight_layout()
    plt.savefig(Test_or_Train + "sigma_xy_" + name + ".png")
    plt.close("all")


def ReadData(stress_scale = 1.0e3, strain_scale=1.0e-1, SYM = False, DIR = "Training_data/", segment = None, segmentFile = None, cat_idx=None):

    dummy = numpy.loadtxt(DIR + "macro.strainxx.1", usecols = (0,1))
    dim = 3
    if segment is not None:
      if segmentFile is not None:
        print("cannot segment with both the tuple-based segmentation and a segmentfile")
        exit(-1)
      #segment is a tuple of 
      xx = numpy.loadtxt(DIR + "macro.strainxx.1", usecols=(1))
      yy = numpy.loadtxt(DIR + "macro.strainyy.1", usecols=(1))
      xy = numpy.loadtxt(DIR + "macro.strainxy.1", usecols=(1))
      mask = numpy.logical_and(xx > segment[0][0], xx < segment[0][1])
      mask = numpy.logical_and(mask, numpy.logical_and(yy > segment[1][0], yy < segment[1][1]))
      mask = numpy.logical_and(mask, numpy.logical_and(xy > segment[2][0], xy < segment[2][1]))
      ntp = numpy.sum(mask)
    elif segmentFile is not None:
      if cat_idx is None:
        print("need to supply cat_idx with segmentFile")
        exit(-1)
      #print('segmenting')
      xx = numpy.loadtxt(DIR + "macro.strainxx.1", usecols=(1))
      yy = numpy.loadtxt(DIR + "macro.strainyy.1", usecols=(1))
      xy = numpy.loadtxt(DIR + "macro.strainxy.1", usecols=(1))
      theta0 = numpy.genfromtxt("{0:s}0".format(segmentFile))
      theta1 = numpy.genfromtxt("{0:s}1".format(segmentFile))
      if len(theta0.shape) == 1:
        theta0 = theta0.reshape((1,-1)) #add an extra dimension to theta0
      n_categories = theta0.shape[0]
      n_categories += 1
      cat_score = numpy.zeros(n_categories)
      nn = len(xx)
      mask = numpy.ones(nn,dtype=bool)
      if cat_idx > -1:
        for i in range(nn):
          cat_score[1:] = theta0.dot(numpy.array([xx[i], yy[i], xy[i]])) + theta1
          mask[i] = numpy.argmax(cat_score) == cat_idx
      ntp = numpy.sum(mask)
    else:
      ntp = dummy.shape[0]


    print("(ntp, dim) = ", ntp, " , ", dim)


    sym_n = 0
    if SYM:
        sym_n = 3
    xx = numpy.zeros((ntp * (sym_n + 1), dim))
    yy = numpy.zeros((ntp * (sym_n + 1), dim))

    xx_ori = numpy.zeros((ntp , dim))
    yy_ori = numpy.zeros((ntp , dim))

    if segment is not None or segmentFile is not None:
      xx_ori[:, 0] = numpy.loadtxt(DIR + "macro.strainxx.1", usecols=(1))[mask] / strain_scale
      xx_ori[:, 1] = numpy.loadtxt(DIR + "macro.strainyy.1", usecols=(1))[mask] / strain_scale
      xx_ori[:, 2] = numpy.loadtxt(DIR + "macro.strainxy.1", usecols=(1))[mask] / strain_scale

      yy_ori[:, 0] = numpy.loadtxt(DIR + "macro.stressxx.1", usecols=(1))[mask] / stress_scale
      yy_ori[:, 1] = numpy.loadtxt(DIR + "macro.stressyy.1", usecols=(1))[mask] / stress_scale
      yy_ori[:, 2] = numpy.loadtxt(DIR + "macro.stressxy.1", usecols=(1))[mask] / stress_scale
    else:

      xx_ori[:, 0] = numpy.loadtxt(DIR + "macro.strainxx.1", usecols=(1)) / strain_scale
      xx_ori[:, 1] = numpy.loadtxt(DIR + "macro.strainyy.1", usecols=(1)) / strain_scale
      xx_ori[:, 2] = numpy.loadtxt(DIR + "macro.strainxy.1", usecols=(1)) / strain_scale

      yy_ori[:, 0] = numpy.loadtxt(DIR + "macro.stressxx.1", usecols=(1)) / stress_scale
      yy_ori[:, 1] = numpy.loadtxt(DIR + "macro.stressyy.1", usecols=(1)) / stress_scale
      yy_ori[:, 2] = numpy.loadtxt(DIR + "macro.stressxy.1", usecols=(1)) / stress_scale



    if False:
        print("Strain_Ranges are ", xx_ori[:,0].min(), " ", xx_ori[:,1].min(), " ", xx_ori[:,2].min(), " ",
                                  xx_ori[:,0].max(), " ", xx_ori[:,1].max(), " ", xx_ori[:,2].max())

        print("Abs_Strain_Ranges are ", numpy.fabs(xx_ori[:, 0]).min(), " ", numpy.fabs(xx_ori[:, 1]).min(), " ", numpy.fabs(xx_ori[:, 2]).min(), " ",
              numpy.fabs(xx_ori[:, 0]).max(), " ", numpy.fabs(xx_ori[:, 1]).max(), " ", numpy.fabs(xx_ori[:, 2]).max())

        print("Stress_Ranges are ", yy_ori[:, 0].min(), " ", yy_ori[:, 1].min(), " ", yy_ori[:, 2].min(), " ",
              yy_ori[:, 0].max(), " ", yy_ori[:, 1].max(), " ", yy_ori[:, 2].max())

        print("Abs_Stress_Ranges are ", numpy.fabs(yy_ori[:, 0]).min(), " ", numpy.fabs(yy_ori[:, 1]).min(), " ", numpy.fabs(yy_ori[:, 2]).min(), " ",
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
    #MyScatter(xx, yy, yy_pred,"Linear", stress_scale, strain_scale)

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
        self.batchnorm = torch.nn.BatchNorm1d(3).double()
        self.fc1 = torch.nn.Linear(3, 2).double()
        #True/False : with/without bias
        self.fc_out = torch.nn.Linear(2, 1, True).double()
        self.softrelu = torch.nn.Softplus()

    def forward(self, x_in):
        #x = self.batchnorm(x_in)
        x = x_in
        x = self.softrelu(self.fc1(x))
        #x = torch.relu(self.fc1(x))
        x = self.fc_out(x)
        return x

class Net_Map2(torch.nn.Module):
    # constitutive relation
    # strain -> H0*strain + NN(stress)
   
    def __init__(self,n_categories=2,ninputs=3):
        super(Net_Map2, self).__init__()
        self.noutput = 1
        self.n_categories = n_categories
        self.ninput = ninputs 
        self.classifier = torch.nn.Linear(self.ninput, n_categories*3).double()
        self.linear = torch.nn.Linear(self.ninput, n_categories*3, True).double()
        #True/False : with/without bias

    def forward(self, x_in):
        x = x_in
        """
        print("blah")
        softmax = torch.softmax(self.classifier(x),dim=1)
        print((softmax*self.linear(x)).shape)
        print((torch.sum(torch.softmax(self.classifier(x),dim=1)*self.linear(x),dim=1)).shape)
        """
        n = x.shape[0]
        phi = torch.softmax(self.classifier(x).view(n,self.n_categories),dim=2)
        theta = self.linear(x).view(n,3,self.n_categories)
        return torch.sum(phi*theta,dim=2)

    def eval(self, x_in):
        x = x_in
        n = x.shape[0]
        if len(x.shape) == 1:
          n = 1
        phi = torch.softmax(self.classifier(x).view(n,3,self.n_categories),dim=2)
        #print(phi)
        theta = self.linear(x).view(n,3,self.n_categories)
        #return torch.sum(phi*theta,dim=2)
        return phi


class Net_Map3(torch.nn.Module):
    # constitutive relation
    # strain -> H0*strain + NN(stress)
   
    def __init__(self,n_categories=2,ninputs=3):
        super(Net_Map3, self).__init__()
        self.n_categories = n_categories
        self.ninput = ninputs 
        if n_categories > 1:
          self.classifier = torch.nn.Linear(self.ninput, n_categories).double()
        else:
          self.classifier = None
        #self.linear = torch.nn.Linear(self.ninput, n_categories, True).double()
        self.fc1 = torch.nn.Linear(self.ninput, 2, True).double()
        #True/False : with/without bias
        self.fc_out = torch.nn.Linear(2, n_categories, True).double()
        self.fc_out2 = torch.nn.Linear(self.ninput, n_categories, True).double()
        #True/False : with/without bias
        self.softrelu = torch.nn.Softplus()

    def forward(self, x_in):
        x = x_in
        """
        print("blah")
        softmax = torch.softmax(self.classifier(x),dim=1)
        print((softmax*self.linear(x)).shape)
        print((torch.sum(torch.softmax(self.classifier(x),dim=1)*self.linear(x),dim=1)).shape)
        """
        n = x.shape[0]
        #theta = self.fc_out(self.softrelu(self.fc1(x)))
        #theta = self.softrelu(self.fc1(x))
        #theta = self.fc_out(theta)
        theta = self.fc_out2(x)
        if self.n_categories > 1:
          phi = torch.softmax(self.classifier(x),dim=1)
          return torch.sum(phi*theta,dim=1).view(n, 1)
        else:
          return theta
    def classify_loss(self,x_in):
       return torch.sum(torch.prod(torch.softmax(self.classifier(x_in),dim=1),1)) 

    def print_linear_params(self):
        parameterlist = []
        for i, parameter in enumerate(self.classifier.parameters()):
          parameter_np = parameter.data.numpy()
          parameterlist.append(parameter_np)
          print(parameter_np[1:] - parameter_np[0])
          numpy.savetxt('param{0:d}'.format(i), parameter_np[1:] - parameter_np[0])
        return parameterlist

# Linear/Quadratic fit with plot
# Scale strain or stress to similar magnitude
def FitH(segmentData=None):
    xx, yy = ReadData(stress_scale, strain_scale, True, "Training_data/",segment=segmentData)
    xx_test, yy_test = ReadData(stress_scale, strain_scale, True, "Test_data/",segment=segmentData)

    H, h = LinearReg(xx, yy, "Orth", stress_scale, strain_scale)
    yy_test_pred = numpy.dot(xx_test, H)
    res_test = yy_test_pred - yy_test
    print("LinearReg-Test-Error ||res||_fro/||yy||_fro = ",
          numpy.linalg.norm(res_test, 'fro') / numpy.linalg.norm(yy_test, 'fro')," ",
                                                            numpy.linalg.norm(res_test[:,0:1], 'fro')/numpy.linalg.norm(yy_test[:,0:1], 'fro')," ",
                                                            numpy.linalg.norm(res_test[:,1:2], 'fro')/numpy.linalg.norm(yy_test[:,1:2], 'fro')," ",
                                                            numpy.linalg.norm(res_test[:,2:3], 'fro')/numpy.linalg.norm(yy_test[:,2:3], 'fro'))
    MyScatter(xx_test, yy_test, yy_test_pred, "Linear", stress_scale, strain_scale, "Test")
    return

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



def AdditiveNN_Train(segmentData=None,segFile=None,cat_idx=None,linearFit=True):

       

    def transform_inputs(xx):
      ntp,dim = xx.shape
      dim_transform = 3
      xx_transform = numpy.zeros((ntp,dim_transform))
      xx_transform[:,:3] = xx
      #xx_transform[:,3] = xx[:,2]*xx[:,2] # xy*xy
      """
      xx_transform[:,4] = xx[:,0]*xx[:,0] # xx*xx
      xx_transform[:,5] = xx[:,1]*xx[:,1] # yy*yy
      xx_transform[:,6] = xx[:,0]*xx[:,1] # xx*yy
      xx_transform[:,7] = xx[:,0]*xx[:,2] # xy*xx
      xx_transform[:,8] = xx[:,2]*xx[:,1] # xy*yy
      """
      return xx_transform,dim_transform


    #Read data
    print("train")
    xx, yy = ReadData(stress_scale, strain_scale, True, "Training_data/",segment=segmentData,segmentFile=segFile,cat_idx=cat_idx)
    print("test")
    xx_test, yy_test = ReadData(stress_scale, strain_scale, True, "Test_data/",segment=segmentData,segmentFile=segFile,cat_idx=cat_idx)
    ntp, dim = xx.shape
    ntp_test, dim_test = xx_test.shape
    xx_transform, dimT = transform_inputs(xx)

    if segFile is None:
      n_cats = N_CATS
    else:
      n_cats = 1
    name = "Net_Map3"
    model = Net_Map3(n_categories=n_cats,ninputs=dimT)
    #name = "Net_Map"
    #model = Net_Map()

    inputs = torch.from_numpy(xx_transform).view(ntp, dimT)
    
    if(linearFit):
      #Linear fit to get H, h
      H, h = LinearReg(xx, yy, "Orth", stress_scale, strain_scale)

      yy_linear_fit = numpy.dot(xx, H)
      yy_test_linear_fit = numpy.dot(xx_test, H)
      
      output_np = (yy - yy_linear_fit)
    else:
      output_np = yy[:,iidx]
    outputs = torch.from_numpy(output_np).view(ntp, 1)
    

    optimizer = optim.LBFGS(model.parameters(), lr=0.8, max_iter=1000, line_search_fn='strong_wolfe')
    #optimizer = optim.SGD(model.parameters(), lr=0.0005)

    # L2 regularization
    factor = torch.tensor(reg_factor*ntp)

    Nite = 5
    for i in range(Nite):
        print("Iteration : ", i)
        def closure(printFlag=False):
            optimizer.zero_grad()
            sigma = model(inputs)
            l2_loss = torch.tensor(0.)
            for param in model.parameters():#
                l2_loss += param.norm()
            loss1 = (torch.sum((sigma - outputs) ** 2)) * stress_scale * stress_scale
            #loss1 = (torch.sum((sigma - outputs) ** 2 * torch.Tensor([1.0,1.0,1.0e3]))) * stress_scale * stress_scale
            loss2 = (factor * l2_loss) * stress_scale * stress_scale
            loss = loss1 + loss2
            #if segFile is None:
              #loss -= 1e12 * model.classify_loss(inputs)
           

            #loss.backward(retain_graph=True)
            loss.retain_grad()
            loss.backward(retain_graph=True)
            if printFlag:
              print("loss {0:e}, loss1 {1:e}, loss2 = {2:e} ".format( loss.item(), loss1.item(), loss2.item()))
              gradnorm = 0
              for param in model.parameters():
                gradnorm += param.grad.norm()
              print("gradnorm: {0:e} ".format(gradnorm))
            return loss
        #closure(printFlag=True)
        optimizer.step(closure)


    xx_transform, dimT = transform_inputs(xx)
    yy_pred_norm = model(torch.from_numpy(xx_transform).view(ntp, dimT))
    yy_pred = yy_pred_norm.data.numpy() 
    if linearFit:
      res_train = yy - yy_linear_fit - yy_pred
    else:
      res_train = numpy.copy(yy) 
      res_train[:,iidx] = res_train[:,iidx] - yy_pred[:,0]
    print("Train fro error =", numpy.linalg.norm(res_train, ord='fro') * stress_scale * stress_scale)
    print("Train fro relative error = ",
          numpy.linalg.norm(res_train, ord='fro') / numpy.linalg.norm(yy, ord = 'fro'), " ", 
          numpy.linalg.norm(res_train[:,0:1], ord='fro') / numpy.linalg.norm(yy[:,0:1], ord = 'fro'), " ", 
          numpy.linalg.norm(res_train[:,1:2], ord='fro') / numpy.linalg.norm(yy[:,1:2], ord = 'fro'), " ",
          numpy.linalg.norm(res_train[:,2:3], ord='fro') / numpy.linalg.norm(yy[:,2:3], ord = 'fro'))
    print("Sum terms train fro error = ",
          numpy.linalg.norm(res_train[:,0:1], ord='fro')**2, " ", numpy.linalg.norm(yy[:,0:1], ord='fro')**2, " ",
          numpy.linalg.norm(res_train[:,1:2], ord='fro')**2, " ", numpy.linalg.norm(yy[:,1:2], ord='fro')**2, " ",
          numpy.linalg.norm(res_train[:,2:3], ord='fro')**2, " ", numpy.linalg.norm(yy[:,2:3], ord='fro')**2)
    if segFile is None and N_CATS > 1:
      model.print_linear_params()

    
    name = "NN-ReLU"
    if linearFit:
      yy_pred = yy_linear_fit + yy_pred
    #MyScatter(xx, yy, yy_pred, name, stress_scale, strain_scale)


    ############ Test
    xx_transform, dimT = transform_inputs(xx_test)
    t0 = time.perf_counter()
    yy_test_pred_norm = model(torch.from_numpy(xx_transform).view(ntp_test, dimT))
    t1 = time.perf_counter()
    """
    print("----------------------------------")
    print("Time spent predicting: ", t1 - t0)
    print("----------------------------------")
    """
    yy_test_pred = yy_test_pred_norm.data.numpy() 
    if linearFit:
      res_test = yy_test - yy_test_linear_fit - yy_test_pred
    else:
      res_test = numpy.copy(yy_test)
      res_test[:,iidx] = res_test[:,iidx] - yy_test_pred[:,0]
    print("Test fro error =",
          numpy.linalg.norm(res_test, ord='fro') * stress_scale * stress_scale)
    print("Test fro relative error = ",
          numpy.linalg.norm(res_test, ord='fro') / numpy.linalg.norm(yy_test, ord='fro'), " ",
          numpy.linalg.norm(res_test[:,0:1], ord='fro') / numpy.linalg.norm(yy_test[:,0:1], ord='fro'), " ",
          numpy.linalg.norm(res_test[:,1:2], ord='fro') / numpy.linalg.norm(yy_test[:,1:2], ord='fro'), " ",
          numpy.linalg.norm(res_test[:,2:3], ord='fro') / numpy.linalg.norm(yy_test[:,2:3], ord='fro'))
    print("Sum terms fro error = ",
          numpy.linalg.norm(res_test[:,0:1], ord='fro')**2, " ", numpy.linalg.norm(yy_test[:,0:1], ord='fro')**2, " ",
          numpy.linalg.norm(res_test[:,1:2], ord='fro')**2, " ", numpy.linalg.norm(yy_test[:,1:2], ord='fro')**2, " ",
          numpy.linalg.norm(res_test[:,2:3], ord='fro')**2, " ", numpy.linalg.norm(yy_test[:,2:3], ord='fro')**2)

    if linearFit:
      yy_test_pred =  yy_test_linear_fit + yy_test_pred
    #MyScatter(xx_test, yy_test, yy_test_pred, name, stress_scale, strain_scale, "Test")


    ###########  Save to cpp file
    example = torch.rand([1, dimT]).double()
    traced_script_module = torch.jit.trace(model, example)
    output = traced_script_module(torch.ones([1, dimT]).double())
    traced_script_module.save("model" + name + "Additive.pt")







if __name__ == "__main__":

    #FitH()
    #FitH(segmentData=([0.0, 1e10], [0.0, 1e10], [-1e20, 1e20]))
    """
    print("-----------------------xx tension, yy tension-----------------------")
    AdditiveNN_Train(segmentData=([0.0, 1e10], [0.0, 1e10], [-1e20, 1e20]))
    print("-----------------------xx tension, yy compression-----------------------")
    AdditiveNN_Train(segmentData=([0.0, 1e10], [-1e10,0.0], [-1e20, 1e20]))
    print("-----------------------xx compression, yy tension-----------------------")
    AdditiveNN_Train(segmentData=([-1e10, 0.0], [0.0,1e20], [-1e20, 1e20]))
    print("-----------------------xx compression, yy compression-----------------------")
    AdditiveNN_Train(segmentData=([-1e10, 0.0], [0.0, 1e20], [-1e20, 1e20]))
    """
    #AdditiveNN_Train(segmentData=([-1e20, 1e20], [-1e20, 1e20], [-0.05, 0.05]))
    if classify:
      AdditiveNN_Train(linearFit=False)
    else:
      print("-----------------------no classifying-----------------------")
      AdditiveNN_Train(segFile="param",cat_idx=-1, linearFit=False)
      for i in range(N_CATS):
        print("-----------------------classification {0:d}-----------------------".format(i))
        AdditiveNN_Train(segFile="param",cat_idx=i, linearFit=False)

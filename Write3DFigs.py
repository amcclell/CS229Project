from TrainNN import *
import subprocess
from timeit import default_timer as timer


rc('text', usetex=True)


def plotBin(x, y, z, c, edges, titleS, xLabel, yLabel, zLabel, fPath, i, show=False):
    # s, ls = 35, 33

    fig = plt.figure(figsize=(12, 10))
    ax = fig.gca(projection='3d')

    # ax.plot_trisurf(x, y, z)
    scat = ax.scatter(x * strain_scale, y * strain_scale, z * stress_scale / th, c=(c * strain_scale))
    cbar = fig.colorbar(scat, shrink=0.5, aspect=5)
    ax.set_xlabel(r"$E^{(%s)}$" % xLabel)  # , size=s)
    if yLabel == "xy":
        ax.set_ylabel(r"$2E^{(%s)}$" % yLabel)  # , size=s)
    else:
        ax.set_ylabel(r"$E^{(%s)}$" % yLabel)  # , size=s)
    if titleS == "xy":
        ax.set_title(r"$%.4e \leq 2E^{(12)} \leq %.4e$" % (strain_scale * edges[0], strain_scale * edges[1]))  # , size=s)
        cbar.set_label(r"$2E^{(12)}$")
    else:
        ax.set_title(r"$%.4e \leq E^{(%s)} \leq %.4e$" % (strain_scale * edges[0], titleS, strain_scale * edges[1]))  # , size=s)
        cbar.set_label(r"$E^{(%s)}$" % titleS)
    ax.set_zlabel(r"$S^{(%s)}$" % zLabel[2:])  # , size=s)
    # ax.tick_params(axis="both", labelsize=ls)

    fig.savefig(fPath + "_%d.png" % i)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plotBins(XX, YY, edges, niu, fPath, eLabels, slabels, start):
    Inds = np.arange(3)
    Inds = Inds[Inds != niu]

    figS = "/fig"
    fSxx = fPath + eLabels[niu] + slabels[0] + figS
    fSyy = fPath + eLabels[niu] + slabels[1] + figS
    fSxy = fPath + eLabels[niu] + slabels[2] + figS

    for i in range(edges.shape[0] - 1):
        inds = (XX[:, niu] >= edges[i]) & (XX[:, niu] <= edges[i + 1])
        plotE = XX[inds, :]
        plotS = YY[inds, :]

        plotBin(plotE[:, Inds[0]], plotE[:, Inds[1]], plotS[:, 0], plotE[:, niu], edges[i:i+2], eLabels[niu], eLabels[Inds[0]], eLabels[Inds[1]], slabels[0], fSxx, i)
        plotBin(plotE[:, Inds[0]], plotE[:, Inds[1]], plotS[:, 1], plotE[:, niu], edges[i:i + 2], eLabels[niu], eLabels[Inds[0]], eLabels[Inds[1]], slabels[1], fSyy, i)
        plotBin(plotE[:, Inds[0]], plotE[:, Inds[1]], plotS[:, 2], plotE[:, niu], edges[i:i + 2], eLabels[niu], eLabels[Inds[0]], eLabels[Inds[1]], slabels[2], fSxy, i)

        end = timer()
        print('Completed {}-th bin for E{}    Elapsed Time: {}'.format(i, eLabels[niu], end - start))


def plotBinFlipped(xTup, xFTup, edges, titleS, xLabel, yLabel, zLabel, fPath, i, flip="Symmetrized", show=False):
    s, ls, lp, lpz = 25, 18, 9, 14

    fig = plt.figure(figsize=(12, 9))
    ax = fig.gca(projection='3d')

    x = xTup[0]
    y = xTup[1]
    z = xTup[2]
    c = xTup[3]

    xF = xFTup[0]
    yF = xFTup[1]
    zF = xFTup[2]
    cF = xFTup[3]

    scat = ax.scatter(x * strain_scale, y * strain_scale, z * stress_scale / th, c=(c * strain_scale), cmap=plt.cm.autumn, vmin=edges[0] * strain_scale, vmax=edges[1] * strain_scale)
    scatF = ax.scatter(xF * strain_scale, yF * strain_scale, zF * stress_scale / th, c=(cF * strain_scale), cmap=plt.cm.winter, vmin=edges[0] * strain_scale, vmax=edges[1] * strain_scale)
    ticks = np.linspace(edges[0], edges[1], 5) * strain_scale
    cbar = fig.colorbar(scat, shrink=0.5, orientation='vertical', ticks=ticks, pad=0.1)
    cbarF = fig.colorbar(scatF, shrink=0.5, orientation='vertical', ticks=ticks)
    ax.set_xlabel(r"$E^{(%s)}$" % xLabel, size=s, labelpad=lp)
    if yLabel == "12":
        ax.set_ylabel(r"$2E^{(%s)}$" % yLabel, size=s, labelpad=lp)
    else:
        ax.set_ylabel(r"$E^{(%s)}$" % yLabel, size=s, labelpad=lp)
    if titleS == "12":
        ax.set_title(r"$%.4e \leq 2E^{(12)} \leq %.4e$" % (strain_scale * edges[0], strain_scale * edges[1]), size=s)
        cbar.set_label(r"$2E^{(12)}$ Original", fontsize=s)
        cbar.ax.tick_params(labelsize=ls)
        cbarF.set_label(r"$2E^{(12)}$ %s" % flip, fontsize=s)
        cbarF.ax.tick_params(labelsize=ls)
    else:
        ax.set_title(r"$%.4e \leq E^{(%s)} \leq %.4e$" % (strain_scale * edges[0], titleS, strain_scale * edges[1]), size=s)
        cbar.set_label(r"$E^{(%s)}$ Original" % titleS, fontsize=s)
        cbar.ax.tick_params(labelsize=ls)
        cbarF.set_label(r"$E^{(%s)}$ %s" % (titleS, flip), fontsize=s)
        cbarF.ax.tick_params(labelsize=ls)
    ax.set_zlabel(r"$S^{(%s)}$" % zLabel[2:], size=s, labelpad=lpz)
    ax.ticklabel_format(axis='z', style='sci', scilimits=(0, 0))
    ax.zaxis.get_offset_text().set_size(ls)
    ax.tick_params(axis="both", labelsize=ls)

    fig.savefig(fPath + "_%d.png" % i)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plotBinsFlipped(XX, YY, niu, fPath, eLabels, slabels, start):
    Inds = np.arange(3)
    Inds = Inds[Inds != niu]

    figS = "/flipped/fig_flipped"
    fSxx = fPath + eLabels[niu] + slabels[0] + figS
    fSyy = fPath + eLabels[niu] + slabels[1] + figS
    fSxy = fPath + eLabels[niu] + slabels[2] + figS

    nP, dim = XX.shape

    XXF = np.empty((nP * 3, dim), dtype=np.float64)
    XXF[0:nP, 0] = XX[:, 0]
    XXF[0:nP, 1] = XX[:, 1]
    XXF[0:nP, 2] = -XX[:, 2]
    XXF[nP:2 * nP, 0] = XX[:, 1]
    XXF[nP:2 * nP, 1] = XX[:, 0]
    XXF[nP:2 * nP, 2] = XX[:, 2]
    XXF[2 * nP:3 * nP, 0] = XX[:, 1]
    XXF[2 * nP:3 * nP, 1] = XX[:, 0]
    XXF[2 * nP:3 * nP, 2] = -XX[:, 2]

    YYF = np.empty((nP * 3, dim), dtype=np.float64)
    YYF[0:nP, 0] = YY[:, 0]
    YYF[0:nP, 1] = YY[:, 1]
    YYF[0:nP, 2] = -YY[:, 2]
    YYF[nP:2 * nP, 0] = YY[:, 1]
    YYF[nP:2 * nP, 1] = YY[:, 0]
    YYF[nP:2 * nP, 2] = YY[:, 2]
    YYF[2 * nP:3 * nP, 0] = YY[:, 1]
    YYF[2 * nP:3 * nP, 1] = YY[:, 0]
    YYF[2 * nP:3 * nP, 2] = -YY[:, 2]

    XXFu = np.unique(XXF[:, niu])
    edges = np.histogram(XXFu, 30)[1]

    show = False

    for i in range(edges.shape[0] - 1):
        inds = (XX[:, niu] >= edges[i]) & (XX[:, niu] <= edges[i + 1])
        plotE = XX[inds, :]
        plotS = YY[inds, :]

        indsF = (XXF[:, niu] >= edges[i]) & (XXF[:, niu] <= edges[i + 1])
        plotEF = XXF[indsF, :]
        plotSF = YYF[indsF, :]

        xTup = (plotE[:, Inds[0]], plotE[:, Inds[1]], plotS[:, 0], plotE[:, niu])
        xFTup = (plotEF[:, Inds[0]], plotEF[:, Inds[1]], plotSF[:, 0], plotEF[:, niu])
        plotBinFlipped(xTup, xFTup, edges[i:i + 2], eLabels[niu], eLabels[Inds[0]], eLabels[Inds[1]], slabels[0], fSxx, i)

        xTup = (plotE[:, Inds[0]], plotE[:, Inds[1]], plotS[:, 1], plotE[:, niu])
        xFTup = (plotEF[:, Inds[0]], plotEF[:, Inds[1]], plotSF[:, 1], plotEF[:, niu])
        plotBinFlipped(xTup, xFTup, edges[i:i + 2], eLabels[niu], eLabels[Inds[0]], eLabels[Inds[1]], slabels[1], fSyy, i)

        xTup = (plotE[:, Inds[0]], plotE[:, Inds[1]], plotS[:, 2], plotE[:, niu])
        xFTup = (plotEF[:, Inds[0]], plotEF[:, Inds[1]], plotSF[:, 2], plotEF[:, niu])
        plotBinFlipped(xTup, xFTup, edges[i:i + 2], eLabels[niu], eLabels[Inds[0]], eLabels[Inds[1]], slabels[2], fSxy, i, show=show)

        end = timer()
        print('Completed {}-th bin for E{}    Elapsed Time: {}'.format(i, eLabels[niu], end - start))


def plotBinsPartialFlip(XX, YY, niu, fPath, eLabels, slabels, start):
    Inds = np.arange(3)
    Inds = Inds[Inds != niu]

    figS = "/partialFlip/fig_partialFlip"
    fSxx = fPath + eLabels[niu] + slabels[0] + figS
    fSyy = fPath + eLabels[niu] + slabels[1] + figS
    fSxy = fPath + eLabels[niu] + slabels[2] + figS

    inds = (XX[:, 2] > np.fabs(XX[:, 2].min()))
    XXF = np.vstack((XX[inds, 0], XX[inds, 1], -XX[inds, 2])).T
    tmp = np.vstack((XX, XXF))

    YYF = np.vstack((YY[inds, 0], YY[inds, 1], -YY[inds, 2])).T

    XXFu = np.unique(tmp[:, niu])
    edges = np.histogram(XXFu, 30)[1]

    show = False
    flip = "Selectively Symmetrized"

    for i in range(edges.shape[0] - 1):
        inds = (XX[:, niu] >= edges[i]) & (XX[:, niu] <= edges[i + 1])
        plotE = XX[inds, :]
        plotS = YY[inds, :]

        indsF = (XXF[:, niu] >= edges[i]) & (XXF[:, niu] <= edges[i + 1])
        plotEF = XXF[indsF, :]
        plotSF = YYF[indsF, :]

        xTup = (plotE[:, Inds[0]], plotE[:, Inds[1]], plotS[:, 0], plotE[:, niu])
        xFTup = (plotEF[:, Inds[0]], plotEF[:, Inds[1]], plotSF[:, 0], plotEF[:, niu])
        plotBinFlipped(xTup, xFTup, edges[i:i + 2], eLabels[niu], eLabels[Inds[0]], eLabels[Inds[1]], slabels[0], fSxx, i, flip)

        xTup = (plotE[:, Inds[0]], plotE[:, Inds[1]], plotS[:, 1], plotE[:, niu])
        xFTup = (plotEF[:, Inds[0]], plotEF[:, Inds[1]], plotSF[:, 1], plotEF[:, niu])
        plotBinFlipped(xTup, xFTup, edges[i:i + 2], eLabels[niu], eLabels[Inds[0]], eLabels[Inds[1]], slabels[1], fSyy, i, flip)

        xTup = (plotE[:, Inds[0]], plotE[:, Inds[1]], plotS[:, 2], plotE[:, niu])
        xFTup = (plotEF[:, Inds[0]], plotEF[:, Inds[1]], plotSF[:, 2], plotEF[:, niu])
        plotBinFlipped(xTup, xFTup, edges[i:i + 2], eLabels[niu], eLabels[Inds[0]], eLabels[Inds[1]], slabels[2], fSxy, i, flip, show)

        end = timer()
        print('Completed {}-th bin for E{}    Elapsed Time: {}'.format(i, eLabels[niu], end - start))


def main():
    xx, yy = ReadData(stress_scale, strain_scale, False, "Training_Data/")
    xx0u = np.unique(xx[:, 0])
    xx1u = np.unique(xx[:, 1])
    xx2u = np.unique(xx[:, 2])

    figPath = "figs3D/VaryE"
    flipped = "/flipped"
    partFlip = "/partialFlip"
    xxS = ["11", "22", "12"]
    SxxS = ["/S11", "/S22", "/S12"]

    cmdOut = subprocess.run(["mkdir", "-p", figPath + xxS[0] + SxxS[0] + flipped])
    cmdOut = subprocess.run(["mkdir", "-p", figPath + xxS[0] + SxxS[1] + flipped])
    cmdOut = subprocess.run(["mkdir", "-p", figPath + xxS[0] + SxxS[2] + flipped])

    cmdOut = subprocess.run(["mkdir", "-p", figPath + xxS[1] + SxxS[0] + flipped])
    cmdOut = subprocess.run(["mkdir", "-p", figPath + xxS[1] + SxxS[1] + flipped])
    cmdOut = subprocess.run(["mkdir", "-p", figPath + xxS[1] + SxxS[2] + flipped])

    cmdOut = subprocess.run(["mkdir", "-p", figPath + xxS[2] + SxxS[0] + flipped])
    cmdOut = subprocess.run(["mkdir", "-p", figPath + xxS[2] + SxxS[1] + flipped])
    cmdOut = subprocess.run(["mkdir", "-p", figPath + xxS[2] + SxxS[2] + flipped])

    cmdOut = subprocess.run(["mkdir", "-p", figPath + xxS[0] + SxxS[0] + partFlip])
    cmdOut = subprocess.run(["mkdir", "-p", figPath + xxS[0] + SxxS[1] + partFlip])
    cmdOut = subprocess.run(["mkdir", "-p", figPath + xxS[0] + SxxS[2] + partFlip])

    cmdOut = subprocess.run(["mkdir", "-p", figPath + xxS[1] + SxxS[0] + partFlip])
    cmdOut = subprocess.run(["mkdir", "-p", figPath + xxS[1] + SxxS[1] + partFlip])
    cmdOut = subprocess.run(["mkdir", "-p", figPath + xxS[1] + SxxS[2] + partFlip])

    cmdOut = subprocess.run(["mkdir", "-p", figPath + xxS[2] + SxxS[0] + partFlip])
    cmdOut = subprocess.run(["mkdir", "-p", figPath + xxS[2] + SxxS[1] + partFlip])
    cmdOut = subprocess.run(["mkdir", "-p", figPath + xxS[2] + SxxS[2] + partFlip])

    start = timer()

    edges0 = np.histogram(xx0u, 30)[1]
    edges1 = np.histogram(xx1u, 30)[1]
    edges2 = np.histogram(xx2u, 30)[1]

    plotBins(xx, yy, edges0, 0, figPath, xxS, SxxS, start)
    plotBins(xx, yy, edges1, 1, figPath, xxS, SxxS, start)
    plotBins(xx, yy, edges2, 2, figPath, xxS, SxxS, start)

    plotBinsFlipped(xx, yy, 0, figPath, xxS, SxxS, start)
    plotBinsFlipped(xx, yy, 1, figPath, xxS, SxxS, start)
    plotBinsFlipped(xx, yy, 2, figPath, xxS, SxxS, start)

    plotBinsPartialFlip(xx, yy, 0, figPath, xxS, SxxS, start)
    plotBinsPartialFlip(xx, yy, 1, figPath, xxS, SxxS, start)
    plotBinsPartialFlip(xx, yy, 2, figPath, xxS, SxxS, start)

    H, model, so, muo = AdditiveNNModified_Train()

    s, ls = 15, 12

    xx, yy = ReadData(stress_scale, strain_scale, True, "Training_Data/")
    ntp, dim = xx.shape
    inds = (np.abs(xx[:, 2] * strain_scale + 0.12172) <= 1e-2) & (xx[:, 2] * strain_scale >= -0.12172)
    inds = inds & (np.abs(xx[:, 1] - xx[inds, 1].min()) <= 1e-4)

    yyfit = np.dot(xx, H) + (so * model(torch.from_numpy(xx).view(ntp, dim)) + muo).data.numpy()

    fig = plt.figure()
    scat = plt.scatter(xx[inds, 0] * strain_scale, yy[inds, 2] * stress_scale / th, c=xx[inds, 2] * strain_scale, vmin=xx[inds, 2].min() * strain_scale, vmax=xx[inds, 2].max() * strain_scale, cmap=plt.cm.winter)
    plot = plt.plot(xx[inds, 0] * strain_scale, yyfit[inds, 2] * stress_scale / th, 'r.')
    ax = fig.gca()
    ax.tick_params(axis='both', labelsize=ls)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_size(ls)
    ax.set_xlabel(r"$E^{(11)}$", size=s)
    ax.set_ylabel(r"$S^{(12)}$", size=s)
    ticks = np.linspace(xx[inds, 2].min(), xx[inds, 2].max(), 5) * strain_scale
    cbar = plt.colorbar(scat, ticks=ticks)
    cbar.set_label(r"$2E^{(12)}$", size=s)
    cbar.ax.tick_params(labelsize=ls)
    ax.set_title(r"$E^{(22)} = %.4f$" % (np.mean(xx[inds, 1]) * strain_scale), size=s)
    fig.savefig("E22_E12_Slice_Sym.png")
    plt.show()


if __name__ == "__main__":
    main()

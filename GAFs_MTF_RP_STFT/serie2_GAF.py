# -*- coding: utf-8 -*-
"""
@author: Stephen Wang
@modified: masterqkk, masterqkk@outlook.com
Environment:
    python: 3.6
    Pandas: 1.0.3
    matplotlib: 3.2.1
    pyts: 0.11.0
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle, sys
from pyts.datasets import load_gunpoint
from pyts.image import GramianAngularField


def window_time_series(series, n, step=1):
    # Define sliding window
    #    print "in window_time_series",series
    if step < 1.0:
        step = max(int(step * n), 1)
    return [series[i:i + n] for i in range(0, len(series) - n + 1, step)]


def paa(series, now, opw):
    '''
    PAA function
    :param series: 
    :param now: the number of seg.
    :param opw: the length of each seg.
    :return: 
    '''
    if now == None:
        now = int(len(series) / opw)
    if opw == None:
        opw = int(len(series) / now)
    return [sum(series[i * opw: (i + 1) * opw]) / float(opw) for i in range(now)]


def standardize(serie):
    dev = np.sqrt(np.var(serie))
    mean = np.mean(serie)
    return [(each - mean) / dev for each in serie]


def rescale(serie):
    # Rescale data into [0,1]
    maxval = max(serie)
    minval = min(serie)
    gap = float(maxval - minval)
    return [(each - minval) / gap for each in serie]


def rescaleminus(serie):
    # Rescale data into [-1,1]
    maxval = max(serie)
    minval = min(serie)
    gap = float(maxval - minval)
    return [(each - minval) / gap * 2 - 1 for each in serie]


def QMeq(series, Q):
    # Generate quantile bins
    q = pd.qcut(list(set(series)), Q)
    dic = dict(zip(set(series), q.labels))
    MSM = np.zeros([Q, Q])
    label = []
    for each in series:
        label.append(dic[each])
    for i in range(0, len(label) - 1):
        MSM[label[i]][label[i + 1]] += 1
    for i in range(Q):
        if sum(MSM[i][:]) == 0:
            continue
        MSM[i][:] = MSM[i][:] / sum(MSM[i][:])
    return np.array(MSM), label, q.levels


def QVeq(series, Q):
    # Generate quantile bins when equal values exist in the array (slower than QMeq)
    q = pd.qcut(list(set(series)), Q)
    dic = dict(zip(set(series), q.labels))
    qv = np.zeros([1, Q])
    label = []
    for each in series:
        label.append(dic[each])
    for i in range(0, len(label)):
        qv[0][label[i]] += 1.0
    return np.array(qv[0][:] / sum(qv[0][:])), label



def paaMarkovMatrix(paalist, level):
    # Generate Markov Matrix given a spesicif number of quantile bins
    paaindex = []
    for each in paalist:
        for k in range(len(level)):
            lower = float(level[k][1:-1].split(',')[0])
            upper = float(level[k][1:-1].split(',')[-1])
            if lower <= each <= upper:
                paaindex.append(k)
    return paaindex


def gengrampdfs(image, label, name):
    # Generate pdf files of generated images
    import matplotlib.backends.backend_pdf as bpdf
    import operator
    index = zip(range(len(label)), label)
    index.sort(key=operator.itemgetter(1))
    with bpdf.PdfPages(name) as pdf:
        count = 0
        for p, q in index:
            count += 1
            print('generate fig of pdfs: {}'.format(p))
            plt.ioff()
            fig = plt.figure()
            plt.suptitle(datafile + '_' + str(label[p]))
            ax1 = plt.subplot(121)
            plt.imshow(image[p])
            divider = make_axes_locatable(ax1)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            plt.colorbar(cax=cax)
            ax2 = plt.subplot(122)
            plt.imshow(paaimage[p])
            divider = make_axes_locatable(ax2)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            plt.colorbar(cax=cax)
            pdf.savefig(fig)
            plt.close(fig)
            if count > 30:
                break
    pdf.close


# Generate pdf files of trainsisted array in porlar coordinates
def genpolarpdfs(raw, label, name):
    import matplotlib.backends.backend_pdf as bpdf
    import operator
    index = zip(range(len(label)), label)
    index.sort(key=operator.itemgetter(1))
    with bpdf.PdfPages(name) as pdf:
        for p, q in index:
            print('generate fig of pdfs: {}'.format(p))
            plt.ioff()
            r = np.array(range(1, length + 1))
            r = r / 100.0
            theta = np.arccos(np.array(rescaleminus(standardize(raw[p][1:])))) * 2
            fig = plt.figure()
            plt.suptitle(datafile + '_' + str(label[p]))
            ax = plt.subplot(111, polar=True);
            ax.plot(theta, r, color='r', linewidth=3)
            pdf.savefig(fig)
            plt.close(fig)
    pdf.close


def maxsample(mat, s):
    # return the max value instead of mean value in PAAs
    retval = []
    x, y, z = mat.shape
    l = np.int(np.floor(y / float(s)))
    for each in mat:
        block = []
        for i in range(s):
            block.append([np.max(each[i * l:(i + 1) * l, j * l:(j + 1) * l]) for j in range(s)])
        retval.append(np.asarray(block))
    return np.asarray(retval)


def pickledata(mat, label, train, name):
    # Pickle the data (train, test) and save in the pkl file
    print('..pickling data:'.format(name))
    traintp = (mat[:train], label[:train])
    testtp = (mat[train:], label[train:])
    with open(name + '.pkl', 'wb') as f:
        pickletp = [traintp, testtp]
        pickle.dump(pickletp, f, protocol=pickle.HIGHEST_PROTOCOL)


def pickle3data(mat, label, train, name):
    # Pickle the data (train, validation, test) and save in the pkl file
    print('..pickling data:'.format(name))
    traintp = (mat[:train], label[:train])
    validtp = (mat[:train], label[:train])
    testtp = (mat[train:], label[train:])
    with open(name + '.pkl', 'wb') as f:
        pickletp = [traintp, validtp, testtp]
        pickle.dump(pickletp, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    datafiles = ['Coffee_ALL']  # Data fine name

    num_trains = [28]  # Number of training instances (because we assume training and test data are mixed in one file)
    size = [64]  # PAA size
    GAF_type = 'GADF'  # GAF type: GASF, GADF
    save_PAA = True  # Save the GAF with or without dimension reduction by PAA (Piecewise Aggregation Approximation): True, False
    rescale_type = 'Zero'  # Rescale the data into [0,1] or [-1,1]: Zero, Minusone
    length = 0

    for datafile, num_train in zip(datafiles, num_trains):
        fn = datafile
        for s in size:
            print('read file: {}, size: {}, GAF type: {}, rescale_type: {}'.format(datafile, s, GAF_type, rescale_type))
            raw = open(fn).readlines()
            raw = [list(map(float, each.strip().split())) for each in raw]
            length = len(raw[0]) - 1

            print('format data')
            label = []
            image = []
            paaimage = []
            patchimage = []
            matmatrix = []
            fullmatrix = []
            for each in raw:
                label.append(each[0])
                if rescale_type == 'Zero':
                    std_data = rescale(each[1:])
                elif rescale_type == 'Minusone':
                    std_data = rescaleminus(each[1:])
                else:
                    sys.exit('Unknown rescaling type!')

                paalistcos = paa(std_data, s, None)
                # paalistcos = rescale(paa(each[1:],s,None))
                # paalistcos = rescaleminus(paa(each[1:],s,None))

                ################raw###################
                datacos = np.array(std_data)
                datasin = np.sqrt(1 - np.array(std_data) ** 2)

                paalistcos = np.array(paalistcos)
                paalistsin = np.sqrt(1 - paalistcos ** 2)

                datacos = np.matrix(datacos)
                datasin = np.matrix(datasin)

                paalistcos = np.matrix(paalistcos)
                paalistsin = np.matrix(paalistsin)
                if GAF_type == 'GASF':
                    paamatrix = paalistcos.T * paalistcos - paalistsin.T * paalistsin
                    matrix = np.array(datacos.T * datacos - datasin.T * datasin)
                elif GAF_type == 'GADF':
                    paamatrix = paalistsin.T * paalistcos - paalistcos.T * paalistsin
                    matrix = np.array(datasin.T * datacos - datacos.T * datasin)
                else:
                    sys.exit('Unknown GAF type!')
                paamatrix = np.array(paamatrix)

                image.append(matrix)
                paaimage.append(np.array(paamatrix))
                matmatrix.append(paamatrix.flatten())
                fullmatrix.append(matrix.flatten())
            #
            label = np.asarray(label)
            image = np.asarray(image)
            paaimage = np.asarray(paaimage)
            patchimage = np.asarray(patchimage)
            matmatrix = np.asarray(matmatrix)
            fullmatrix = np.asarray(fullmatrix)

            datafilename = datafile + '_PAA_' + str(s) + '_' + GAF_type
            if not save_PAA:
                finalmatrix = matmatrix
            else:
                finalmatrix = fullmatrix

            pickledata(finalmatrix, label, num_train, datafilename)
    ## rescaled time series and polar coordinates
    k = 0
    print(length)
    r = np.array(range(1, length + 1))
    r = r / 100.0
    # theta = np.array(rescale(raw[k][1:])) * 2 * np.pi  # radian -> Angle
    theta = np.arccos(np.array(rescale(raw[k][1:]))) * 2 * np.pi # radian -> Angle
    plt.figure()
    ax1 = plt.subplot(121)
    ax1.plot(np.arange(len(rescale(raw[k][1:]))), rescale(raw[k][1:]))
    plt.title('rescaled time series')
    ax2 = plt.subplot(122, polar=True)
    ax2.plot(theta, r, color='r', linewidth=3)
    plt.title('polar system')
    plt.show()

    ## draw large image (without PAA) and paa image
    k = 0
    plt.figure()
    plt.suptitle(datafile + '_index_' + str(k) + '_label_' + str(label[k]))
    ax1 = plt.subplot(121)
    plt.title(GAF_type + 'without PAA')
    plt.imshow(image[k])
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.2) # Create an axes at the given *position*=right with the same height (or width) of the main axes
    plt.colorbar(cax=cax)

    ax2 = plt.subplot(122)
    plt.title(GAF_type + 'with PAA')
    plt.imshow(paaimage[k])
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    plt.colorbar(cax=cax)
    plt.show()
    # plt.figure();plt.suptitle(datafile+'_'+str(label[k])+'_'+str(Q));ax1 = plt.subplot(131);plt.imshow(image[k]);divider = make_axes_locatable(ax1);cax = divider.append_axes("right", size="5%", pad=0.2);plt.colorbar(cax = cax);ax2 = plt.subplot(132);plt.imshow(paaimage[k]);divider = make_axes_locatable(ax2);cax = divider.append_axes("right", size="5%", pad=0.2);plt.colorbar(cax = cax);ax3 = plt.subplot(133);plt.imshow(patchimage[k]);divider = make_axes_locatable(ax3);cax = divider.append_axes("right", size="5%", pad=0.2);plt.colorbar(cax = cax);


    ### call API, 以 gunpoint 数据为例
    X, _, _, _ = load_gunpoint(return_X_y=True)
    gasf = GramianAngularField(method='summation')
    X_gasf = gasf.transform(X)
    gadf = GramianAngularField(method='difference')
    X_gadf = gadf.transform(X)

    plt.figure()
    plt.suptitle('gunpoint_index_' + str(0))
    ax1 = plt.subplot(121)
    ax1.plot(np.arange(len(rescale(X[k][:]))), rescale(X[k][:]))
    plt.title('rescaled time series')
    ax2 = plt.subplot(122, polar=True)
    r = np.array(range(1, len(X[k]) + 1)) / 150
    # theta = np.array(rescale(raw[k][1:])) * 2 * np.pi  # radian -> Angle
    theta = np.arccos(np.array(rescale(X[k][:]))) * 2 * np.pi  # radian -> Angle

    ax2.plot(theta, r, color='r', linewidth=3)
    plt.title('polar system')
    plt.show()

    plt.figure()
    plt.suptitle('gunpoint_index_' + str(0))
    ax1 = plt.subplot(121)
    plt.imshow(X_gasf[k])
    plt.title('GASF')
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.2) # Create an axes at the given *position*=right with the same height (or width) of the main axes
    plt.colorbar(cax=cax)

    ax2 = plt.subplot(122)
    plt.imshow(X_gadf[k])
    plt.title('GASF')
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%",
                              pad=0.2)  # Create an axes at the given *position*=right with the same height (or width) of the main axes
    plt.colorbar(cax=cax)
    plt.show()
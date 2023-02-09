import numba as nb
import numpy as np
import json


def getAllParms():
    weightsDict = {}
    shapeDict = {"c1.conv": [6, 1, 5, 5],
                 "c3.conv": [16, 6, 5, 5],
                 "c5.conv": [120, 16, 5, 5],
                 "f6.fc": [84, 120],
                 "output.fc": [10, 84]}
    ArrList = []
    for key in shapeDict:
        Arr = np.loadtxt('./weights/'+key+".weight.csv",
                         delimiter=',').astype(int)
        shape = shapeDict[key]
        Arr = Arr.reshape(([i for i in shape]))
        ArrList.append(Arr)
        weightsDict[key] = Arr
    weightsDict["outputBias"] = np.loadtxt(
        './weights/'+key+".bias.csv", delimiter=',').reshape(([1, 10])).astype(int)

    scalesDict = {}
    with open('scale_hw.json') as json_file:
        scalesDict = json.load(json_file)

    return weightsDict, scalesDict


@nb.jit()
def MaxPool2d(x, kernel_size=2, stride=2):
    # TODO
    N, C, H, W = x.shape
    x_out = np.zeros((N, C, int(((H-kernel_size)/stride)+1),
                     int((W-kernel_size)/stride + 1)), dtype='int32')

    P = int((H-kernel_size)/stride+1)
    Q = int((W-kernel_size)/stride+1)
    for n in range(N):
        for c in range(C):
            for p in range(P):
                for q in range(Q):
                    # find max
                    max = x[n,c,p*stride,q*stride]
                    for a in range(kernel_size):
                        for b in range(kernel_size):
                            if x[n,c,p*stride+a,q*stride+b] > max:
                                max = x[n,c,p*stride+a,q*stride+b]
                    x_out[n,c,p,q] = max
    
    return x_out


@nb.jit()
def ReLU(x):
    # TODO
    x = np.where(x > 0, x, 0)

    return x


@nb.jit()
def Linear(psum_range, x, weights, weightsBias=None, psum_record=False):
    # TODO
    psum_record_list = [np.complex64(x) for x in range(0)]
    H, W = x.shape

    C = weights.shape[0]
    x_out = np.zeros((H, C), dtype='int32')
    
    for h in range(H):
        for c in range(C):
            if weightsBias != None:
                x_out[h,c] += weightsBias[0,c]
                # record
                if psum_record:
                    psum_record_list.append(x_out[h,c])

                # clamp
                if x_out[h,c] < psum_range[0]:
                    x_out[h,c] = psum_range[0]
                elif x_out[h,c] > psum_range[1]:
                    x_out[h,c] = psum_range[1]

            for w in range(W):
                x_out[h,c] += x[h,w] * weights[c,w]
                
                # record
                if psum_record:
                    psum_record_list.append(x_out[h,c])
                
                # clamp
                if x_out[h,c] < psum_range[0]:
                    x_out[h,c] = psum_range[0]
                elif x_out[h,c] > psum_range[1]:
                    x_out[h,c] = psum_range[1]
                

    return x_out, psum_record_list


@nb.jit()
def Conv2d(psum_range, x, weights, out_channels, kernel_size=5, stride=1, bias=False, psum_record=False):
    # TODO
    psum_record_list = [np.complex64(x) for x in range(0)]
    N, C, H, W = x.shape
    x_out = np.zeros((N, out_channels, int(((H-kernel_size)/stride)+1),
                     int((W-kernel_size)/stride + 1)), dtype='int32')

    M = out_channels
    P = int((H-kernel_size)/stride+1)
    Q = int((W-kernel_size)/stride+1)
    _, _, R, S = weights.shape

    for n in range(N):
        for m in range(M):
            for p in range(P):
                for q in range(Q):
                    for r in range(R):
                        for s in range(S):
                            for c in range(C):
                                h = p*stride+r
                                w = q*stride+s
                                x_out[n,m,p,q] += x[n,c,h,w] * weights[m,c,r,s]
                                
                                # record
                                if psum_record:
                                    psum_record_list.append(x_out[n,m,p,q])
                                
                                # clamp
                                if x_out[n,m,p,q] < psum_range[0]:
                                    x_out[n,m,p,q] = psum_range[0]
                                elif x_out[n,m,p,q] > psum_range[1]:
                                    x_out[n,m,p,q] = psum_range[1]
                                

    return x_out, psum_record_list


def ActQuant(x, scale, shiftbits=16):
    x = np.clip(
        np.floor(x*scale).astype('int') >> shiftbits, -128, 127)
    return x

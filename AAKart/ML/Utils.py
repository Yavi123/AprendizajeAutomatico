from skl2onnx import to_onnx
from onnx2json import convert
import pickle
import json
import numpy as np


def ExportONNX_JSON_TO_Custom(onnx_json,mlp):
    graphDic = onnx_json["graph"]
    initializer = graphDic["initializer"]
    s= "num_layers:"+str(mlp.n_layers_)+"\n"
    index = 0
    parameterIndex = 0;
    for parameter in initializer:
        s += "parameter:"+str(parameterIndex)+"\n"
        print(parameter["dims"])
        s += "dims:"+str(parameter["dims"])+"\n"
        print(parameter["name"])
        s += "name:"+str(parameter["name"])+"\n"
        print(parameter["doubleData"])
        s += "values:"+str(parameter["doubleData"])+"\n"
        index = index + 1
        parameterIndex = index // 2
    return s

def ExportAllformatsMLPSKlearn(mlp,X,picklefileName,onixFileName,jsonFileName,customFileName):
    with open(picklefileName,'wb') as f:
        pickle.dump(mlp,f)
    
    onx = to_onnx(mlp, X[:1])
    with open(onixFileName, "wb") as f:
        f.write(onx.SerializeToString())
    
    onnx_json = convert(input_onnx_file_path=onixFileName,output_json_path=jsonFileName,json_indent=2)
    
    customFormat = ExportONNX_JSON_TO_Custom(onnx_json,mlp)
    with open(customFileName, 'w') as f:
        f.write(customFormat)

def CreateTxt(thetas, file):
    with open(file, 'w') as f:
        num_layers = len(thetas)
        f.write(f"num_layers:{num_layers + 1}\n")

        parameter_num = 0
        for layer_num, theta in enumerate(thetas):

            for param_type, param_values in [('coefficient', theta[:, 1:]), ('intercepts', theta[:, 0])]:
                dims = list(map(str, param_values.shape))
                f.write(f"parameter:{parameter_num}\n")
                f.write(f"dims:{dims}\n")
                f.write(f"name:{param_type}\n")
                f.write(f"values:{param_values.flatten().tolist()}\n")
            parameter_num += 1



def ExportThetasToFile(thetaList, file):
    s= "num_layers:"+str(len(thetaList))+"\n"
    
    parameterIndex = 0
    for i in range(len(thetaList)):
        s += "parameter:"+str(i)+"\n"
        
        s += "dims:"+str(list(map(str, thetaList[i].shape)))+"\n"
        
        s += "name:coefficient"+"\n"
        
        s += "values:"+str(thetaList[i].flatten())+"\n"
        
        parameterIndex += 1

    with open(file, 'w') as f:
        f.write(s)

import torch
import model_zoo.LaneNet as LaneNet
from Configs import config as cfg
import onnx

def toOnnx():
    model = LaneNet.LaneNet()

    try:
        model.load_state_dict(torch.load(cfg.training_cfg.savepath + 'tempmodel.pkl'))
    except:
        temp = torch.load(cfg.training_cfg.savepath + 'tempmodel.pkl', map_location="cuda:0")
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in temp.items():
            name = k[7:]
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)


    print("load...")
    with torch.no_grad():
        model.postbranch.layers7.weight.data.copy_(model.postbranch.weights.data)

    model.cuda()
    model.eval()
    ##1. set the input name
    randinput=torch.randn(1,3,256,512,device='cuda:0')
    model.forward(randinput)
    inputname=['input']
    outputname=['ins','seg','cen']
    torch.onnx.export(model,randinput,'model.onnx',opset_version=10,verbose=True,input_names=inputname,output_names=outputname)



def runOnnx():
    model=onnx.load("model.onnx")
    onnx.checker.check_model(model)
    onnx.helper.printable_graph(model.graph)

if __name__ == '__main__':
    toOnnx()
    runOnnx()
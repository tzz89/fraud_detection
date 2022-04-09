import argparse
import torch
from fraud_model import FraudModel
import onnx
import onnxruntime
import numpy as np

parser = argparse.ArgumentParser(description="convert pytorch model to onnx model")
parser.add_argument('--model_weights', type=str, default="models/Best_NN_model.pth")
parser.add_argument('--n_features', type=int, default=30)
parser.add_argument('--map_location', type=str, default="cpu")
parser.add_argument('--output_fp', type=str, default="models/Best_NN_model.onnx")

if __name__ == "__main__":
    args = parser.parse_args()

    model = FraudModel(args.n_features)
    model.load_state_dict(torch.load(args.model_weights, map_location=args.map_location))
    model.eval()

    x = torch.randn(1, args.n_features, requires_grad=True)
    x_numpy = x.detach().numpy()

    torch_output = model(x).detach().cpu().numpy()

    torch.onnx.export(model, x, args.output_fp, 
                      export_params=True, opset_version=10, 
                      do_constant_folding=True, input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input':{0 :'batchsize'}, 
                                    'output':{0: 'batchsize'}})

    # checking if the export is successful
    onnx_model = onnx.load(args.output_fp)
    onnx.checker.check_model(onnx_model)

    ort_session = onnxruntime.InferenceSession(args.output_fp)
    ort_inputs = {ort_session.get_inputs()[0].name: x_numpy}
    ort_outputs = ort_session.run(None, ort_inputs)

    np.testing.assert_allclose(torch_output, ort_outputs[0], rtol=1e-03, atol=1e-05)
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")


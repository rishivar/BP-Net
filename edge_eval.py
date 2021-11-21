import onnxruntime
import time


ort_session = onnxruntime.InferenceSession('onnx.onnx')

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# compute ONNX Runtime output prediction

ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
start = time.time()
ort_outs = ort_session.run(None, ort_inputs)
print(f'Time: {time.time() - start}')

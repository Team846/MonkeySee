from rknn.api import RKNN

rknn = RKNN(verbose=True)

print('configuring for RK3588')
rknn.config(
    mean_values=[[0, 0, 0]], 
    std_values=[[255, 255, 255]], 
    target_platform='rk3588'
)

print('Loading ONNX model')
ret = rknn.load_onnx(model='./fuel.onnx')
if ret != 0:
    exit(ret)

print('Building FP16 RKNN model')
ret = rknn.build(do_quantization=False)
if ret != 0:
    exit(ret)

print('Exporting FP16 RKNN model')
ret = rknn.export_rknn('./fuel_fp16_test.rknn')
if ret != 0:
    exit(ret)

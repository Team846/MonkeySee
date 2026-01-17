from rknn.api import RKNN
import numpy as np

def test_inference():
    rknn = None  # initialize safely

    in0 = np.random.rand(1, 3, 1280, 1280).astype(np.float32)

    try:
        rknn = RKNN()
        ret = rknn.load_rknn(
            '/home/orangepi/monkeyvisiongpd/GPD2026/best_rknn_model/best-rk3588.rknn')

        if ret != 0:
            print("Load model failed")
            return None

        ret = rknn.init_runtime(target='rk3588')
        
        if ret != 0:
            print("Initialize runtime failed")
            return None

        print("Running inference")
        outputs = rknn.inference(inputs=[in0])
        out0 = outputs[0]
        return out0

    except Exception as e:
        print(f'Error: {e}')
        return None

    finally:
        if rknn is not None:
            rknn.release()

if __name__ == "__main__":
    print(test_inference())
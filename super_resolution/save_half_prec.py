import torch
import argparse
from model import NAS


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='convert full precision to half')
    parser.add_argument('--quality', required=True, type=str, choices=('low', 'medium', 'high', 'ultra'), help='used for selecting device')

    opt = parser.parse_args()
    # quality="high"
    model = NAS.Multi_Network(opt.quality)
    model = model.cuda()
    model = model.half()

    basedir = f"./sr_training/checkpoint/LOL/{opt.quality}"

    num_chunks = 5

    for idx in range(1, num_chunks+1):
        model_path = f"{basedir}/DNN_chunk_{idx}.pth"
        print(model_path)
        weights = torch.load(model_path)
        # model.load_state_dict(weights, strict=False)
        convert = False
        # Check the data type of the model's parameters
        param_data_types = [param.dtype for param in weights.values()]
        if all(data_type == torch.float16 for data_type in param_data_types):
            print("The model is in half precision (FP16).")
        elif all(data_type == torch.float32 for data_type in param_data_types):
            print("The model is in full precision (FP32).")
            convert = True
        else:
            print("The model has mixed data types (not strictly half or full precision).")

        if not convert:
            continue
        
        print(f"Converting {model_path} to float16")
        # Convert the model's parameters to half precision
        for key, param in weights.items():
            weights[key] = param.to(torch.float16)
            
        # Save the model in half precision
        half_precision_model_path = f"{basedir}/DNN_chunk_{idx}_half.pth"
        torch.save(weights, half_precision_model_path)
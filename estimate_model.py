import os
import time

import torch
from torchvision import transforms
import numpy as np
from PIL import Image

from models import MSCAEfficientFormerV2


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    classes = 2  # include background

    weights_path = "/mnt/d/PythonCode/CAFormerSwinV2/CAFormerSwinV2-Pytorch/save_weights/MSCAEfficientFormerV2_best_model_0.0.pth"
    img_path = "/mnt/d/MedicalSeg/CVC-ClinicDB/Original/1.png"
    roi_mask_path = "/mnt/d/MedicalSeg/CVC-ClinicDB/Ground Truth/1.png"
    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(img_path), f"image {img_path} not found."
    assert os.path.exists(roi_mask_path), f"image {roi_mask_path} not found."

    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)

    # get devices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = MSCAEfficientFormerV2(img_size=224, num_classes=classes)

    # load weights
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model_state'])
    model.to(device)

    # load roi mask & resize it to 224 for matching prediction's size
    roi_img = Image.open(roi_mask_path).convert('L').resize((224, 224), resample=Image.BICUBIC)
    roi_img = np.array(roi_img)

    # load image
    original_img = Image.open(img_path).convert('RGB')

    # from pil image to tensor and normalize
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.CenterCrop(224),
                                         transforms.Normalize(mean=mean, std=std)])
    img = data_transform(original_img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    model.eval()  # 进入验证模式
    with torch.no_grad():
        # init model
        # img_height, img_width = img.shape[-2:]
        # init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        # model(init_img)

        # t_start = time_synchronized()
        img = img.to(device)
        output = model(img)

        # t_end = time_synchronized()
        # print("inference time: {}".format(t_end - t_start))

        prediction = output.argmax(1).squeeze(0)
        prediction = prediction.to("cpu").numpy().astype(np.uint8)
        # 将前景对应的像素值改成255(白色)
        prediction[prediction == 1] = 255
        # 将不敢兴趣的区域像素设置成0(黑色)
        prediction[roi_img == 0] = 0
        mask = Image.fromarray(prediction)
        mask.save("./test_result.png")


if __name__ == '__main__':
    main()
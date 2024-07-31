import torch



def demo(model, img, target, MASK_THRESHOLD, device):
    model.eval()
    with torch.no_grad():
        prediction = model([img.to(device)])
        prediction = {k: v.to("cpu") for k, v in prediction[0].items()}

    for i in range(len(prediction["masks"])):
        prediction["masks"][i] = prediction["masks"][i] > MASK_THRESHOLD

    return {
        "img": img,
        "target": target,
        "prediction": prediction
    }

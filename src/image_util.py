import os
import gc
import logging
import random
import numpy as np
import torch
import cv2
import PIL
import sklearn.metrics
from torchvision.transforms import functional as F
from PIL import Image
from tqdm import tqdm


LOGGER = logging.getLogger()
SIZE = 128
HEIGHT=137
WIDTH=236
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def resize_to_square_PIL(image, size):
    w, h = image.size
    ratio = size / max(h, w)
    resized_image = image.resize((int(w*ratio), int(h*ratio)), Image.BILINEAR)
    return resized_image


def pad_PIL(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def threshold_image(img):
    '''
    Helper function for thresholding the images
    '''
    gray = PIL.Image.fromarray(np.uint8(img), 'L')
    ret,th = cv2.threshold(np.array(gray),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return th


def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax


def crop_resize(img0, size=SIZE, pad=16):
    
    #crop a box around pixels large than the threshold 
    #some images contain line at the sides
    ymin,ymax,xmin,xmax = bbox(img0[5:-5,5:-5] > 80)
    
    #cropping may cut too much, so we need to add it back
    xmin = xmin - 13 if (xmin > 13) else 0
    ymin = ymin - 10 if (ymin > 10) else 0
    xmax = xmax + 13 if (xmax < WIDTH - 13) else WIDTH
    ymax = ymax + 10 if (ymax < HEIGHT - 10) else HEIGHT
    img = img0[ymin:ymax,xmin:xmax]
    
    #remove lo intensity pixels as noise
    img[img < 28] = 0
    lx, ly = xmax-xmin,ymax-ymin
    l = max(lx,ly) + pad
    
    #make sure that the aspect ratio is kept in rescaling
    img = np.pad(img, [((l-ly)//2,), ((l-lx)//2,)], mode='constant')
    return cv2.resize(img,(size,size))


def Resize(df,size=128):
    resized = {} 
    df = df.set_index('image_id')
    
    for i in tqdm(range(df.shape[0])): 
        image0 = 255 - df.loc[df.index[i]].values.reshape(137,236).astype(np.uint8)
        
        #normalize each image by its max val
        img = (image0*(255.0/image0.max())).astype(np.uint8)
        image = crop_resize(img)
        resized[df.index[i]] = image.reshape(-1)
    resized = pd.DataFrame(resized).T.reset_index()
    resized.columns = resized.columns.astype(str)
    resized.rename(columns={'index':'image_id'},inplace=True)
    return resized


def image_to_tensor(image, normalize=None):
    tensor = torch.from_numpy(np.moveaxis(image / (255. if image.dtype == np.uint8 else 1), -1, 0).astype(np.float32))
    if normalize is not None:
        return F.normalize(tensor, **normalize)
    return tensor


def train_one_epoch(model, train_loader, criterion, optimizer, device, steps_upd_logging=500, accumulation_steps=1,
                                 multi_loss=None):
    model.train()

    total_loss = 0.0
    for step, (input_dic, targets) in tqdm(enumerate(train_loader), total=len(train_loader)):
        for k in input_dic.keys():
            input_dic[k] = input_dic[k].to(device)
        targets = targets.to(device)
        optimizer.zero_grad()

        if input_dic["image"].shape[1] not in [1, 3]:
            input_dic["image"] = input_dic["image"].unsqueeze(1)

        logits = model(input_dic["image"])

        loss = criterion(logits, targets)
        loss.backward()

        optimizer.step()
        # if (step + 1) % accumulation_steps == 0:  # Wait for several backward steps
            # optimizer.step()  # Now we can do an optimizer step

        total_loss += loss.item()

        if (step + 1) % steps_upd_logging == 0:
            LOGGER.info('Train loss on step {} was {}'.format(step + 1, round(total_loss / (step + 1), 5)))


    return total_loss / (step + 1)


def validate(model, val_loader, criterion, device, multi_loss=None):
    model.eval()

    val_loss = 0.0
    true_ans_list = []
    preds_cat = []
    for step, (input_dic, targets) in tqdm(enumerate(val_loader), total=len(val_loader)):
        for k in input_dic.keys():
            input_dic[k] = input_dic[k].to(device)
        targets = targets.to(device)

        if input_dic["image"].shape[1] not in [1, 3]:
            input_dic["image"] = input_dic["image"].unsqueeze(1)

        with torch.no_grad():

            logits = model(input_dic["image"])

            loss = criterion(logits, targets)
        
            val_loss += loss.item()
        
            targets = targets.float().cpu().detach().numpy()
            # logits = logits.argmax(1).float().cpu().detach().numpy().astype("float32")
            logits = logits.float().cpu().detach().numpy().astype("float32")            

            true_ans_list.append(targets)
            preds_cat.append(logits)
            del input_dic
            del targets, logits
            gc.collect()

    all_true_ans = np.concatenate(true_ans_list, axis=0)
    all_preds = np.concatenate(preds_cat, axis=0)
    
    return all_preds, all_true_ans, val_loss / (step + 1)

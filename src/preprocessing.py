import numpy as np

def normalize_modality(img):
    mask = img > 0
    mean = img[mask].mean()
    std = img[mask].std()

    img_normalized = (img - mean) / (std + 1e-8)
    img_normalized[~mask] = 0

    return img_normalized


def normalize_image(image):
    normalized_channels = []
    for c in range(image.shape[0]):
        normalized_channels.append(normalize_modality(image[c]))

    return np.stack(normalized_channels, axis=0)


def remap_labels(label):
    label[label == 4] = 3
    return label

def crop_volume(image, label, crop_size=(128, 128, 128)):
    _, h, w, d = image.shape
    ch, cw, cd = crop_size

    # pick a random starting point, but stay within bounds
    h_start = np.random.randint(0, max(h - ch, 1))
    w_start = np.random.randint(0, max(w - cw, 1))
    d_start = np.random.randint(0, max(d - cd, 1))

    image_cropped = image[:, h_start:h_start+ch, w_start:w_start+cw, d_start:d_start+cd]
    label_cropped = label[h_start:h_start+ch, w_start:w_start+cw, d_start:d_start+cd]

    return image_cropped, label_cropped

def preprocess(image, label):
    image = normalize_image(image)
    label = remap_labels(label)
    image, label = crop_volume(image, label)
    return image, label
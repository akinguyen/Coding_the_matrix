from vec import Vec

def _read_byte(f):
    return int.from_bytes(f.read(1), byteorder='big', signed=False)

def _load_images(n_img=3100):
    images = []
    D = {(i,j) for i in range(28) for j in range(28)}
    with open('mnist-images.dat', 'rb') as f:
        for img_num in range(n_img):
            image = Vec(D, {})
            for i in range(28):
                for j in range(28):
                    image[i,j] = _read_byte(f)
            images.append(image)
    return images

def _load_labels(n_img=3100):
    labels = []
    with open('mnist-labels.dat', 'rb') as f:
        for img in range(n_img):
            labels.append(_read_byte(f))
    return labels

def load_data(n=3100):
    '''
    Returns:
        - list of Vecs, each of which is an image
        - list of corresponding labels
    '''
    return _load_images(n), _load_labels(n)

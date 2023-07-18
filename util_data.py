import numpy as np
from utils import util_dicom
def normalize_image(image, min_value=None, max_value=None, clip=True):
    # Convert the image to a NumPy array of float32 data type
    image = np.array(image, dtype=np.float32)

    # Determine the minimum and maximum values of the image
    if min_value is None:
        min_value = np.min(image)
    if max_value is None:
        max_value = np.max(image)

    # Clip the pixel values inside the min-max range if specified
    if clip:
        image = np.clip(image, min_value, max_value)

    # Normalize the pixel values to the range [0.0, 1.0] using the min-max scaling method
    image = (image - min_value) / (max_value - min_value)

    # Scale the pixel values to the range [0.0, 255.0]
    image *= 255.0

    return image

# Let's cut a small cube out of this scan containing this fake cancer

# Here is the helper function
def cutCube(X, center, shape, spacing, padd=0): # center is a 3d coord (zyx) -> center of the real/fake lesion
    center = center.astype(int)
    hlz = np.round(shape[0] / 2)
    hly = np.round(shape[1] / 2)
    hlx = np.round(shape[2] / 2)

    # add padding if out of bounds
    if ((center - np.array([hlz,hly,hlx])) < 0).any() or (
        (center + np.array([hlz,hly,hlx]) + 1) > np.array(X.shape)).any():  # if cropping is out of bounds, add padding
        if X.shape != shape:
             # Handle the case when center is (0, 0, 0)
            resized_scan, resize_factor = util_dicom.scale_scan(X, spacing, factor=0.5)  # Adjust the spacing and factor as per your requirement

            # Get the new shape of the cropped cube after scaling
            new_shape = util_dicom.get_scaled_shape(shape, spacing)  # Adjust the spacing as per your requirement
            
            # Calculate the new center based on the resized scan
            new_center = np.array([hlz, hly, hlx])

            # Perform the cropping on the resized scan
            cube = resized_scan[int(new_center[0] - hlz):int(new_center[0] + hlz),
                                int(new_center[1] - hly):int(new_center[1] + hly),
                                int(new_center[2] - hlx):int(new_center[2] + hlx)]
                        
        else:
            Xn = np.ones(np.array(X.shape) + shape * 2) * padd
            Xn[shape[0]:(shape[0] + X.shape[0]), shape[1]:(shape[1] + X.shape[1]), shape[2]:(shape[2] + X.shape[2])] = X        
            centern = center + shape
            cube = Xn[int(centern[0] - hlz):int(centern[0] - hlz + shape[0]),
                int(centern[1] - hly):int(centern[1] - hly + shape[1]),
                int(centern[2] - hlx):int(centern[2] - hlx + shape[2])]
        return np.copy(cube)
    else:
        cube = X[int(center[0] - hlz):int(center[0] - hlz + shape[0]), int(center[1] - hly):int(center[1] - hly + shape[1]),
               int(center[2] - hlx):int(center[2] - hlx + shape[2])]
        return np.copy(cube)
    

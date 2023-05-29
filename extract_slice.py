import sys
import numpy as np
import SimpleITK as sitk

def _vrnormalize(vec,epsilon):
    '''
    Normalize a vector.
    ref. Matlab vrnormalize.m    
    '''
    norm_vec = np.linalg.norm(vec)
    if norm_vec <= epsilon:
        vec_n = np.zeros(vec.size)
    else:
        vec_n = np.divide(vec,norm_vec)
    return vec_n

def vrrotvec(v1,v2,epsilon=1e-12):
    '''
    Calculate rotation between two vectors.
    ref. Matlab vrrotvec.m, vrrotvec2mat.m
    '''
    v1 = np.array(v1)
    v2 = np.array(v2)
    v1n = _vrnormalize(v1,epsilon)
    v2n = _vrnormalize(v2,epsilon)
    v1xv2 = _vrnormalize(np.cross(v1n,v2n),epsilon)
    ac = np.arccos(np.vdot(v1n,v2n))    
    # build the rotation matrix
    s = np.sin(ac)
    c = np.cos(ac)
    t = 1 - c
    n = _vrnormalize(v1xv2, epsilon)
    x = n[0]
    y = n[1]
    z = n[2]
    m = [ [t*x*x + c,    t*x*y - s*z,  t*x*z + s*y],
          [t*x*y + s*z,  t*y*y + c,    t*y*z - s*x],
          [t*x*z - s*y,  t*y*z + s*x,  t*z*z + c], ]
    return np.array(m)

#
# "Extract an oblique 2D slice from a 3D volume"
#
# https://itk.org/pipermail/insight-users/2007-May/022171.html   lol
#
def extract_slice(itk_image,slice_center,slice_normal,slice_spacing,slice_size,is_label,out_value=0):
    
    image_normal = list(itk_image.GetDirection())[6:]

    slice_direction = vrrotvec(image_normal,slice_normal)
    slice_direction = tuple(slice_direction.ravel())
    
    slice_size_mm = np.array(slice_size)*np.array(slice_spacing)
    
    # our normal is (a,b,c)
    # thus plane equation is ax+by+cz+d=0
    a,b,c = image_normal
    x,y,z = slice_center
    d = -a*x-b*y*c*z
    
    # we project origin of input image to plane to get 2nd point on plane
    alt_x,alt_y,_ = itk_image.GetOrigin()
    alt_z = (-a*alt_x-b*alt_x-d)/c
    alt_vec = np.array([alt_x,alt_y,alt_z])-np.array(slice_center)
    alt_vec = _vrnormalize(alt_vec,1e-12)
    print('alt_vec',alt_vec)
    slice_origin = slice_center
    slice_origin = slice_center-np.multiply(alt_vec,slice_size_mm/2)
    

    resample = sitk.ResampleImageFilter()
    resample.SetOutputOrigin(slice_origin)
    resample.SetOutputDirection(slice_direction)
    resample.SetOutputSpacing(slice_spacing)
    resample.SetSize(slice_size) # unit is voxel
    resample.SetDefaultPixelValue(out_value)
    print('!',resample.GetOutputOrigin())

    axis = slice_normal
    rotation_center = slice_center
    angle = 0
    translation = (0,0,0)
    scale_factor = 1
    similarity = sitk.Similarity3DTransform(
        scale_factor, axis, angle, translation, rotation_center
    )

    affine = sitk.AffineTransform(3)
    affine.SetMatrix(similarity.GetMatrix())
    affine.SetTranslation(similarity.GetTranslation())
    affine.SetCenter(similarity.GetCenter())

    resample.SetTransform(affine)
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkLinear)

    return resample.Execute(itk_image)

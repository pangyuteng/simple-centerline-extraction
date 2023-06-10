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


def get_slice_origin(slice_center,slice_normal,slice_radius):

    epsilon = 1e-12

    image_normal = np.array([0.,0.,1.])
    slice_direction = vrrotvec(image_normal,slice_normal).ravel()

    direction_x = np.array(slice_direction[0:3])
    direction_y = np.array(slice_direction[3:6])
    direction_z = np.array(slice_direction[6:9])

    #direction_x, direction_y = get_orthonormals(slice_normal)
    vec_on_plane = _vrnormalize(direction_x+direction_y,epsilon)

    # 45-45-90 triangle
    # side length ratio: 1:1:sqrt(2)
    # so the offset from center of square is...
    #
    offset = slice_radius*2/np.sqrt(2)
    slice_origin = slice_center - vec_on_plane*offset

    # slice_origin should be on the plane
    a,b,c = tuple(direction_z)
    x,y,z = tuple(slice_center)
    d = -a*x-b*y-c*z
    ox,oy,oz = slice_origin

    assert(a*ox+b*oy+c*oz+d <= 1e-4)

    return tuple(slice_origin)

#
# "Extract an oblique 2D slice from a 3D volume"
#
# https://itk.org/pipermail/insight-users/2007-May/022171.html   lol
#
def extract_slice(itk_image,slice_center,slice_normal,slice_spacing,slice_radius,is_label):
    slice_center = np.array(slice_center)

    image_normal = np.array([0.,0.,1.])
    rotation_matrix = vrrotvec(image_normal,slice_normal)

    slice_direction = rotation_matrix.ravel()

    direction_x = np.array(slice_direction[0:3])
    direction_y = np.array(slice_direction[3:6])
    direction_z = np.array(slice_direction[6:9])

    slice_direction = []
    slice_direction.extend(direction_x)
    slice_direction.extend(direction_y)
    slice_direction.extend(direction_z*-1) # what???
    slice_direction = np.array(slice_direction)

    slice_origin = get_slice_origin(slice_center,slice_normal,slice_radius)
    radius_voxel = int(np.array(slice_radius)/np.array(slice_spacing[0]))
    factor = 2
    slice_size = (radius_voxel*factor,radius_voxel*factor,1)
    resample = sitk.ResampleImageFilter()
    
    resample.SetOutputOrigin(slice_origin)
    resample.SetOutputDirection(slice_direction)
    resample.SetOutputSpacing(slice_spacing)
    resample.SetSize(slice_size) # unit is voxel
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(0)

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkLinear)

    itk_image = resample.Execute(itk_image)
    return itk_image
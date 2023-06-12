import numpy as np
from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage import distance_transform_edt
import unittest
import SimpleITK as sitk
import imageio
NUM = 10000

def mock_image():    
    const = 100
    theta = np.linspace(-4 * np.pi, 4 * np.pi, NUM)
    z = np.linspace(0, 10, NUM)
    r = z**2 + 3
    x = r * np.sin(theta) + const
    y = r * np.cos(theta) + const
    img = np.zeros([256,256,15])
    for i,j,k in zip(x,y,z):
        img[int(i),int(j),int(k)]=1
    img=binary_dilation(img,iterations=2)
    return img,x,y,z

class MyTest(unittest.TestCase):

    def test_centerline_extracted(self):        
        img,x,y,z = mock_image()
        start_point = np.array([x[0],y[0],z[0]]).astype(int)
        end_point = np.array([x[-1],y[-1],z[-1]]).astype(int)
        from extract_centerline import extract_centerline
        centerline_list = extract_centerline(img,start_point,end_point)
        actual_x, actual_y, actual_z = centerline_list
        # expected length
        self.assertEqual(len(actual_x),385)
        self.assertEqual(len(actual_y),385)
        self.assertEqual(len(actual_z),385)
        # z should be descending
        self.assertTrue(all(np.logical_or(np.gradient(actual_z)==0,np.gradient(actual_z)<0)))

    def test_extract_slice(self):


        from extract_centerline import extract_centerline, smooth_3d_array
        from extract_slice import extract_slice

        img,x,y,z = mock_image()

        start_point = np.array([x[0],y[0],z[0]]).astype(int)
        end_point = np.array([x[-1],y[-1],z[-1]]).astype(int)
        search_radius=(1,1,1)
        centerline_list = extract_centerline(img,start_point,end_point,search_radius=search_radius)
        c_x,c_y,c_z = centerline_list
        c_x,c_y,c_z = c_x[::20],c_y[::20],c_z[::20]
        smothing_factor = 100
        s_x,s_y,s_z = smooth_3d_array(c_x,c_y,c_z,s=smothing_factor)
        offset = 1
        
        bsfield = distance_transform_edt(img)
        img_obj = sitk.GetImageFromArray(bsfield)
        img_obj.SetOrigin((0,0,0))
        img_obj.SetDirection((1,0,0,0,1,0,0,0,1))
        img_obj.SetSpacing((1,1,1))
        tmp = sitk.GetArrayFromImage(img_obj)

        idx = 4
        slice_center = int(c_z[idx  ]),int(c_y[idx  ]),int(c_x[idx  ])
        before_point = int(s_z[idx-offset]),int(s_y[idx-offset]),int(s_x[idx-offset])
        after_point  = int(s_z[idx+offset]),int(s_y[idx+offset]),int(s_x[idx+offset])

        appx_radius = int(bsfield[c_x[idx],c_y[idx],c_z[idx]])


        tmp = sitk.GetArrayFromImage(img_obj)

        expected_value = 1000
        tmp[c_x[idx],c_y[idx],c_z[idx]]=expected_value
        tmp_obj = sitk.GetImageFromArray(tmp)
        tmp_obj.CopyInformation(img_obj)

        slice_center = tmp_obj.TransformContinuousIndexToPhysicalPoint(slice_center)
        before_point = tmp_obj.TransformContinuousIndexToPhysicalPoint(before_point)
        after_point = tmp_obj.TransformContinuousIndexToPhysicalPoint(after_point)

        slice_normal = np.array(after_point)-np.array(before_point) # not normalized
        slice_spacing = (1,1,1) #mm
        slice_radius = appx_radius
        is_label = False
        
        slice_obj = extract_slice(tmp_obj,slice_center,slice_normal,slice_spacing,slice_radius,is_label)    
        slice_arr = sitk.GetArrayFromImage(slice_obj)
        assert(slice_arr[0,appx_radius,appx_radius]-expected_value < 1)


















if __name__ == '__main__':
    unittest.main()
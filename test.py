import numpy as np
from scipy.ndimage.morphology import binary_dilation
import unittest
from scipy import interpolate
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

class TestCenterlineExtraction(unittest.TestCase):

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
        
if __name__ == '__main__':
    unittest.main()
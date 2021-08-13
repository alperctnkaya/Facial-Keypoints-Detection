class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['keypoints']
        return {'image': torch.tensor(image),
                'keypoints': torch.tensor(landmarks)}

      
class RandomHorizontalFlip(object):    
    def __init__(self, p = 0.5):
        self.p = p 
    
    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']
        
        if torch.rand(1) > self.p:
            return {'image': image, 'keypoints': keypoints}
        
        w,h = image.shape
        image = image[:,::-1] - np.zeros_like(image)
        keypoints[::2] = w - keypoints[::2]
        return {'image': image, 'keypoints': keypoints}
      
      
class RandomRotation(torchvision.transforms.RandomRotation):
    def __init__(self, degrees, p):
        super(RandomRotation, self).__init__(degrees)
        self.p = p
        
    def forward(self, sample):
        image, keypoints = sample['image'], sample['keypoints']
        
        if torch.rand(1) > self.p:
            return {'image': image, 'keypoints': keypoints}
        
        w,h = image.shape[:2]
        fill = self.fill
        
        if isinstance(image, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * F._get_image_num_channels(image)
            else:
                fill = [float(f) for f in fill]
        else:
            image = torch.tensor(image)
        
        image = image.reshape(-1,1,w,h)
        angle = self.get_params(self.degrees)
        
        centerPoint = (w/2, h/2)
        image = functional.rotate(image, angle, self.resample, self.expand, centerPoint, fill)
        image = image.reshape(w,h)
        
        keypointsR = []
        it = iter(keypoints)
        
        for pointX, pointY in zip(it,it):
            xR , yR = self.rotate([pointX, pointY], origin = centerPoint, degrees = -angle) 
            keypointsR.append(xR)
            keypointsR.append(yR)
        
        return {'image': image, 'keypoints': keypointsR}
    
    def rotate(self, p, origin=(0, 0), degrees=0):
        angle = np.deg2rad(degrees)
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle),  np.cos(angle)]])
        o = np.atleast_2d(origin)
        p = np.atleast_2d(p)
        return np.squeeze((R @ (p.T-o.T) + o.T).T)

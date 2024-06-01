#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import skimage.io as skio


# In[ ]:


# load in training data

nose_train_pts = []
nose_test_pts = []

for j in range(1, 7): # viewpoint index
    for i in range(1, 41):  # person index
        if i == 8 or i == 12 or i == 14 or i == 15 or i == 22 or i == 30 or i == 35:
            gender = 'f' # gender

        else:
            gender = 'm'
        
        root_dir = './imm_face_db/'

        # load all facial keypoints/landmarks
        file = open(root_dir + '{:02d}-{:d}{}.asf'.format(i,j,gender))
        points = file.readlines()[16:74]
        landmark = []

        for point in points:
            x,y = point.split('\t')[2:4]
            landmark.append([float(x), float(y)])

        # the nose keypoint
        nose_keypoint = np.array(landmark).astype('float32')[-6]
        
        if (i < 33):
            nose_train_pts.append(nose_keypoint)
        else:
            nose_test_pts.append(nose_keypoint)



# In[ ]:


len(nose_train_pts)


# In[ ]:


import cv2
import numpy as np
import PIL
from PIL import Image
import torchvision.transforms

train_imgs = []
test_imgs = []
for j in range(1, 7): # viewpoint index
    for i in range(1, 41):  # person index
        if i == 8 or i == 12 or i == 14 or i == 15 or i == 22 or i == 30 or i == 35:
            gender = 'f' # gender

        else:
            gender = 'm'
        
        root_dir = './imm_face_db/'

        # load all facial keypoints/landmarks
        path = root_dir + '{:02d}-{:d}{}.jpg'.format(i,j,gender)

        img = cv2.imread(str(path))

        # You may need to convert the color.
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        normal_img = (gray_img/255) - 0.5
        print(normal_img)
        
        resize_img = cv2.resize(normal_img, (80, 60), interpolation = cv2.INTER_LINEAR)
#         im_pil = Image.fromarray(resize_img)
        
        if (i < 33):
            train_imgs.append(resize_img)
        else:
            test_imgs.append(resize_img)


# In[ ]:


train_imgs[1]


# In[ ]:


import matplotlib.pyplot as plt


plt.figure(figsize=(3,4))
plt.imshow(train_imgs[0],  cmap = "gray")


# In[ ]:


plt.figure(figsize=(3,4))
implot = plt.imshow(train_imgs[2],  cmap = "gray")

plt.scatter(nose_train_pts[2][0]*80, nose_train_pts[2][1]*60, c = 'r', s= 10)


plt.show()


# In[ ]:


import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=12, kernel_size=5) # 60 x 80 (H x W)
        self.conv2 = nn.Conv2d(12, 16, 5) 
        self.conv3 = nn.Conv2d(16, 32, 5)
#         self.conv4 = nn.Conv2d(64, 96, 5)
#         self.conv4 = nn.Conv2d(96, 128, 5)

#         self.max_pool2d = MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.fc1 = nn.Linear(in_features=768, out_features=100, bias=True) 
        self.fc2 = nn.Linear(in_features=100, out_features=2, bias=True)
        self.flatten = nn.Flatten(start_dim=0, end_dim=-1)

    def forward(self, x):
        print(x.shape)
        x = self.conv1(x)
        print(x.shape)

        x = F.relu(x)
        print(x.shape)

        x = F.max_pool2d(x, 2)
        print(x.shape)
        x = self.conv2(x) # b c h w
        print(x.shape)
        x = F.relu(x)
        print(x.shape)
        x = F.max_pool2d(x, 2)
        print(x.shape)
        x = self.conv3(x)
        print(x.shape)
        x = F.relu(x)
        print(x.shape)
        x = F.max_pool2d(x, 2)
        print(x.shape)

        x = torch.flatten(x, 1) #  b c h w -> b c
        print(x.shape)
        x = self.fc1(x) # b c
        print(x.shape)
        x = F.relu(x)
        print(x.shape)
        x = self.fc2(x)
        print(x.shape)
        output = x
        return output

net = Net()
print(net)


# In[ ]:


from torch.utils.data import Dataset
from torchvision import transforms

class TrainDataset(Dataset):
    def __init__(self):
        
        self.image_paths = train_imgs
        self.label_paths = nose_train_pts
        # TODO: Iterate over files in dataset path, add image and label paths to lists.
        # TODO: Randomly split into train and test partitions. Make sure the random split is the same each time.

        assert len(self.image_paths) == len(self.label_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        label_path = self.label_paths[index]
        
#         transform = transforms.Compose([transforms.PILToTensor()])
        # transform = transforms.PILToTensor()
        # Convert the PIL image to Torch tensor
        print(label_path.shape)

        image_path = image_path.reshape(image_path.shape[0], image_path.shape[1], 1)
        image_path = image_path.transpose((2, 0, 1))

        image_tensor = torch.tensor(image_path).to(torch.float32)
        label_tensor = torch.tensor(label_path)
        return image_tensor, label_tensor

    def __len__(self) -> int:
        return len(self.image_paths)
  


# In[ ]:


class TestDataset(Dataset):
    def __init__(self):
        
        self.image_paths = test_imgs
        self.label_paths = nose_test_pts
        # TODO: Iterate over files in dataset path, add image and label paths to lists.
        # TODO: Randomly split into train and test partitions. Make sure the random split is the same each time.

        assert len(self.image_paths) == len(self.label_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        label_path = self.label_paths[index]
        
#         transform = transforms.Compose([transforms.PILToTensor()])
        # transform = transforms.PILToTensor()
        # Convert the PIL image to Torch tensor
        image_path = image_path.reshape(image_path.shape[0], image_path.shape[1], 1)
        image_path = image_path.transpose((2, 0, 1))
        print(image_path)
        image_tensor = torch.tensor(image_path).to(torch.float32)
        label_tensor = torch.tensor(label_path)
        return image_tensor, label_tensor

    def __len__(self) -> int:
        return len(self.image_paths)


# In[ ]:


train_data = TrainDataset()
# image_tensor, label_tensor = data
len(train_data)
# torch.max(image_tensor)
# label_tensor


# In[ ]:


test_data = TestDataset()
# image_tensor, label_tensor = data
len(test_data)
# torch.max(image_tensor)
# label_tensor


# In[ ]:


from torch.utils.data import DataLoader

train_loader = DataLoader(train_data, batch_size = 1)
test_loader = DataLoader(test_data, batch_size = 1)


# for i, data in enumerate(train_loader):
#     print(i , data)


# In[ ]:





# In[ ]:


net = Net()
criterion = nn.MSELoss()  # nn.MSELoss() nn.L1Loss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.003)


# In[ ]:


training_loader = list(train_loader)[:152]

validation_loader = list(train_loader)[152:193]

print(len(validation_loader))


# In[ ]:


# loop over the dataset multiple times
num_epochs = 15
loss_values = []
valid_loss_values = []

for epoch in range(num_epochs):  

    running_losses = []
    running_loss = 0.0
    valid_running_loss = 0.0
    
    for i, data in enumerate(training_loader):

        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        print("input", inputs.shape)
        print("labels", labels.shape)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward pass
        outputs = net(inputs)
        print("outut", outputs.shape)

        # compute loss
        loss = criterion(outputs, labels)

        # backward pass
        loss.backward()

        # update network parameters
        optimizer.step()

        # print statistics
        running_losses.append(loss.item())
        
        running_loss += loss.item() * inputs.size(0) 

        if i % 50 == 0:    # print every 50 minibatches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, sum(running_losses) / len(running_losses)))
        
    loss_values.append(running_loss / len(training_loader))
    
    
    for i, data in enumerate(validation_loader):
            inputs, labels = data
            
            # calculate outputs
            outputs = net(inputs)

            # compute loss
            loss = criterion(outputs, labels)

            # print statistics
            running_losses.append(loss.item())

            running_loss += loss.item() * inputs.size(0) 


    valid_loss_values.append(running_loss / len(validation_loader))

print('Finished Training')

plt.plot(loss_values)
plt.plot(valid_loss_values)
plt.ylabel('MSE Loss')
plt.xlabel('Epoch #')



# In[ ]:


with torch.no_grad():
    
    # initialize a list to store our predictions
    preds = []
    
    for i, data in enumerate(test_loader):
        images, labels = data

        # calculate outputs by running images through the network
        outputs = net(images)
        preds.append(outputs)
        
print(preds)
# plt.plot(loss_values)
# plt.ylabel('MSE Loss')
# plt.xlabel('Epoch #')  
    
    


# In[ ]:


plt.figure(figsize=(3,4))

i = 1

implot = plt.imshow(test_imgs[i],  cmap = "gray")

plt.scatter(nose_test_pts[i][0]*80, nose_test_pts[i][1]*60, c = 'r', s= 10)
plt.scatter(preds[i][0][0]*80, preds[i][0][1]*60, c = 'b', s= 10)




plt.show()


# ## Part 2

# In[ ]:


# load in training data

face_train_pts = []
face_test_pts = []

for j in range(1, 7): # viewpoint index
    for i in range(1, 41):  # person index
        if i == 8 or i == 12 or i == 14 or i == 15 or i == 22 or i == 30 or i == 35:
            gender = 'f' # gender

        else:
            gender = 'm'
        
        root_dir = './imm_face_db/'

        # load all facial keypoints/landmarks
        file = open(root_dir + '{:02d}-{:d}{}.asf'.format(i,j,gender))
        points = file.readlines()[16:74]
        landmark = []

        for point in points:
            x,y = point.split('\t')[2:4]
            landmark.append([float(x), float(y)])
            
            # the nose keypoint
        face_keypoint = np.array(landmark).astype('float32')
        
        if (i < 33):
            face_train_pts.append(face_keypoint)
        else:
            face_test_pts.append(face_keypoint)



# In[ ]:


face_train_pts[23]


# In[ ]:


face_train_pts


# In[ ]:


face_train_imgs = []
face_test_imgs = []
for j in range(1, 7): # viewpoint index
    for i in range(1, 41):  # person index
        if i == 8 or i == 12 or i == 14 or i == 15 or i == 22 or i == 30 or i == 35:
            gender = 'f' # gender

        else:
            gender = 'm'
        
        root_dir = './imm_face_db/'

        # load all facial keypoints/landmarks
        path = root_dir + '{:02d}-{:d}{}.jpg'.format(i,j,gender)

        img = cv2.imread(str(path))

        # You may need to convert the color.
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        normal_img = ((gray_img.astype(np.float32))/255) - 0.5
        print(normal_img)
        
        resize_img = cv2.resize(normal_img, (160, 120), interpolation = cv2.INTER_LINEAR)
        print(resize_img.shape)
#         im_pil = Image.fromarray(resize_img)
        
        if (i < 33):
            face_train_imgs.append(resize_img)
        else:
            face_test_imgs.append(resize_img)


# In[ ]:


import imgaug.augmenters as iaa

class FaceTrainDataset(Dataset):
    def __init__(self):
        
        self.image_paths = face_train_imgs
        self.label_paths = face_train_pts
        # TODO: Iterate over files in dataset path, add image and label paths to lists.
        # TODO: Randomly split into train and test partitions. Make sure the random split is the same each time.

        assert len(self.image_paths) == len(self.label_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        label_path = self.label_paths[index]
        
        # Convert the PIL image to Torch tensor

#         image_path = image_path.astype(np.uint8)
#         image_path = image_path.reshape(1 , image_path.shape[0], image_path.shape[1])
        
        seq = iaa.SomeOf(2, 
                [iaa.AdditiveGaussianNoise(scale=0.05*255),
                iaa.Affine(translate_px={"x": (-10, 10)}),
                iaa.Affine(rotate = (-15, 15)),
                iaa.Fliplr(1),
                iaa.Sharpen(alpha=0.5),
                ])
        
        label_path = label_path.reshape((1, 58, 2))
        images_aug, points_aug = seq(images=image_path, keypoints=label_path)

        label_path = label_path.reshape((58, 2))
# #         print(images_aug.shape)
# #         print(points_aug)
#         image_path = (images_aug/255) - 0.5
#         image_tensor = torch.tensor(image_path)
#         label_tensor = torch.tensor(points_aug)
#         return image_tensor, label_tensor

#         image_path = self.image_paths[index]
#         label_path = self.label_paths[index]


        image_path = image_path.reshape(image_path.shape[0], image_path.shape[1], 1)
        image_path = image_path.transpose((2, 0, 1))

        image_tensor = torch.tensor(image_path).to(torch.float32)
        label_tensor = torch.tensor(label_path)
        label_tensor = torch.reshape(label_tensor, (116, 1))
        return image_tensor, label_tensor

    def __len__(self) -> int:
        return len(self.image_paths)
  


# In[ ]:


face_train_data = FaceTrainDataset()
# image_tensor, label_tensor = data
len(face_train_data)


# In[ ]:


face_train_loader = DataLoader(face_train_data, batch_size = 1)
face_training_loader = list(face_train_loader)[:152]
face_validation_loader = list(face_train_loader)[152:193]


# In[ ]:


list(face_train_loader)[0][0]


# In[ ]:


face_test_data = FaceTestDataset()
face_test_loader = DataLoader(face_test_data, batch_size = 1)


# In[ ]:


plt.figure(figsize=(6,8))
i = 1 #batch number #(1-24)
j = 1 #face in batch (1-4)
face = face_training_loader[i][0][j].numpy()
print(face.shape)
implot = plt.imshow(face.reshape(face.shape[1], face.shape[2], 1),  cmap = "gray")

xs = []
ys = []
for pt in face_training_loader[i][1][j].numpy():
    for index in pt:
        xs.append(index[0] * 160)
        ys.append(index[1] * 120)
        
        
plt.scatter(np.asarray(xs), np.asarray(ys), c = 'r', s= 6)


plt.show()


# In[ ]:


class FaceNet(nn.Module):
    def __init__(self):
        super(FaceNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5) # 60 x 80 (H x W)
        self.conv2 = nn.Conv2d(16, 24, 5) 
        self.conv3 = nn.Conv2d(24, 32, 1)
        self.conv4 = nn.Conv2d(32, 64, 1)
        self.conv5 = nn.Conv2d(64, 128, 1)

        self.fc1 = nn.Linear(in_features=1536, out_features=256, bias=True) 
        self.fc2 = nn.Linear(in_features=256, out_features=58*2, bias=True)
        self.flatten = nn.Flatten(start_dim=0, end_dim=-1)

    def forward(self, x):
#         print(x.shape)
        x = self.conv1(x)
#         print(x.shape)

        x = F.relu(x)
#         print(x.shape)

        x = F.max_pool2d(x, 2)
#         print(x.shape)
        x = self.conv2(x) # b c h w
#         print("conv2", x.shape)
        x = F.relu(x)
#         print(x.shape)
        x = F.max_pool2d(x, 2)
#         print(x.shape)
        x = self.conv3(x)
#         print("conv3", x.shape)
        x = F.relu(x)
#         print(x.shape)
        x = F.max_pool2d(x, 2)
#         print("last 1" , x.shape)
        x = self.conv4(x)
#         print("conv4", x.shape)
        x = F.relu(x)
#         print(x.shape)
        x = F.max_pool2d(x, 2)
#         print(x.shape)
        x = self.conv5(x)
#         print("conv5", x.shape)
        x = F.relu(x)
#         print(x.shape)
        x = F.max_pool2d(x, 2)
#         print("last pool", x.shape)

        x = torch.flatten(x, 1) #  b c h w -> b c
#         print(x.shape)
        x = self.fc1(x) # b c
#         print(x.shape)
        x = F.relu(x)
#         print(x.shape)
        x = self.fc2(x)
#         print(x.shape)
        output = x
        return output

facenet = FaceNet()
print(facenet)


# import torch
# from torch import nn
# import torch.nn.functional as F

# class CNN2(nn.Module):
    
#     def __init__(self):
#         super(CNN2, self).__init__()
#         self.conv1 = nn.Conv2d(1, 12, 3)
#         self.conv2 = nn.Conv2d(12, 20, 3)
#         self.conv3 = nn.Conv2d(20, 32, 3)
#         self.conv4 = nn.Conv2d(32, 40, 3)
#         self.conv5 = nn.Conv2d(40, 60, 3)
        
#         self.fully_connected_1 = nn.Linear(180, 140)
#         self.fully_connected_2 = nn.Linear(140, 116)
        
#     def forward(self, x):
#         x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=2)
#         x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=2)
#         x = F.max_pool2d(F.relu(self.conv3(x)), kernel_size=2)
#         x = F.max_pool2d(F.relu(self.conv4(x)), kernel_size=2)
#         x = F.max_pool2d(F.relu(self.conv5(x)), kernel_size=2)
        
#         x = torch.flatten(x, 1)
        
#         x = F.relu(self.fully_connected_1(x))
#         x = self.fully_connected_2(x)
    
#         return x
    
# facenet = CNN2()


# In[ ]:


# facenet = CNN2()
# epochs = 25
# size = (120, 160)
# transform = transforms.Compose([Rescale(size), Augment(), ToTensor()])
# train_dataset, test_dataset, dataloader_test, dataloader_test = getDataloaders(size, transform, 1)
# train_losses, test_losses, prediction = run_epoch(facenet, epochs, nn.MSELoss(), dataloader_test, dataloader_test)


# In[ ]:


# _, _, _, dataloader_test = getDataloaders(size, transform, 1)
# i = 0
# for pred in prediction:
#     pred = pred.reshape(58,2)
#     image = torch.reshape(dataloader_test.dataset[i]['image'], (120,160))
#     keypoints = torch.reshape(dataloader_test.dataset[i]['landmarks'],(58,2))
#     plt.imshow(image, cmap='gray')
#     plt.scatter(160*keypoints[:,0], 120*keypoints[:,1], color='green')
#     plt.scatter(160*pred[:,0], 120*pred[:,1], color='red')
#     plt.show()
#     i+=1
# epoch = [i for i in range(len(train_losses))]
# plt.plot(epoch, train_losses)
# plt.plot(epoch, test_losses)
# plt.show()


# In[ ]:


facenet = FaceNet()
criterion = nn.MSELoss()  # nn.MSELoss() nn.L1Loss()
optimizer = torch.optim.Adam(facenet.parameters(), lr=1e-3)


# In[ ]:


# loop over the dataset multiple times
num_epochs = 25
loss_values = []
valid_loss_values = []

for epoch in range(num_epochs):  

    running_losses = []
    running_loss = 0.0
    valid_running_loss = 0.0
    
    facenet.train()
    
    for i, data in enumerate(face_training_loader):

        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
#         print("input", inputs.shape)
#         print(inputs.cpu().detach().numpy().shape)
#         plt.imshow(inputs.cpu().detach().numpy()[0,0,:,:])
#         pts = labels.cpu().detach().numpy()[0,:,:].reshape(58, 2)
#         plt.scatter(160*pts[:,0], 120*pts[:,1])
#         plt.show()
#         print("labels", labels.shape)
        # 16,1,120,160
        
        # 16,58,2
        
#         inputs = inputs.float()

        # zero the parameter gradients
        facenet.zero_grad()
        
#         print(labels.shape)

        # forward pass
        outputs = facenet(inputs)
#         print("output" , outputs.shape)
        # 16,58,2
        
        # compute loss
#         outputs = outputs.reshape(labels.shape[0], labels.shape[1], labels.shape[2])

        loss = criterion(outputs, labels[:,:,0])

        # backward pass
        loss.backward()

        # update network parameters
        optimizer.step()

        # print statistics
        running_losses.append(loss.item())
        print(loss)
        
        running_loss += loss * inputs.size(0) 

#         if i % 50 == 0:    # print every 50 minibatches
#             print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, sum(running_losses) / len(running_losses)))
#     print(running_loss)
    loss_values.append(running_loss.detach().numpy() / len(face_training_loader))
    
    facenet.eval()
    for i, data in enumerate(face_validation_loader):
            inputs, labels = data
            
            # calculate outputs
            outputs = facenet(inputs)
#             print(inputs.size)

            # compute loss
            loss = criterion(outputs, labels[:,:,0])
#             print(loss)
            loss = loss.detach().numpy()

            # print statistics
            running_losses.append(loss.item())

            running_loss += loss * inputs.size(0) 

#             if i % 50 == 0:    # print every 50 minibatches
#                 print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, sum(running_losses) / len(running_losses)))


    valid_loss_values.append(running_loss.detach().numpy() / len(face_validation_loader))
    
    

print('Finished Training')

plt.plot(loss_values)
plt.plot(valid_loss_values)
plt.ylabel('MSE Loss')
plt.xlabel('Epoch #')
# plt.savefig("facepts_mse.jpg")


# In[ ]:


for i, data in enumerate(face_test_loader):
    inputs, labels = data
    outputs = facenet(inputs)
    plt.imshow(inputs.cpu().detach().numpy()[0,0,:,:], cmap='gray')
    pts = outputs.cpu().detach().numpy().reshape(58, 2)
    plt.scatter(160*pts[:,0], 120*pts[:,1])
    plt.show()


# In[ ]:


#visualize learned filter
weights = facenet.conv1.weight.cpu().detach().numpy() 
print(len(weights))
# for i in range(len(weights)):
plt.figure()
plt.imshow(np.squeeze(weights[15]), cmap = "gray")
# plt.savefig("conv1_filter15.jpg")


# In[ ]:


with torch.no_grad():
    
    # initialize a list to store our predictions
    preds = []
    
    for i, data in enumerate(face_test_loader):
        images, labels = data
        print(images.shape)
        print(labels.shape)

        # calculate outputs by running images through the network
        outputs = facenet(images)
        print(outputs.shape)
        preds.append(outputs)
        
print(preds)
# plt.plot(loss_values)
# plt.ylabel('MSE Loss')
# plt.xlabel('Epoch #')  
    


# In[ ]:


plt.figure(figsize=(3,4))

i = 1

implot = plt.imshow(face_test_loader[i].numpy(),  cmap = "gray")

# plt.scatter(nose_test_pts[i][0]*80, nose_test_pts[i][1]*60, c = 'r', s= 10)
plt.scatter(preds[i][0][0]*160, preds[i][0][1]*120, c = 'b', s= 10)




plt.show()



plt.figure(figsize=(6,8))
i = 1 #batch number #(1-24)
j = 1 #face in batch (1-8)
face = face_training_loader[i][0][j].numpy()
print(face.shape)
implot = plt.imshow(face.reshape(face.shape[1], face.shape[2], 1),  cmap = "gray")

xs = []
ys = []
for pt in face_training_loader[i][1][j].numpy():
    for index in pt:
        xs.append(index[0] * 160)
        ys.append(index[1] * 120)
        
        
plt.scatter(np.asarray(xs), np.asarray(ys), c = 'r', s= 6)


plt.show()


# In[ ]:





# In[ ]:





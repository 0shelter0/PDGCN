import numpy as np
import skimage.io
import skimage.transform

import torch
import torchvision.transforms as transforms
from torch.utils import data
import torchvision.models as models

from PIL import Image
import random

import os
import pickle
"""
Reference:
https://github.com/cvlab-epfl/social-scene-understanding/blob/master/volleyball.py
"""

ACTIVITIES = ['r_set', 'r_spike', 'r-pass', 'r_winpoint',
              'l_set', 'l-spike', 'l-pass', 'l_winpoint']

NUM_ACTIVITIES = 8

ACTIONS = ['blocking', 'digging', 'falling', 'jumping',
           'moving', 'setting', 'spiking', 'standing',
           'waiting']
NUM_ACTIONS = 9


def volley_read_annotations(path):
    """
    reading annotations for the given sequence
    """
    annotations = {}

    gact_to_id = {name: i for i, name in enumerate(ACTIVITIES)}
    act_to_id = {name: i for i, name in enumerate(ACTIONS)}

    with open(path) as f:
        for l in f.readlines():
            values = l[:-1].split(' ')
            file_name = values[0]
            activity = gact_to_id[values[1]]

            values = values[2:]
            num_people = len(values) // 5

            action_names = values[4::5]
            actions = [act_to_id[name]
                       for name in action_names]

            def _read_bbox(xywh):
                x, y, w, h = map(int, xywh)
                return y, x, y+h, x+w
            bboxes = np.array([_read_bbox(values[i:i+4])
                               for i in range(0, 5*num_people, 5)])

            fid = int(file_name.split('.')[0])
            annotations[fid] = {
                'file_name': file_name,
                'group_activity': activity,
                'actions': actions,
                'bboxes': bboxes,
            }
    return annotations


def volley_read_dataset(path, seqs):
    data = {}
    for sid in seqs:
        data[sid] = volley_read_annotations(path + '/%d/annotations.txt' % sid)
    return data


def volley_all_frames(data):
    frames = []
    for sid, anns in data.items():
        for fid, ann in anns.items():
            frames.append((sid, fid))
    return frames


def volley_random_frames(data, num_frames):
    frames = []
    for sid in np.random.choice(list(data.keys()), num_frames):
        fid = int(np.random.choice(list(data[sid]), []))
        frames.append((sid, fid))
    return frames


def volley_frames_around(frame, num_before=5, num_after=4):
    sid, src_fid = frame
    return [(sid, src_fid, fid)
            for fid in range(src_fid-num_before, src_fid+num_after+1)]


def transform_img(images, img_h, img_w):
    """
     preprocess images
     Args:
         images: PIL Image object
    """
    transform = transforms.Compose([
        transforms.Resize((img_h, img_w)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    images = transform(images)
    return images


def load_samples_sequence(anns,tracks,images_path,frames,image_size,num_boxes=12,):
    """
    load samples of a bath
    
    Returns:
        pytorch tensors
    """
    images, boxes, boxes_idx = [], [], []
    activities, actions = [], []
    
    for i, (sid, src_fid, fid) in enumerate(frames):
        #img=skimage.io.imread(images_path + '/%d/%d/%d.jpg' % (sid, src_fid, fid))
        #img=skimage.transform.resize(img,(720, 1280),anti_aliasing=True)
        
        img = Image.open(images_path + '/%d/%d/%d.jpg' % (sid, src_fid, fid))

        use_imageNet_normal = False
        if use_imageNet_normal:
            img_h, img_w = image_size
            transform = transforms.Compose([
                transforms.Resize((img_h, img_w)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            img = transform(img)
        else:
            img=transforms.functional.resize(img,image_size)
            img=np.array(img)
            
            #  H,W,3 -> 3,H,W
            img=img.transpose(2,0,1)
        
        images.append(img)

        boxes.append(tracks[(sid, src_fid)][fid])
        actions.append(anns[sid][src_fid]['actions'])
        if len(boxes[-1]) != num_boxes:
          boxes[-1] = np.vstack([boxes[-1], boxes[-1][:num_boxes-len(boxes[-1])]])
          actions[-1] = actions[-1] + actions[-1][:num_boxes-len(actions[-1])]
        boxes_idx.append(i * np.ones(num_boxes, dtype=np.int32))
        activities.append(anns[sid][src_fid]['group_activity'])

    

    # images = np.stack(images)
    activities = np.array(activities, dtype=np.int32)
    bboxes = np.vstack(boxes).reshape([-1, num_boxes, 4])
    bboxes_idx = np.hstack(boxes_idx).reshape([-1, num_boxes])
    actions = np.hstack(actions).reshape([-1, num_boxes])
    
    #convert to pytorch tensor
    # images=torch.from_numpy(images).float()
    

    if use_imageNet_normal:
        images = torch.stack(images)
    else:
        images = np.stack(images)
        images=torch.from_numpy(images).float()

    bboxes=torch.from_numpy(bboxes).float()
    bboxes_idx=torch.from_numpy(bboxes_idx).int()
    actions=torch.from_numpy(actions).long()
    activities=torch.from_numpy(activities).long()

    return images, bboxes, bboxes_idx, actions, activities


class VolleyballDataset(data.Dataset):
    """
    Characterize volleyball dataset for pytorch
    """
    def __init__(self,anns,tracks,frames,images_path,image_size,feature_size,inference_module_name,num_boxes=12,num_before=4,num_after=4,is_training=True,is_finetune=False):
        self.anns=anns
        self.tracks=tracks
        self.frames=frames
        self.images_path=images_path
        self.image_size=image_size
        self.feature_size=feature_size
        self.inference_module_name = inference_module_name
        
        self.num_boxes=num_boxes
        self.num_before=num_before
        self.num_after=num_after
        self.num_segment=3
        
        self.is_training=is_training
        self.is_finetune=is_finetune

        # self.frames_seq = np.empty((1337, 2), dtype = np.int)
        # self.flag = 0

    def __len__(self):
        """
        Return the total number of samples
        """
        return len(self.frames)
    
    def __getitem__(self,index):
        """
        Generate one sample of the dataset
        """
        # Save frame sequences
        # self.frames_seq[self.flag] = self.frames[index]# [0], self.frames[index][1]
        # if self.flag == 1336:
        #     save_seq = self.frames_seq
        #     np.savetxt('vis/frames_seq.txt', save_seq)
        # self.flag += 1

        select_frames = self.volley_frames_sample(self.frames[index])
        sample = self.load_samples_sequence(select_frames)
        
        return sample
    
    def volley_frames_sample(self,frame):
        sid, src_fid = frame
        
        if self.is_finetune:
            if self.is_training:
                fid=random.randint(src_fid-self.num_before, src_fid+self.num_after)
                return [(sid, src_fid, fid)]
            else:
                return [(sid, src_fid, fid)
                        for fid in range(src_fid-self.num_before, src_fid+self.num_after+1)]
        else:
            # if self.is_training:
            #     sample_frames=random.sample(range(src_fid-self.num_before, src_fid+self.num_after+1), 3)
            #     return [(sid, src_fid, fid)
            #             for fid in sample_frames]
            # else:
            #     return [(sid, src_fid, fid)
            #             for fid in  [src_fid-3,src_fid,src_fid+3, src_fid-4,src_fid-1,src_fid+2, src_fid-2,src_fid+1,src_fid+4 ]]
            if self.inference_module_name == 'arg_volleyball':
                if self.is_training:
                    sample_frames=random.sample(range(src_fid-self.num_before, src_fid+self.num_after+1), 3)
                    return [(sid, src_fid, fid)
                            for fid in sample_frames]
                else:
                    return [(sid, src_fid, fid)
                            for fid in  [src_fid-3,src_fid,src_fid+3, src_fid-4,src_fid-1,src_fid+2, src_fid-2,src_fid+1,src_fid+4 ]]
            else:
                # if self.is_training:
                #     return [(sid, src_fid, fid)  for fid in range(src_fid-self.num_before, src_fid+self.num_after+1)]
                # else:
                #     return [(sid, src_fid, fid) for fid in range(src_fid - self.num_before, src_fid + self.num_after + 1)]
                if self.is_training:
                    num_frames_segment = (self.num_before+self.num_after+1) // self.num_segment # 3
                    start = src_fid-self.num_before
                    fids=[]
                    for i in range(self.num_segment):
                        if i==self.num_segment-1: #
                            end = src_fid+self.num_after
                            fid = random.randint(start,end)
                        else:
                            end = start + num_frames_segment -1
                            fid = random.randint(start, end)
                            start = end + 1
                        fids.append(fid)

                    return [(sid, src_fid, fid)  for fid in fids]

                    # sample_frames = random.sample(range(src_fid - self.num_before, src_fid), 3) + [src_fid] + \
                    #         random.sample(range(src_fid + 1, src_fid + self.num_after + 1), 3)
                    # sample_frames.sort()
                    # return [(sid, src_fid, fid)  for fid in sample_frames]

                else:                            
                    num_frames_segment = (self.num_before+self.num_after+1) // self.num_segment # 3
                    start = src_fid-self.num_before
                    fids=[]
                    for i in range(self.num_segment):
                        if i==self.num_segment-1: #
                            end = src_fid+self.num_after
                            fid = random.randint(start,end)
                        else:
                            end = start + num_frames_segment -1
                            fid = random.randint(start, end)
                            start = end + 1
                        fids.append(fid)

                    return [(sid, src_fid, fid)  for fid in fids]
                    # return [(sid, src_fid, fid)  for fid in range(src_fid - 3, src_fid + 4, 1)]



    def load_samples_sequence(self,select_frames):
        """
        load samples sequence

        Returns:
            pytorch tensors
        """
        
        OH, OW=self.feature_size
        use_imageNet_normal = False
        
        images, boxes = [], []
        activities, actions = [], []
        poses = []

        video_id = select_frames[0][0]
        clip_id = select_frames[0][1]
        dataset_dir = '/home/shelter/shelterX/data/volleyball'
        joints_path = os.path.join(dataset_dir, 'joints', str(video_id), f'{clip_id}.pickle')
        with open(joints_path, 'rb') as f:
            joint_raw = pickle.load(f) # dict
        
        for i, (sid, src_fid, fid) in enumerate(select_frames):

            img = Image.open(self.images_path + '/%d/%d/%d.jpg' % (sid, src_fid, fid))
            H,W = self.image_size
            
            if use_imageNet_normal:
                img_h, img_w = self.image_size
                transform = transforms.Compose([
                    transforms.Resize((img_h, img_w)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])
                img = transform(img)
            else:
                img=transforms.functional.resize(img, self.image_size)
                img=np.array(img)
            
                #  H,W,3 -> 3,H,W
                img=img.transpose(2,0,1)

            # img=transforms.functional.resize(img,self.image_size)
            # img=np.array(img)

            # # H,W,3 -> 3,H,W
            # img=img.transpose(2,0,1)

            images.append(img)

            temp_boxes = np.ones_like(self.tracks[(sid, src_fid)][fid])
            this_frame_poses = []
            # pose prepare
            joints_this_frame = joint_raw[fid] # [12, 17, 3]
            joints_all_people  = joints_this_frame[:, :, 0:2] # x,y
            for i,track in enumerate(self.tracks[(sid, src_fid)][fid]):
                
                y1,x1,y2,x2 = track
                w1,h1,w2,h2 = x1*OW, y1*OH, x2*OW, y2*OH  
                temp_boxes[i]=np.array([w1,h1,w2,h2])

                X1 = int(round(x1*W))
                Y1 = int(round(y1*H))
                X2 = int(round(x2*W))
                Y2 = int(round(y2*H))

                X1 = min(max(X1,0),W)
                X2 = min(max(X2,0),W)
                Y1 = min(max(Y1,0),H)
                Y2 = min(max(Y2,0),H)
                center_this_person = [(X1+X2)/2.,(Y1+Y2)/2.]
                size = np.sqrt((X2-X1)*(Y2-Y1)/4)
                joints_this_person = joints_all_people[i] #[17, 2]
                joints_this_person = joints_this_person.reshape(17,2)
                center_this_person = np.array(center_this_person)
                joints_this_person = (joints_this_person - center_this_person) / size
                this_frame_poses.append(joints_this_person)

            if len(this_frame_poses) != self.num_boxes:
                this_frame_poses = this_frame_poses + this_frame_poses[:self.num_boxes-len(this_frame_poses)]
            this_frame_poses = np.vstack(this_frame_poses)
            boxes.append(temp_boxes)
            poses.append(this_frame_poses) # [T, 12, 17, 2]
            
            
            actions.append(self.anns[sid][src_fid]['actions'])
            
            if len(boxes[-1]) != self.num_boxes:
                boxes[-1] = np.vstack([boxes[-1], boxes[-1][:self.num_boxes-len(boxes[-1])]])
                actions[-1] = actions[-1] + actions[-1][:self.num_boxes-len(actions[-1])]
            activities.append(self.anns[sid][src_fid]['group_activity'])

        
        activities = np.array(activities, dtype=np.int32)
        bboxes = np.vstack(boxes).reshape([-1, self.num_boxes, 4])
        actions = np.hstack(actions).reshape([-1, self.num_boxes])
        poses = np.vstack(poses).reshape([-1, self.num_boxes, 17, 2])
        

        #convert to pytorch tensor
        # images = np.stack(images)
        # images = torch.from_numpy(images).float()

        if use_imageNet_normal:
            images = torch.stack(images)
        else:
            images = np.stack(images)
            images=torch.from_numpy(images).float()


        bboxes = torch.from_numpy(bboxes).float()#[T, 12, 4]
        poses = torch.from_numpy(poses).float() #[T*12, 17, 2]
        actions = torch.from_numpy(actions).long()
        activities = torch.from_numpy(activities).long()

        return images, bboxes,  actions, activities, poses
    

import torch
from torch.utils import data
import torchvision.models as models
import torchvision.transforms as transforms

import os
import random
from PIL import Image
import numpy as np
import pickle

from collections import Counter


FRAMES_NUM={1: 302, 2: 347, 3: 194, 4: 257, 5: 536, 6: 401, 7: 968, 8: 221, 9: 356, 10: 302, 
            11: 1813, 12: 1084, 13: 851, 14: 723, 15: 464, 16: 1021, 17: 905, 18: 600, 19: 203, 20: 342, 
            21: 650, 22: 361, 23: 311, 24: 321, 25: 617, 26: 734, 27: 1804, 28: 470, 29: 635, 30: 356, 
            31: 690, 32: 194, 33: 193, 34: 395, 35: 707, 36: 914, 37: 1049, 38: 653, 39: 518, 40: 401, 
            41: 707, 42: 420, 43: 410, 44: 356}

 
FRAMES_SIZE={1: (480, 720), 2: (480, 720), 3: (480, 720), 4: (480, 720), 5: (480, 720), 6: (480, 720), 7: (480, 720), 8: (480, 720), 9: (480, 720), 10: (480, 720), 
             11: (480, 720), 12: (480, 720), 13: (480, 720), 14: (480, 720), 15: (450, 800), 16: (480, 720), 17: (480, 720), 18: (480, 720), 19: (480, 720), 20: (450, 800), 
             21: (450, 800), 22: (450, 800), 23: (450, 800), 24: (450, 800), 25: (480, 720), 26: (480, 720), 27: (480, 720), 28: (480, 720), 29: (480, 720), 30: (480, 720), 
             31: (480, 720), 32: (480, 720), 33: (480, 720), 34: (480, 720), 35: (480, 720), 36: (480, 720), 37: (480, 720), 38: (480, 720), 39: (480, 720), 40: (480, 720), 
             41: (480, 720), 42: (480, 720), 43: (480, 720), 44: (480, 720)}


ACTIONS=['NA','Crossing','Waiting','Queueing','Walking','Talking']
ACTIVITIES=['Crossing','Waiting','Queueing','Walking','Talking']


ACTIONS_ID={a:i for i,a in enumerate(ACTIONS)}
ACTIVITIES_ID={a:i for i,a in enumerate(ACTIVITIES)}
Action6to5 = {0:0, 1:1, 2:2, 3:3, 4:1, 5:4}
# 'NA' 'Moving' 'Waiting'  'Queueing' 'Talking'
Activity5to4 = {0:0, 1:1, 2:2, 3:0, 4:3}
# 'Moving' 'Waiting'  'Queueing' 'Talking'


def collective_read_annotations(path,sid):
    annotations={}
    path=path + '/seq%02d/annotations.txt' % sid
    
    with open(path,mode='r') as f:
        frame_id=None
        group_activity=None
        actions=[]
        bboxes=[]
        for l in f.readlines():
            values=l[:-1].split('	')
            
            if int(values[0])!=frame_id:
                if frame_id!=None and frame_id%10==1 and frame_id+9<=FRAMES_NUM[sid]:
                    counter = Counter(actions).most_common(2)
                    group_activity= counter[0][0]-1 if counter[0][0]!=0 else counter[1][0]-1
                    annotations[frame_id]={ 
                        'frame_id':frame_id,
                        'group_activity':group_activity,
                        'actions':actions,
                        'bboxes':bboxes
                    }
                    
                frame_id=int(values[0])
                group_activity=None
                actions=[]
                bboxes=[]
                
            actions.append(int(values[5])-1)
            x,y,w,h = (int(values[i])  for i  in range(1,5))
            H,W=FRAMES_SIZE[sid]
            
            bboxes.append( (y/H,x/W,(y+h)/H,(x+w)/W) )
        
        if frame_id!=None and frame_id%10==1 and frame_id+9<=FRAMES_NUM[sid]: # 最后一个10帧
            counter = Counter(actions).most_common(2)
            group_activity= counter[0][0]-1 if counter[0][0]!=0 else counter[1][0]-1
            annotations[frame_id]={
                'frame_id':frame_id,
                'group_activity':group_activity,
                'actions':actions,
                'bboxes':bboxes
            }

    return annotations
            
        
        
def collective_read_dataset(path,seqs):
    data = {}
    for sid in seqs:
        data[sid] = collective_read_annotations(path,sid)
    return data

def collective_all_frames(anns):
    return [(s,f)  for s in anns for f in anns[s] ]


class CollectiveDataset(data.Dataset):
    """
    Characterize collective dataset for pytorch
    """
    def __init__(self,anns,tracks,frames,images_path,image_size,feature_size,num_boxes=13, num_frames = 10, is_training=True,is_finetune=False):
        self.anns=anns
        self.frames=frames
        self.images_path=images_path
        self.image_size=image_size
        self.feature_size=feature_size
        
        self.num_boxes = num_boxes
        self.num_frames = num_frames
        
        self.is_training=is_training
        self.is_finetune=is_finetune

        self.tracks = tracks

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
        # self.frames_seq[self.flag] = self.frames[index] # [0], self.frames[index][1]
        # if self.flag == 764: # 1336
        #     save_seq = self.frames_seq
        #     np.savetxt('vis/Collective/frames_seq.txt', save_seq)
        # self.flag += 1

        select_frames=self.get_frames(self.frames[index])
        sample=self.load_samples_sequence(select_frames)
        
        return sample
    
    def get_frames(self,frame):
        
        sid, src_fid = frame
        
        if self.is_finetune:
            if self.is_training:
                num_frames = 10
                fid=random.randint(src_fid, src_fid+num_frames-1)
                return [(sid, src_fid, fid)]
        
            else:
                num_frames = 10
                return [(sid, src_fid, fid) 
                        for fid in range(src_fid, src_fid+num_frames)]
            
        else:
            # if self.is_training:
            #     sample_frames=random.sample(range(src_fid,src_fid+self.num_frames),3)
            #     return [(sid, src_fid, fid) for fid in sample_frames]
            #
            # else:
            #     sample_frames=[ src_fid, src_fid+3, src_fid+6, src_fid+1, src_fid+4, src_fid+7, src_fid+2, src_fid+5, src_fid+8 ]
            #     return [(sid, src_fid, fid) for fid in sample_frames]
            if self.is_training:
                # return [(sid, src_fid, fid)  for fid in range(src_fid , src_fid + self.num_frames)]
                num_frames_segment = 3
                num_segment = 3
                start = src_fid + 1
                fids = [src_fid]
                for i in range(num_segment):
                    if i == num_segment-1: #
                        end = src_fid + 9
                        fid = random.randint(start,end)
                    else:
                        end = start + num_frames_segment -1
                        fid = random.randint(start, end)
                        start = end + 1
                    fids.append(fid)
                return [(sid, src_fid, fid)  for fid in fids]

            else:
                # return [(sid, src_fid, fid) for fid in range(src_fid, src_fid + self.num_frames)]
                num_frames_segment = 3
                num_segment = 3
                start = src_fid + 1
                fids = [src_fid]
                for i in range(num_segment):
                    if i == num_segment-1: #
                        end = src_fid + 9
                        fid = random.randint(start,end)
                    else:
                        end = start + num_frames_segment -1
                        fid = random.randint(start, end)
                        start = end + 1
                    fids.append(fid)
                return [(sid, src_fid, fid)  for fid in fids]

    
    
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
        bboxes_num=[]
        poses = []

        video_id = select_frames[0][0]
        clip_id = select_frames[0][1]
        dataset_dir = '/home/shelter/shelterX/data/collective'
        joints_path = os.path.join(dataset_dir, 'joints', str(video_id), f'{clip_id}.pickle')
        with open(joints_path, 'rb') as f:
            joint_raw = pickle.load(f) # dict

        for i, (sid, src_fid, fid) in enumerate(select_frames):

            img = Image.open(self.images_path + '/seq%02d/frame%04d.jpg'%(sid,fid))

            # W,H = img.size
            H, W = FRAMES_SIZE[sid]
            # img=transforms.functional.resize(img, self.image_size)
            # img=np.array(img)

            # # H,W,3 -> 3,H,W
            # img=img.transpose(2,0,1)

            
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
            images.append(img)

            temp_boxes = np.ones_like(self.tracks[(sid, src_fid)][fid])
            this_frame_poses = []

            # pose prepare
            joints_this_frame = joint_raw[fid] # [12, 17, 3]
            joints_all_people  = joints_this_frame[:, :, 0:2] # x,y

            for i,track in enumerate(self.tracks[(sid, src_fid)][fid]):

                joints_this_person = joints_all_people[i] #[17, 2]
                joints_this_person = joints_this_person.reshape(17,2)
                track_ = np.array(track)
                if np.isnan(np.sum(track_)):
                    temp_boxes[i]=np.array([0.0, 0.0, 0.0, 0.0])
                else:
                    y1,x1,y2,x2 = track
                    w1,h1,w2,h2 = x1*OW, y1*OH, x2*OW, y2*OH  
                    temp_boxes[i]=np.array([w1,h1,w2,h2])
                
                if np.sum(temp_boxes[i]) == 0:
                    this_frame_poses.append(np.zeros((17,2)))
                    continue

                if np.isnan(np.sum(joints_this_person)):
                    # replace nan with zero
                    joints_this_person[np.isnan(joints_this_person)] = 0.0
                    
                    
                if np.sum(joints_this_person) == 0:
                    this_frame_poses.append(np.zeros((17,2)))
                    # print(f'missed tracks:({sid},{src_fid})[{fid}] {i}\n')
                    continue

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
                center_this_person = np.array(center_this_person)
                joints_this_person = (joints_this_person - center_this_person) / size
                this_frame_poses.append(joints_this_person)

            # for box in self.anns[sid][src_fid]['bboxes']:
            #     y1,x1,y2,x2=box
            #     w1,h1,w2,h2 = x1*OW, y1*OH, x2*OW, y2*OH  
            #     temp_boxes.append((w1,h1,w2,h2))
                
            # temp_actions=self.anns[sid][src_fid]['actions'][:]
            # bboxes_num.append(len(temp_boxes))

            while len(this_frame_poses) != self.num_boxes:
                this_frame_poses.append(np.zeros((17,2)))
            
            if len(temp_boxes) != self.num_boxes:
                temp_boxes = np.vstack([temp_boxes, np.zeros((self.num_boxes-len(temp_boxes), 4))])

            this_frame_poses = np.vstack(this_frame_poses)
            poses.append(this_frame_poses)
            boxes.append(temp_boxes)

            temp_actions = [Action6to5[i] for i in self.anns[sid][src_fid]['actions'][:]]
            bboxes_num.append(len(temp_actions))
            
            while len(temp_actions)!=self.num_boxes:
                temp_actions.append(-1)
            actions.append(temp_actions)
            
            activities.append(Activity5to4[self.anns[sid][src_fid]['group_activity']])
        
        
        # images = np.stack(images)
        activities = np.array(activities, dtype=np.int32)
        bboxes_num = np.array(bboxes_num, dtype=np.int32)
        bboxes = np.vstack(boxes).reshape([-1, self.num_boxes, 4])
        poses = np.vstack(poses).reshape([-1, self.num_boxes, 17, 2])
        actions = np.array(actions, dtype=np.int32).reshape(-1,self.num_boxes)
        
        #convert to pytorch tensor
        # images=torch.from_numpy(images).float()
        if use_imageNet_normal:
            images = torch.stack(images)
        else:
            images = np.stack(images)
            images=torch.from_numpy(images).float()
        
        bboxes=torch.from_numpy(bboxes).float()
        actions=torch.from_numpy(actions).long()
        activities=torch.from_numpy(activities).long()
        bboxes_num=torch.from_numpy(bboxes_num).int()
        poses = torch.from_numpy(poses).float()
        
        return images, bboxes,  actions, activities, bboxes_num, poses
    
    

    

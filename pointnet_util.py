import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np
import random
from torch.autograd import Variable
def chuli(l1,l2,l3):
    x = np.array(l1)
    x = x[:,np.newaxis]
    y = np.array(l2)
    y = y[:,np.newaxis]
    z_MF= np.array(l3)
    z = z_MF[:,np.newaxis]

    


    temp = np.hstack((x,y,z))
    
    return K_means(temp)
q=0
def K_means(data):
  
  num = np.shape(data)[0]

  cls = np.zeros([num], np.int)
  new_data = np.zeros(shape=(1,3))
  random_array = np.random.random(size = 3)
  random_array = np.floor(random_array*num)
  rarray = random_array.astype(int)
  center_point =data[rarray]
  change = True  
  global  distance
  while change:
    for i in range(num):
      temp = data[i] - center_point   
      temp = np.square(temp)         
      distance = np.sum(temp,axis=1)  
      cls[i] = np.argmin(distance)   
    
    change = False
  a1=np_count(cls, 0)
  a2=np_count(cls, 1)
  a3=np_count(cls, 2)
 
  a=getThreeNumberMin(a1,a2,a3)
  
  if a==a2:
   aa=0
   bb=2
   
  elif a==a1:
   aa=1
   bb=2
  
  elif a==a3:
   aa=0
   bb=1
  
  
  
  for  i  in range(len(cls)):
     #print(cls)
     if cls[i]==aa or cls[i]==bb   :
       
        new_data= np.insert(new_data,0,data[i],0)
        new_data = torch.as_tensor(new_data) 
  '''  
     else:
        new_data= np.insert(new_data,0,data[i],0)
        new_data = torch.tensor(new_data) 
        print(new_data.shape)
  #print(new_data.shape)
  '''
  new_data=new_data[:-1, :]#去掉初始那一行
  #print(new_data) 
  #print(new_data.shape)
  a,_=data.shape
  #print(a)
  b,_=new_data.shape
  #print(b)
  c=a-b
  
  pad =nn.ConstantPad2d(padding=(0, 0, 0, c), value=1)
  aaa=pad(new_data)
  
  aaa = aaa.unsqueeze(0) #二維變三維前面多了1
  #print(aaa.shape)
  #aaa = aaa.view(1, a, 3).cuda()
  aaa=aaa.cuda()
  torch.cuda.empty_cache()
  return  aaa

def getThreeNumberMin(x,y,z):
    min=x if x<y else y
    min=min if min<z else z
    return min


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


    



def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    
    B, N, _ = src.shape
    _, M, _ = dst.shape
    #src=torch.tensor(src,dtype=torch.float)
    #src=trans_to_cuda(src)
    
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def np_count(nparray, x):
    i = 0
    for n in nparray:
        if n == x:
            i += 1
    return i

def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    #print(points)
    #print(points.shape)
    #print(idx.shape)
    #print(idx.shape)
    
    

    new_points = points[batch_indices, idx, :]
    #print(new_points.shape)
    #print(1111)
    
    #print(xyz222.shape)
    #print(1111)
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
        '''
        aa = random.randint(0,9)
        if aa >5:
            
         mask = dist < distance
         distance[mask] = dist[mask]
         farthest = torch.max(distance, -1)[1]
        else:
          mask = dist
          distance[mask] = 1
          farthest = torch.max(distance, -1)[1]
        '''
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]11111111111111999999999999999999999999999999999999999999999999
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    #print(N)
    if N < 200:
     xyz0=chuli(xyz.cpu().numpy()[0][:,0],xyz.cpu().numpy()[0][:,1],xyz.cpu().numpy()[0][:,2])
     xyz1=chuli(xyz.cpu().numpy()[1][:,0],xyz.cpu().numpy()[1][:,1],xyz.cpu().numpy()[1][:,2])
     xyz2=chuli(xyz.cpu().numpy()[2][:,0],xyz.cpu().numpy()[2][:,1],xyz.cpu().numpy()[2][:,2])
     xyz3=chuli(xyz.cpu().numpy()[3][:,0],xyz.cpu().numpy()[3][:,1],xyz.cpu().numpy()[3][:,2])
     xyz4=chuli(xyz.cpu().numpy()[4][:,0],xyz.cpu().numpy()[4][:,1],xyz.cpu().numpy()[4][:,2])
     xyz5=chuli(xyz.cpu().numpy()[5][:,0],xyz.cpu().numpy()[5][:,1],xyz.cpu().numpy()[5][:,2])
     xyz6=chuli(xyz.cpu().numpy()[6][:,0],xyz.cpu().numpy()[6][:,1],xyz.cpu().numpy()[6][:,2])
     xyz7=chuli(xyz.cpu().numpy()[7][:,0],xyz.cpu().numpy()[7][:,1],xyz.cpu().numpy()[7][:,2])
     xyz8=chuli(xyz.cpu().numpy()[8][:,0],xyz.cpu().numpy()[8][:,1],xyz.cpu().numpy()[8][:,2])
     xyz9=chuli(xyz.cpu().numpy()[9][:,0],xyz.cpu().numpy()[9][:,1],xyz.cpu().numpy()[9][:,2])
     xyz10=chuli(xyz.cpu().numpy()[10][:,0],xyz.cpu().numpy()[10][:,1],xyz.cpu().numpy()[10][:,2])
     xyz11=chuli(xyz.cpu().numpy()[11][:,0],xyz.cpu().numpy()[11][:,1],xyz.cpu().numpy()[11][:,2])
     xyz12=chuli(xyz.cpu().numpy()[12][:,0],xyz.cpu().numpy()[12][:,1],xyz.cpu().numpy()[12][:,2])
     xyz13=chuli(xyz.cpu().numpy()[13][:,0],xyz.cpu().numpy()[13][:,1],xyz.cpu().numpy()[13][:,2])
     xyz14=chuli(xyz.cpu().numpy()[14][:,0],xyz.cpu().numpy()[14][:,1],xyz.cpu().numpy()[14][:,2])
     xyz15=chuli(xyz.cpu().numpy()[15][:,0],xyz.cpu().numpy()[15][:,1],xyz.cpu().numpy()[15][:,2])
     torch.cuda.empty_cache()
     bbb=torch.cat((xyz0,xyz1,xyz2,xyz3,xyz4,xyz5,xyz6,xyz7,xyz8,xyz9,xyz10,xyz11,xyz12,xyz13,xyz14,xyz15), 0)
     aaa =  bbb.view(16, N, 3)
     torch.cuda.empty_cache()
     new_xyz = torch.as_tensor(new_xyz, dtype=torch.float64)
     aaa = torch.as_tensor(bbb, dtype=torch.float64)
     torch.cuda.empty_cache()
     sqrdists = square_distance(new_xyz,aaa )
    else:
     torch.cuda.empty_cache()
     sqrdists = square_distance(new_xyz,xyz )
   
    #sqrdists = square_distance(new_xyz,xyz )
    group_idx[sqrdists > radius** 2 ] = N-1
    #print(111)
    #print(group_idx)
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    #print(group_idx)
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    #print(mask)
    group_idx[mask] = group_first[mask]
    
    
    return group_idx
class SA_Layer(nn.Module):
    def __init__(self, channels):
        super(SA_Layer, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight 
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x_q = self.q_conv(x).permute(0, 2, 1) # b, n, c 
        x_k = self.k_conv(x)# b, c, n        
        x_v = self.v_conv(x)
        energy = torch.bmm(x_q, x_k) # b, n, n 
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        x_r = torch.bmm(x_v, attention) # b, c, n 
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
       
        return x
class Point_Transformer_partseg(nn.Module):
    def __init__(self):
        super(Point_Transformer_partseg, self).__init__()
        #self.part_num = part_num
        self.conv1 = nn.Conv1d(16, 128, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(128, 16, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(16)
        
        self.sa1 = SA_Layer(16)
        self.sa2 = SA_Layer(16)
        self.sa3 = SA_Layer(16)
        self.sa4 = SA_Layer(16)
        
        self.conv_fuse = nn.Sequential(nn.Conv1d(64, 128, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(128),
                                   nn.LeakyReLU(0.2))

        #self.label_conv = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   #nn.BatchNorm1d(64),
                                   #nn.LeakyReLU(0.2))

        self.convs1 = nn.Conv1d(264, 512, 1)
        self.dp1 = nn.Dropout(0.5)
        self.convs2 = nn.Conv1d(512, 256, 1)
        self.convs3 = nn.Conv1d(256, 16, 1)
        self.bns1 = nn.BatchNorm1d(512)
        self.bns2 = nn.BatchNorm1d(256)

        self.relu = nn.ReLU()

    def forward(self, x):
        
        batch_size, _, N = x.size()
        x = self.relu(self.bn1(self.conv1(x))) # B, D, N
        x = self.relu(self.bn2(self.conv2(x)))
        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        xqq = torch.cat((x1, x2, x3, x4), dim=1)
        
        xqqw = self.conv_fuse(xqq)
        #print(xqqw)
        x_max = torch.max(xqqw, 2)[1]
        #x_max =  x_max
        x_avg = torch.mean(xqqw, 2)[1]
        
        x_max_feature = x_max.view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)
        x_avg_feature = x_avg.view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)
        
       
             
        x_global_feature = torch.cat((x_max_feature.float(), x_avg_feature.float()),1)# 1024 + 64
        xqqw = torch.cat((xqqw.float(), x_global_feature.float()), 1) # 1024 * 3 + 64 
        xqqw = self.relu(self.bns1(self.convs1(xqqw)))
        xqqw = self.dp1(xqqw)
        xqqw = self.relu(self.bns2(self.convs2(xqqw)))
        xqqw = self.convs3(xqqw)
        #print(xqqw.size())
        #print(11111)
        return xqqw
model11 = Point_Transformer_partseg().cuda()
torch.cuda.empty_cache()
def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False,):
    
    B, N, C = xyz.shape
    S = npoint
    
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    torch.cuda.empty_cache()
    #points = torch.tensor(new_xyz, dtype=torch.float64)
    #xyz = torch.tensor(xyz, dtype=torch.float64)
   
    new_xyz = index_points(xyz, fps_idx)
    torch.cuda.empty_cache()
    idx = query_ball_point(radius, nsample, xyz, new_xyz) #wentiWEN問題所在
    torch.cuda.empty_cache()
    l3=[]
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    
    
     
    torch.cuda.empty_cache()
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    torch.cuda.empty_cache()

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
       
        return new_xyz, new_points, grouped_xyz, fps_idx
        
    else:
       # print(44444)
        #print(new_xyz.size())[16, 1024, 3])
        #print(new_points.size())([16, 1024, 32, 12])1111111111111111
        a,b,c = new_xyz.size()  
        if b == 166 :
           #print(new_xyz)
         new_xyz = model11(new_xyz)
           #print(new_xyz)
           #print(111111)
         torch.cuda.empty_cache()
         return new_xyz, new_points
        else:
           return new_xyz, new_points
def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
        
    else:
        new_points = grouped_xyz
        
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all
        
        #print(batch_label)
    
    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
            
           
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
            
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        #xyz=chuli(xyz.cpu().numpy()[0][:,0],xyz.cpu().numpy()[0][:,1],xyz.cpu().numpy()[0][:,2],points.cpu().numpy()[0][:,0],points.cpu().numpy()[0][:,1],points.cpu().numpy()[0][:,2])
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        #points = torch.tensor(new_xyz, dtype=torch.float64)
        #xyz = torch.tensor(xyz, dtype=torch.float64)
        
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points =  F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points


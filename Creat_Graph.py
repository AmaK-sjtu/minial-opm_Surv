import re, os
import cv2
import math
import random
import torch
#import resnet
import skimage.feature
import pdb
from PIL import Image
#from pyflann import *
from torch_geometric.data import Data
from collections import OrderedDict
import networkx as nx
from networks import *
from utils import *
import numpy as np
import pandas as pd
import torchvision.transforms.functional as F
import torch_geometric.data as data
import torch_geometric.utils as utils
import pdb
import torch_geometric
from model import CPCModel
from torchvision import transforms
import itertools
from tqdm import tqdm
device = torch.device('cuda:1')
model = MaxNet2(64,50,0.15, 256)
encoder = model.encoder.to(device)
data_dir=sys.args[1] ## the full patch for data
img_dir = os.path.join(data_dir, 'Image')
seg_dir =  os.path.join(data_dir,'segment')
save_dir = os.path.join(data_dir, 'NSCLC_graph')
pt_dir = os.path.join(save_dir, 'pt_bi')
graph_dir = os.path.join(save_dir, 'graphs')
fail_list = []
def get_cell_image(img, cx, cy, size=512):
    cx = 32 if cx < 32 else size-32 if cx > size-32 else cx
    cy = 32 if cy < 32 else size-32 if cy > size-32 else cy
    return img[cy-32:cy+32, cx-32:cx+32, :]
def get_cpc_features(cell):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    cell = transform(cell)
    cell = cell.unsqueeze(0)
    device = torch.device('cuda:1')
    feats = encoder(cell.to(device)).cpu().detach().numpy()[0]
    return feats
def get_cell_features(img, contour):  
    (cx, cy), (short_axis, long_axis), angle = cv2.fitEllipse(contour)
    cx, cy = int(cx), int(cy)   
    img_cell = get_cell_image(img, cx, cy)
    grey_region = cv2.cvtColor(img_cell, cv2.COLOR_RGB2GRAY)
    img_cell_grey = np.pad(grey_region, [(0, 64-grey_region.shape[0]), (0, 64-grey_region.shape[1])], mode = 'reflect') 
    eccentricity = math.sqrt(1-(short_axis/long_axis)**2)
    convex_hull = cv2.convexHull(contour)
    area, hull_area = cv2.contourArea(contour), cv2.contourArea(convex_hull)
    solidity = float(area)/hull_area
    arc_length = cv2.arcLength(contour, True)
    roundness = (arc_length/(2*math.pi))/(math.sqrt(area/math.pi))   
    out_matrix = skimage.feature.greycomatrix(img_cell_grey, [1], [0])
    dissimilarity = skimage.feature.greycoprops(out_matrix, 'dissimilarity')[0][0]
    homogeneity = skimage.feature.greycoprops(out_matrix, 'homogeneity')[0][0]
    energy = skimage.feature.greycoprops(out_matrix, 'energy')[0][0]
    ASM = skimage.feature.greycoprops(out_matrix, 'ASM')[0][0]  
    cpc_feats = get_cpc_features(img_cell)
    tmp=[]
    for i in range(cpc_feats.shape[1]):
        for j in range(cpc_feats.shape[2]):
            tmp.append(cpc_feats[0,i,j])
    x = [short_axis, long_axis, angle, area, arc_length, eccentricity, roundness, solidity, dissimilarity, homogeneity, energy, ASM]
    for i in tmp:
        x.append(i)
    return x,cx, cy
def seg2graph(img, contours):
    G = nx.Graph()   
    contours = [c for c in contours if c.shape[0] > 15]
    if len(contours)>2:
        for v, contour in enumerate(contours):
            if(cv2.contourArea(contour)>0):
                features, cx, cy = get_cell_features(img, contour)
                G.add_node(v, centroid = [cx, cy], x = features)
    if v < 5: return None
    return G
def from_networkx(G):
    r"""Converts a :obj:`networkx.Graph` or :obj:`networkx.DiGraph` to a
    :class:`torch_geometric.data.Data` instance.
    Args:
        G (networkx.Graph or networkx.DiGraph): A networkx graph.
    """
    G = G.to_directed() if not nx.is_directed(G) else G
    edge_index = torch.tensor(list(G.edges)).t().contiguous()
    keys = []
    keys += list(list(G.nodes(data=True))[0][1].keys())
    keys += list(list(G.edges(data=True))[0][2].keys())
    data = {key: [] for key in keys}
    for _, feat_dict in G.nodes(data=True):
        #print(feat_dict)
        for key, value in feat_dict.items():
            # keys are centorid,x 
            data[key].append(value)

    for _, _, feat_dict in G.edges(data=True):
        for key, value in feat_dict.items():
            data[key].append(value)
    for key, item in data.items():
        data[key] = item
    data['edge_index'] = edge_index
    data = torch_geometric.data.Data.from_dict(data)
    data.num_nodes = G.number_of_nodes()
    return data
def get_cell_image_og(img, cx, cy):
    if cx < 32 and cy < 32:
        return img[0: cy+32, 0:cx+32, :]
    elif cx < 32:
        return img[cy-32: cy+32, 0:cx+32, :] 
    elif cy < 32:
        return img[0: cy+32, cx-32:cx+32, :]
    else:
        return img[cy-32: cy+32, cx-32:cx+32, :]
    
def my_transform(img):
    img = F.to_tensor(img)
    img = F.normalize(img, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    return img

def get_cpc_features_og(cell):
    cell_R = np.squeeze(cell[:, :, 2:3])
    cell_G = np.squeeze(cell[:, :, 1:2])
    cell_B = np.squeeze(cell[:, :, 0:1])
    cell_R = np.pad(cell_R, [(0, 64-cell_R.shape[0]), (0, 64-cell_R.shape[1])], mode = 'constant')
    cell_G = np.pad(cell_G, [(0, 64-cell_G.shape[0]), (0, 64-cell_G.shape[1])], mode = 'constant')
    cell_B = np.pad(cell_B, [(0, 64-cell_B.shape[0]), (0, 64-cell_B.shape[1])], mode = 'constant')
    cell = np.stack((cell_R, cell_B, cell_G))  
    cell = np.transpose(cell, (1, 2, 0))
    cell = my_transform(cell)
    cell = cell.unsqueeze(0) 
    device =torch.device('cuda:1')    
    feats = encoder(cell.to(device)).cpu().detach().numpy()
    return feats

for img_fname in tqdm( os.listdir(seg_dir)):
    img = np.array(Image.open(os.path.join(img_dir, img_fname)))
    seg = np.array(Image.open(os.path.join(seg_dir, img_fname)))
    ret, binary = cv2.threshold(seg, 127, 255, cv2.THRESH_BINARY) 
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) < 2: continue 
    G = seg2graph(img, contours)
    if G is None: 
        fail_list.append(img_fname)
        continue
    centroids = []
    for u, attrib in G.nodes(data=True):
        centroids.append(attrib['centroid'])
    
    cell_centroids = np.array(centroids).astype(np.float64)
    dataset = cell_centroids
    start = None        
    for idx, attrib in list(G.nodes(data=True)):
        start = idx
        flann = FLANN()
        testset = np.array([attrib['centroid']]).astype(np.float64)
        results, dists = flann.nn(dataset, testset, num_neighbors=6, algorithm = 'kmeans', branching = 32, iterations = 100, checks = 16)
        results, dists = results[0], dists[0]
        nns_fin = []
        for i in range(1, len(results)):
            G.add_edge(idx, results[i], weight = dists[i])
            nns_fin.append(results[i])
    G = G.subgraph(max(nx.connected_components(G), key=len))
    img2=cv2.drawContours(img, contours, -1, (0,255,0), 2)
    for n, nbrs in G.adjacency():
        for nbr, eattr in nbrs.items():
            if len(G.nodes[nbr])>0:
                if len(G.nodes[n])>0:
                    img2=cv2.line(img2, tuple(G.nodes[n]['centroid']),  tuple(G.nodes[nbr]['centroid']), (0, 0, 255), 2)
    Image.fromarray(img2).save(os.path.join(graph_dir, img_fname))   
    G = from_networkx(G)
    edge_attr_long = (torch.tensor(G.weight).unsqueeze(1)).type(torch.LongTensor)
    G.edge_attr = edge_attr_long    
    edge_index_long = G['edge_index'].type(torch.LongTensor)
    G.edge_index = edge_index_long  
    a=len(G['x'])
    b=G.edge_index.max().data.item()
    x_float = torch.tensor(G['x'])
    if a==b+1:
        G.x = x_float
        G['weight'] = None
        G['nn'] = None
        torch.save(G, os.path.join(pt_dir, img_fname[:-4]+'.pt'))

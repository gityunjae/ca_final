# -*- coding: utf-8 -*-
import os
import pickle
import json
import struct
from typing import List

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"]="1" #"0,1,2,3"

import random as rnd
from collections import Counter, defaultdict
from implicit.nearest_neighbours import bm25_weight
from scipy import sparse
from lightfm import LightFM

import numpy as np
import implicit
import tqdm
import glob
import metrics
from multiprocessing import Pool

rnd.seed(0)

with open('data/song_meta.json', 'r') as json_file:
    song_info = json.load(json_file)
with open('data/train.json', 'r') as json_file:
    ply_info = json.load(json_file)

def save(keys, feats, out_fname):
    ''' Method used to save a list of element ids and the corresponing vectos
    '''
    feats = np.array(feats, dtype=np.float32)
    with open(out_fname + '.tmp', 'wb') as fout:
        fout.write(b' '.join([k.encode() for k in keys]))
        fout.write(b'\n')
        R, C = feats.shape
        fout.write(struct.pack('qq', *(R, C)))
        fout.write(feats.tostring())
    os.rename(out_fname + '.tmp', out_fname)


def load_feats(feat_fname, meta_only=False, nrz=False):
    ''' Method used to load element ids and the corresponing vectos
    '''
    with open(feat_fname, 'rb') as fin:
        keys = fin.readline().strip().split()
        R, C = struct.unpack('qq', fin.read(16))
        if meta_only:
            return keys, (R, C)
        feat = np.fromstring(fin.read(), count=R * C, dtype=np.float32)
        feat = feat.reshape((R, C))
        if nrz:
            feat = feat / np.sqrt((feat ** 2).sum(-1) + 1e-8)[..., np.newaxis]
    return keys, feat


latents_files = {'1': 7, '5': 1, '8': 1}


def song_id_to_info(sid: str) -> dict:
    info = dict()
    for j in song_info:
        if str(j["id"]) == str(sid):
            info = dict({"id": j["id"], "album_name": j["album_name"], "songs": j["song_name"],
                         "artist": j["artist_name_basket"]})
        else:
            pass
    return info


def get_songs(pid: str) -> dict:
    info = dict()
    for p in ply_info:
        if str(p["id"]) == str(pid):
            info = dict({"id": p["id"], "plylst_title": p["plylst_title"], "songs": p["songs"], "tags": p["tags"]})
        else:
            pass
    return info


if __name__ == '__main__':

    N = 10
    # N = 8
    dims = "300"
    #model_folder = 'models_split'
    model_folder = 'orig_models_split'
    model_folder2 = 'models_split(50ep)'
    for split in ['train', '8', '5', '1']:
        # We first load all data for the current split
        tracks_ids = json.load(open(os.path.join(model_folder, 'track_ids_{}.json'.format(split)), 'r'))
        playlists_ids = json.load(open(os.path.join(model_folder, 'playlists_ids_{}.json'.format(split)), 'r'))

        # cf_train 결과물
        item_features_file = os.path.join(model_folder, 'cf_item_{}_{}.feats'.format(dims, split))
        test_ids, track_orig_vects = load_feats(item_features_file)

        if split != 'train':
            pred_test_ids = []
            pred_vecs = []
            pred_test_ids2, pred_vecs2 = [], []
            for i in range(latents_files[split]):
                curr_test_ids, curr_orig_vecs = load_feats(os.path.join(model_folder,
                                                                        "test_pred_{}_{}.npy".format(split, i*5024)))

                pred_test_ids += curr_test_ids
                pred_vecs.append(curr_orig_vecs)
                curr_test_ids2, curr_orig_vecs2 = load_feats(os.path.join(model_folder2,
                                                                        "test_pred_{}_{}.npy".format(split, i * 5024)))

                pred_test_ids2 += curr_test_ids2
                pred_vecs2.append(curr_orig_vecs2)
            # pred_test_ids, pred_vecs 는 autotaggin한 결
            pred_vecs = np.vstack(curr_orig_vecs)
            pred_vecs2 = np.vstack(curr_orig_vecs2)

            # train을 제외한 8,5,1 에 대한 ply ids
            playlists_test = json.load(open(os.path.join(model_folder, 'test_cf_playlists_{}.json'.format(split)), 'r'))
            #playlists_test = json.load(open(os.path.join(model_folder, 'test_playlists_{}.json'.format(split)), 'r'))

            # The first 81219 items are the ones used to train the model, we want to evaluate on the rest of the items
            #test_ids = test_ids[81219:]
            #track_orig_vects = track_orig_vects[81219:]
            test_ids = test_ids[24187:]
            track_orig_vects = track_orig_vects[24187:]

        else:
            playlists_test = json.load(open(os.path.join(model_folder, 'test_playlists_{}.json'.format(split)), 'r'))
            pred_test_ids, pred_vecs = load_feats(os.path.join(model_folder, 'test_pred.npy'))
            playlists_test2 = json.load(open(os.path.join(model_folder2, 'test_playlists_{}.json'.format(split)), 'r'))
            pred_test_ids2, pred_vecs2 = load_feats(os.path.join(model_folder2, 'test_pred.npy'))

        dict_test_ids = {i.decode():1 for i in pred_test_ids}
        track_orig_vects = track_orig_vects[[i for i,x in enumerate(test_ids) if x.decode() in dict_test_ids]]
        test_ids = [x for x in test_ids if x.decode() in dict_test_ids]
        dict_test_ids2 = {i.decode(): 1 for i in pred_test_ids2}
        track_orig_vects2 = track_orig_vects[[i for i, x in enumerate(test_ids) if x.decode() in dict_test_ids2]]
        test_ids2 = [x for x in test_ids if x.decode() in dict_test_ids2]


        # Load the latent representations of playlists to make the predictions 
        train_features_file = os.path.join(model_folder, 'cf_playlist_{}_{}.feats'.format(dims, 'train'))
        train_playlists_ids, train_playlists_vects = load_feats(train_features_file)
        # This are the latent representations that are used to compute the Upper Bound
        user_features_file = os.path.join(model_folder, 'cf_playlist_{}_{}.feats'.format(dims, split))
        playlists_ids, playlists_vects = load_feats(user_features_file)

        inv_dict_id = {i:k.decode() for i,k in enumerate(test_ids)}
        inv_pred_id = {i:k.decode() for i,k in enumerate(pred_test_ids)}
        inv_pred_id2 = {i: k.decode() for i, k in enumerate(pred_test_ids2)}

        def evaluate(pos):
            if playlists_ids[pos].decode() not in playlists_test:
                return [],[],[],[],[]
            rets_pred = []
            plyid = playlists_ids[pos].decode()
            gt = playlists_test[playlists_ids[pos].decode()]
            num_vals = len(gt)
            if num_vals ==0:
                    return [],[],[],[],[]
            y_orig = playlists_vects[pos].dot(track_orig_vects.T)
            y_pred = train_playlists_vects[pos].dot(pred_vecs.T)
            y_pred2 = train_playlists_vects[pos].dot(pred_vecs2.T)
            topn_orig = np.argsort(-y_orig)[:N]
            topn_pred = np.argsort(-y_pred)[:N*3]
            topn_pred2 = np.argsort(-y_pred2)[:N*3]
            rets_orig = [int(inv_dict_id[t]) for t in topn_orig]
            rets_pred = [int(inv_pred_id[t]) for t in topn_pred]
            rets_pred2 = [int(inv_pred_id2[t]) for t in topn_pred2]
            return rets_orig, rets_pred, rets_pred2, gt, plyid

        pool = Pool(40)
        for i in range(int(len(playlists_ids)/1000)):
            results = pool.map(evaluate, range(i*1000, (i+1)*1000))
            e = 0
            for rets_orig, rets_pred, rets_pred2, gt, ply in results:
                if len(gt) > 0:
                    if e % 999 == 0:
                        print("ply")
                        print(ply)
                        ss = get_songs(ply)
                        print("id: ", end='')
                        print(ss["id"], end='')
                        print(", title: ", end='')
                        print(ss["plylst_title"], end='')
                        print(", tags: ", end='')
                        print(ss["tags"])
                        print("real")
                        # print(ss)
                        for v in ss["songs"]:
                            print(song_id_to_info(v))
                        print("orig_cf")
                        for v in rets_orig:
                            print(song_id_to_info(v))
                        print("pred_vgg")
                        for v in rets_pred:
                            print(song_id_to_info(v))
                        print("pred_ss")
                        for v in rets_pred2:
                            print(song_id_to_info(v))
                e += 1

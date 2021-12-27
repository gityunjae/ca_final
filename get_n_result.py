# -*- coding: utf-8 -*-
import os
import json
import struct
from pathlib import Path

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
    model_folder = 'final_orig_models_split'
    ids = [26777, 152444, 68360, 54333, 134799, 7723, 17267, 76793, 84101, 138008, 91299, 133215, 93955, 99882, 66767,
           67974, 149257, 147606, 95781, 54333, 14855, 96692, 38909, 132207, 120225, 114498, 14738, 87595, 70740, 87414,
           21732, 147034, 29382, 53957, 108010, 66350, 64800, 127734, 71192, 98525, 52927, 146256, 48538, 11579, 2775,
           17335, 77260, 151812, 136370, 70392, 118028, 109017, 95781, 68360, 141296, 128988, 83298, 7723, 17267,
           114498, 23317, 87595, 70740, 87414, 94372, 40222, 133719, 14833, 60423, 21732, 116718, 83129, 29382, 108010,
           35723, 56219, 133215, 64800, 361, 93955, 127734, 48898, 71192, 151821, 98525, 52927, 11313, 146256, 48538,
           99882, 143996, 21452, 25449, 120477, 22137, 11579, 17335, 7486, 70613, 100105, 55659, 17520, 104560, 59911,
           2378, 89186, 94831, 135665, 70392, 118028, 147606, 128988, 24015, 7723, 96692, 30366, 132207, 107321, 120225,
           14738, 3541, 70740, 87414, 86710, 34546, 40222, 11382, 21732, 18153, 94683, 83129, 29382, 91299, 48421,
           35723, 66350, 64800, 132507, 361, 66239, 93955, 127734, 151821, 98525, 52927, 38559, 146256, 48538, 99882,
           143996, 21452, 11943, 21315, 22137, 11579, 2775, 7486, 100105, 55659, 40007, 17520, 151812, 90084, 85799,
           59911, 34283, 2378, 135665, 93845, 147606]

    reEx = []
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
            # pred_test_ids, pred_vecs 는 autotaggin한 결
            pred_vecs = np.vstack(curr_orig_vecs)

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

        dict_test_ids = {i.decode():1 for i in pred_test_ids}
        track_orig_vects = track_orig_vects[[i for i,x in enumerate(test_ids) if x.decode() in dict_test_ids]]
        test_ids = [x for x in test_ids if x.decode() in dict_test_ids]


        # Load the latent representations of playlists to make the predictions 
        train_features_file = os.path.join(model_folder, 'cf_playlist_{}_{}.feats'.format(dims, 'train'))
        train_playlists_ids, train_playlists_vects = load_feats(train_features_file)
        # This are the latent representations that are used to compute the Upper Bound
        user_features_file = os.path.join(model_folder, 'cf_playlist_{}_{}.feats'.format(dims, split))
        playlists_ids, playlists_vects = load_feats(user_features_file)

        inv_dict_id = {i:k.decode() for i,k in enumerate(test_ids)}
        inv_pred_id = {i:k.decode() for i,k in enumerate(pred_test_ids)}

        def evaluate(pos):
            if playlists_ids[pos].decode() not in playlists_test:
                return [],[],[],[]
            rets_pred = []
            plyid = playlists_ids[pos].decode()
            gt = playlists_test[playlists_ids[pos].decode()]
            num_vals = len(gt)
            if num_vals ==0:
                    return [],[],[],[]
            y_orig = playlists_vects[pos].dot(track_orig_vects.T)
            y_pred = train_playlists_vects[pos].dot(pred_vecs.T)
            topn_orig = np.argsort(-y_orig)[:N]
            topn_pred = np.argsort(-y_pred)[:N*3]
            rets_orig = [int(inv_dict_id[t]) for t in topn_orig]
            rets_pred = [int(inv_pred_id[t]) for t in topn_pred]
            return rets_orig, rets_pred, gt, plyid

        pool = Pool(40)
        for i in range(int(len(playlists_ids)/1000)):
            results = pool.map(evaluate, range(i*1000, (i+1)*1000))
            for rets_orig, rets_pred, gt, ply in results:
                if len(gt) > 0:
                    temp = dict()
                    temp["ply"] = ply
                    ss = get_songs(ply)
                    temp["ply_id"] = ss["id"]
                    if temp["ply_id"] in ids:
                        temp["ply_title"] = ss["plylst_title"]
                        temp["ply_tags"] = ss["tags"]
                        lgt = []
                        for v in gt:
                            lgt.append(song_id_to_info(v))
                        temp["gt"] = lgt
                        real_s = []
                        for v in ss["songs"]:
                            real_s.append(song_id_to_info(v))
                        temp["real_songs"] = real_s
                        cf_s = []
                        for v in rets_orig:
                            cf_s.append(song_id_to_info(v))
                        temp["cf_preds"] = cf_s
                        vgg_s = []
                        for v in rets_pred:
                            vgg_s.append(song_id_to_info(v))
                        temp["vgg_pred"] = vgg_s
                        reEx.append(temp)
                        print(temp)


    proc_wp = Path('./results_small.json').open('w+', encoding='UTF-8')
    proc_wp.write(json.dumps(reEx))

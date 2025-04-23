import os

import io
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from scipy.interpolate import griddata
import yaml
import time
import json
from collections import OrderedDict
from networks_old.transformer import LGUnet_all
from networks.LGUnet_all import LGUnet_all_1
from nf_model.vae import VAE_lr
from petrel_client.client import Client
from torch.utils.tensorboard import SummaryWriter
import torch.utils.checkpoint as cp
from utils.metrics import Metrics
import torch.nn.functional as F
from torch_harmonics import *
import xarray as xr
from xspharm import xspharm
import argparse

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_time',         type = str,     default = "2018-01-01 00:00:00", )
    parser.add_argument('--end_time',           type = str,     default = "2018-01-02 00:00:00", )
    parser.add_argument('--coeff_dir',          type = str,     default = "dataset/bq_info_lr/", )
    parser.add_argument('--flow_model_dir',     type = str,     default = "world_size16-model-37years-stride1", )
    parser.add_argument('--forecast_model_dir', type = str,     default = "model_0.25degree", )
    parser.add_argument('--da_mode',            type = str,     default = "free_run", )
    parser.add_argument('--da_win',             type = int,     default = 6,     )
    parser.add_argument('--interp_dim',         type = int,     default = 40,    )
    parser.add_argument('--init_lag',           type = int,     default = 8,     )
    parser.add_argument('--init_tp',            type = int,     default = 0,     ) 
    parser.add_argument('--Nit',                type = int,     default = 8,     )
    parser.add_argument('--obs_std',            type = float,   default = 0.001, ) 
    parser.add_argument('--obs_coeff',          type = float,   default = 1.0,     )
    parser.add_argument('--filter_coeff',       type = float,   default = 0.5,     )
    parser.add_argument('--obs_type',           type = str,     default = "random_015", )  
    parser.add_argument('--prefix',             type = str, )  
    parser.add_argument('--q_type',             type = int,     default = 0,     ) 
    parser.add_argument('--scale_factor',       type = float,   default = 1.0, ) 
    parser.add_argument('--save_interval',      type = int,     default = 5,     ) 
    parser.add_argument('--save_field',         action = "store_true") 
    parser.add_argument('--save_gt',            action = "store_true") 
    parser.add_argument('--save_obs',           action = "store_true") 
    parser.add_argument('--forecast_eval',      action = "store_true") 
    parser.add_argument('--use_eval',           action = "store_true") 
    parser.add_argument('--obs_from_numpy',     type = str,     default = None) 
    parser.add_argument('--vae_ckpt',           type = str,     default = None) 
    parser.add_argument('--param_str',          type = str,    default = None) 
    parser.add_argument('--modify_tp',          type = int,    default = 0) 

    args = parser.parse_args()
    return args

class obs_interpolater:
    def __init__(self, dim_in=13, dim_out=40):
        self.dim_in  = dim_in
        self.dim_out = dim_out
        self.device = "cuda"
        self.height_level = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
        self.height_level_new = np.round(np.exp(np.linspace(3.91202301, 6.90775528, dim_out)))
        self.interp = self.get_interp()
        self.interp_inv = self.get_interp_inv()

    def get_interp(self,):
        linear_t = torch.zeros(self.dim_out, self.dim_in).cuda()
        for i in range(len(self.height_level_new)):
            for j in range(len(self.height_level)):
                if self.height_level_new[i] == self.height_level[j]:
                    linear_t[i, j] = 1
                elif self.height_level_new[i] > self.height_level[j] and self.height_level_new[i] < self.height_level[j+1]:
                    linear_t[i, j]   = (np.log(self.height_level[j+1]) - np.log(self.height_level_new[i])) / (np.log(self.height_level[j+1]) - np.log(self.height_level[j]))
                    linear_t[i, j+1] = (np.log(self.height_level_new[i]) - np.log(self.height_level[j])) / (np.log(self.height_level[j+1]) - np.log(self.height_level[j]))
    
        return linear_t

    def get_interp_inv(self,):
        linear_t = torch.zeros(self.dim_in, self.dim_out).cuda()
        for i in range(len(self.height_level)):
            for j in range(len(self.height_level_new)):
                if self.height_level[i] == self.height_level_new[j]:
                    linear_t[i, j] = 1
                elif self.height_level[i] > self.height_level_new[j] and self.height_level[i] < self.height_level_new[j+1]:
                    linear_t[i, j]   = (np.log(self.height_level_new[j+1]) - np.log(self.height_level[i])) / (np.log(self.height_level_new[j+1]) - np.log(self.height_level_new[j]))
                    linear_t[i, j+1] = (np.log(self.height_level[i]) - np.log(self.height_level_new[j])) / (np.log(self.height_level_new[j+1]) - np.log(self.height_level_new[j]))

        return linear_t


class data_reader:
    def __init__(self, obs_type, obs_std, model_std, da_win, cycle_time, step_int_time, obs_interp, obs_from_numpy=False, modify_tp=0):
        self.client = Client(conf_path="~/petreloss.conf")
        self.device = "cuda"
        self.obs_type = obs_type
        self.da_win   = da_win
        self.cycle_time = cycle_time
        self.step_int_time = step_int_time
        # if not obs_type[:4] == "real":
        obs_var_norm = torch.zeros(69, 721, 1440) + obs_std**2
        self.std_compensation = 40 * torch.Tensor([0.18955279, 0.22173745, 0.03315084, 0.08258388, 
        0.03021586, 0.0194484 , 0.01700376, 0.01931592, 0.02327741, 0.02647366, 0.02925515, 0.0304862 , 0.03300306, 0.03865351, 0.05609745, 0.0682424 , 0.07762259, 
        0.50658824, 0.29907974, 0.22097995, 0.22990653, 0.26931248, 0.27226337, 0.26211415, 0.24042704, 0.20803592, 0.18460007, 0.12343913, 0.06593712, 0.04856134, 
        0.11308974, 0.11406155, 0.10717956, 0.12138538, 0.14543332, 0.16263002, 0.17114112, 0.16359221, 0.1600293 , 0.16136173, 0.17905815, 0.19142863, 0.18638292, 
        0.13128242, 0.1593278, 0.16516368, 0.17795471, 0.19510655, 0.20854117, 0.21904777, 0.21593404, 0.21397153, 0.21613599, 0.23249907, 0.23790329, 0.21999044, 
        0.06977215, 0.03924686, 0.06015565, 0.11465897, 0.09490499, 0.06113996, 0.05008726, 0.04878271, 0.04601997, 0.04151259, 0.04477754, 0.04275933, 0.03838996])
        # self.obs_var = obs_var_norm * self.std_compensation.reshape(-1, 1, 1)**2 * model_std.reshape(-1, 1, 1)**2
        self.obs_var = obs_var_norm * model_std.reshape(-1, 1, 1)**2
        if modify_tp == 1:
            self.obs_var[56:] /= 4
        elif modify_tp == 2:
            self.obs_var[56:] /= 16
            self.obs_var[2] /= 16
        elif modify_tp == 3:
            self.obs_var[56:] /= 16
            self.obs_var[2] /= 16
            self.obs_var[30:56] /= 16
        elif modify_tp == 4:
            self.obs_var[56:] /= 16
            self.obs_var[2] /= 16
            self.obs_var[17:30] /= 4
        # if obs_type[:4] == "real":
        self.obs_interp = obs_interp
        self.obs_from_numpy = obs_from_numpy
        self.mean_layer = np.array([-0.14186215714480854, 0.22575792335029873, 278.7854495405721, 100980.83590625007, 199832.31609374992, 157706.1917968749, 132973.8087890624, 115011.55044921875, 100822.13164062506, 88999.83613281258, 69620.0044531249, 53826.54542968748, 40425.96180664062, 28769.254521484374, 13687.02337158203, 7002.870792236329, 777.5631800842285, 2.8248029025235157e-06, 2.557213611567022e-06, 4.689598504228342e-06, 1.7365863168379306e-05, 5.37612270545651e-05, 0.00012106754767955863, 0.0003586592462670523, 0.0007819174298492726, 0.0014082587775192225, 0.002245682779466732, 0.004328316930914292, 0.005698622210184111, 0.006659231842495503, 4.44909584343433, 10.046632840633391, 14.321160042285918, 15.298378415107727, 14.48938421010971, 12.895844810009004, 9.628437678813944, 7.07798705458641, 5.110536641478544, 3.4704639464616776, 1.2827875773236155, 0.3961004569224316, -0.18604825597634778, 0.012106836824341376, 0.1010729405652091, 0.2678451650420902, 0.2956721917196408, 0.21001753183547414, 0.03872977272505523, -0.04722135595180817, 0.0007164070030103152, -0.022026948712546065, 0.0075308467486320295, 0.014846984493779027, -0.062323193841984835, -0.15797925526494516, 214.66564151763913, 210.3573041915893, 215.23375904083258, 219.73181056976318, 223.53410289764412, 228.6614455413818, 241.16466262817383, 251.74072200775146, 259.84156120300344, 265.99485839843743, 272.77368919372566, 275.3001181793211, 278.5929747772214])

        self.std_layer = np.array([5.610453475051704, 4.798220612223473, 21.32010786700973, 1336.2115992274876, 3755.2810557402927, 4357.588191568988, 5253.301115477269, 5540.73074484052, 5405.73040397736, 5020.194961603476, 4104.233456672573, 3299.702929930327, 2629.7201995715513, 2060.9872289877453, 1399.3410970050247, 1187.5419349409494, 1098.9952409939283, 1.1555282996146702e-07, 4.2315237954921815e-07, 3.1627283344500357e-06, 2.093742795871515e-05, 7.02963683704546e-05, 0.00016131853114827985, 0.00048331132466880735, 0.001023028433607086, 0.0016946778969914426, 0.0024928432426471183, 0.004184742037434761, 0.005201345241925773, 0.00611814321149996, 11.557361639969054, 11.884088705628045, 15.407016747306344, 17.286773058038722, 17.720698660431694, 17.078782531259524, 14.509924979003983, 12.215305549952125, 10.503871726997783, 9.286354460633103, 8.179197305830433, 7.93264239491015, 6.126056325796786, 8.417864770061094, 8.178248048405905, 9.998695230009567, 11.896325029659364, 13.360381609448558, 13.474533447403218, 11.44656476066317, 9.321096224035244, 7.835396470389893, 6.858187372121642, 6.186618416862026, 6.345356147017278, 5.23175612906023, 9.495652698988557, 13.738672642636256, 9.090666595626503, 5.933385737657316, 7.389004707914384, 10.212310312072752, 12.773099916244078, 13.459313552230206, 13.858620163486986, 15.021590351519892, 16.00275340237577, 16.88523210573196, 18.59201174892538])

        self.std_layer_aug = [self.std_layer[:4]]
        for i in range(5):
            self.std_layer_aug.append(np.matmul(self.obs_interp.interp.cpu().numpy(), self.std_layer[4+13*i:17+13*i]))
        self.std_layer_aug = np.concatenate(self.std_layer_aug, 0)

        self.mean_layer_aug = [self.mean_layer[:4]]
        for i in range(5):
            self.mean_layer_aug.append(np.matmul(self.obs_interp.interp.cpu().numpy(), self.mean_layer[4+13*i:17+13*i]))
        self.mean_layer_aug = np.concatenate(self.mean_layer_aug, 0)

        self.max_layer_aug = self.mean_layer_aug + 3 * self.std_layer_aug
        self.min_layer_aug = self.mean_layer_aug - 3 * self.std_layer_aug

    def get_state(self, tstamp, data_dir="cluster3:s3://era5_np_float32"):
        state = []
        single_level_vnames = ['u10', 'v10', 't2m', 'msl']
        multi_level_vnames = ['z','q', 'u', 'v', 't']
        height_level = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
        for vname in single_level_vnames:
            file = os.path.join('single/'+str(tstamp.year), str(tstamp.to_datetime64()).split('.')[0]).replace('T', '/')
            url = f"{data_dir}/{file}-{vname}.npy"
            with io.BytesIO(self.client.get(url)) as f:
                state.append(np.load(f).reshape(1, 721, 1440))
        for vname in multi_level_vnames:
            file = os.path.join(str(tstamp.year), str(tstamp.to_datetime64()).split('.')[0]).replace('T', '/')
            for idx in range(13):
                height = height_level[idx]
                url = f"{data_dir}/{file}-{vname}-{height}.0.npy"
                with io.BytesIO(self.client.get(url)) as f:
                    state.append(np.load(f).reshape(1, 721, 1440))
        state = np.concatenate(state, 0)
        return torch.from_numpy(state)

    def read_json(self, tstamp, data_dir="cluster2:s3://global_data_assimilation_raw/Processed"):
        file = os.path.join(str(tstamp.year), str(tstamp.to_datetime64())[:13])
        url = f"{data_dir}/{file}.json"
        if self.client.get(url):
            with io.BytesIO(self.client.get(url)) as f:
                d = json.load(f)
        else:
            d = []
            print("no obs at time", tstamp)
        return d

    def read_numpy(self, tstamp, obs_from_numpy, data_dir="cluster2:s3://global_data_assimilation_raw/"):
        data_dir = data_dir + obs_from_numpy
        file = os.path.join(str(tstamp.year), str(tstamp.to_datetime64())[:13])
        url_obs  = f"{data_dir}/{file}-obs.npy"
        url_mask = f"{data_dir}/{file}-mask.npy"
        with io.BytesIO(self.client.get(url_obs)) as f:
            obs = np.load(f).astype(np.float32)
        with io.BytesIO(self.client.get(url_mask)) as f:
            mask = np.load(f).astype(np.float32)
        return torch.from_numpy(obs), torch.from_numpy(mask)

    def get_obs_mask(self, tstamp):
        if self.obs_type[:8] == "prepbufr":
            if not self.da_win == 6 and not self.da_win == 1:
                raise NotImplementedError("da win must equal six or one")
            H  = torch.zeros(self.da_win, 69, 721, 1440)
            height = np.array([75, 125, 175, 225, 275, 350, 450, 550, 650, 775, 887.5, 962.5])

            d = self.read_json(tstamp)
            for message in d:
                elem = d[message]
                if elem['position'][0] == None or elem['position'][1] == None or elem['position'][2] == None or elem['position'][3] == None:
                    continue
                lon = int(np.round(elem['position'][0] / 360 * 1440))
                if lon == 1440:
                    lon = 0
                lat = int(np.round((90 - elem['position'][1]) / 180 * 721))
                if lat == 721:
                    lat = 720
                h = int(np.sum((height - elem['position'][2]) <= 0))
                if self.da_win == 1:
                    if elem['position'][3] >= -0.5 and elem['position'][3] < 0.5:
                        t = 0
                    else:
                        continue
                else:
                    if elem['position'][3] >= -0.5 and elem['position'][3] < 0.5:
                        t = 0
                    elif elem['position'][3] >= 0.5 and elem['position'][3] < 1.5:
                        t = 1
                    elif elem['position'][3] >= 1.5 and elem['position'][3] < 2.5:
                        t = 2
                    elif elem['position'][3] >= 2.5:
                        t = 3
                    else:
                        continue
                if elem['value'][1]:
                    H[t, 4+h, lat, lon] = 1
                if elem['value'][2]:
                    H[t, 4+h+13, lat, lon] = 1
                if elem['value'][3]:
                    H[t, 4+h+26, lat, lon] = 1
                if elem['value'][4]:
                    H[t, 4+h+39, lat, lon] = 1
                if elem['value'][5]:
                    H[t, 4+h+52, lat, lon] = 1
                if elem['value'][7]:
                    H[t, 3, lat, lon] = 1

            if self.da_win > 3:
                d = self.read_json(tstamp + self.cycle_time)
                for message in d:
                    elem = d[message]
                    if elem['position'][0] == None or elem['position'][1] == None or elem['position'][2] == None or elem['position'][3] == None:
                        continue
                    lon = int(np.round(elem['position'][0] / 360 * 1440))
                    if lon == 1440:
                        lon = 0
                    lat = int(np.round((90 - elem['position'][1]) / 180 * 721))
                    if lat == 721:
                        lat = 720
                    h = int(np.sum((height - elem['position'][2]) <= 0))
                    if elem['position'][3] < -2.5:
                        t = 3
                    elif elem['position'][3] >= -2.5 and elem['position'][3] < -1.5:
                        t = 4
                    elif elem['position'][3] >= -1.5 and elem['position'][3] < -0.5:
                        t = 5
                    else:
                        continue
                    if elem['value'][1]:
                        H[t, 4+h, lat, lon] = 1
                    if elem['value'][2]:
                        H[t, 4+h+13, lat, lon] = 1
                    if elem['value'][3]:
                        H[t, 4+h+26, lat, lon] = 1
                    if elem['value'][4]:
                        H[t, 4+h+39, lat, lon] = 1
                    if elem['value'][5]:
                        H[t, 4+h+52, lat, lon] = 1
                    if elem['value'][7]:
                        H[t, 3, lat, lon] = 1

            H[:, 0] = H[:, 42]
            H[:, 1] = H[:, 55]
            H[:, 2] = H[:, 68]

        elif self.obs_type[:4] == "free":
            if len(self.obs_type) == 9:
                obs_amount = int(self.obs_type[5:9]) * 1000
            else:
                obs_amount = int(self.obs_type[5:10]) * 100
            print("obs amount:", obs_amount)
            matrix = np.zeros((721, 1440), dtype=int)
            indices = np.random.choice(matrix.size, obs_amount, replace=False)

            # 将这些索引位置的元素设置为 1
            np.put(matrix, indices, 1)

            x = np.stack([matrix] * 69)

            H  = torch.zeros(self.da_win, 69, 721, 1440)
            H_file = torch.from_numpy(x)
            H = H + H_file

        else:
            H  = torch.zeros(self.da_win, 69, 721, 1440)
            H_file = torch.from_numpy(np.load("dataset/mask_%s.npy"%self.obs_type)).float()
            H = H + H_file

        return H

    def get_real_obs(self, tstamp):
        if self.obs_from_numpy:
            print("reading from numpy")
            return self.read_numpy(tstamp, self.obs_from_numpy)

        if not self.da_win == 6 and not self.da_win == 1:
            raise NotImplementedError("da win must equal six")
        H   = torch.zeros(self.da_win, 4+5*self.obs_interp.dim_out, 721, 1440)
        cnt = torch.zeros(self.da_win, 4+5*self.obs_interp.dim_out, 721, 1440) + 1e-10
        obs = torch.zeros(self.da_win, 4+5*self.obs_interp.dim_out, 721, 1440)
        height = np.zeros(self.obs_interp.dim_out - 1)
        for i in range(len(height)):
            height[i] = np.sqrt(self.obs_interp.height_level_new[i] * self.obs_interp.height_level_new[i+1])

        def get_geopotential_coeff(idx):
            if idx == 0:
                return 61245
            elif idx <= 16:
                return 62000
            else:
                return 927.87 * idx + 47138.48 

        def get_temperature_coeff(idx):
            if idx <= 21:
                return 0
            else:
                return -25

        geopotential_interp_coeff = []
        temperature_interp_coeff = []
        for i in range(self.obs_interp.dim_out):
            geopotential_interp_coeff.append(get_geopotential_coeff(i))
            temperature_interp_coeff.append(get_temperature_coeff(i))

        def assign_value(t, layer, lat, lon, value):
            H[t, layer, lat, lon] = 1
            cnt[t, layer, lat, lon] += 1
            obs[t, layer, lat, lon] += value

        def assign_upper_value(elem_arr, pressure, obstype, h, t, lat, lon):
            for i in range(5):
                if elem_arr[i+1]:
                    layer = 4 + h + i * len(self.obs_interp.height_level_new)
                    value = elem_arr[i+1]
                    if i == 0:
                        value *= 9.8
                    elif i == 1:
                        value *= 1e-6
                    elif i == 4:
                        value += 273.15

                    if i == 0:
                        value += geopotential_interp_coeff[h] * (np.log(pressure) - np.log(self.obs_interp.height_level_new[h]))
                    elif i == 4:
                        value += temperature_interp_coeff[h] * (np.log(pressure) - np.log(self.obs_interp.height_level_new[h]))

                    assign_value(t, layer, lat, lon, value)
            
            if elem_arr[-1]:
                layer = 3
                value = elem_arr[-1] * 100
                assign_value(t, layer, lat, lon, value)

        def assign_surface_value(elem_arr, obstype, t, lat, lon):
            for i in range(3):
                if elem_arr[i+3]:
                    value = elem_arr[i+3]
                    if i == 2:
                        value += 273.15
                    assign_value(t, i, lat, lon, value)

        d = self.read_json(tstamp)

        for message in d:
            elem = d[message]
            if elem['position'][0] == None or elem['position'][1] == None or elem['position'][2] == None or elem['position'][3] == None:
                continue
            lon = int(np.round(elem['position'][0] / 360 * 1440))
            if lon == 1440:
                lon = 0
            lat = int(np.round((90 - elem['position'][1]) / 180 * 721))
            if lat == 721:
                lat = 720
            h = int(np.sum((height - elem['value'][0]) <= 0))

            if self.da_win == 1:
                if elem['position'][3] >= -0.5 and elem['position'][3] < 0.5:
                    t = 0
                else:
                    continue
            else:
                if elem['position'][3] >= -0.5 and elem['position'][3] < 0.5:
                    t = 0
                elif elem['position'][3] >= 0.5 and elem['position'][3] < 1.5:
                    t = 1
                elif elem['position'][3] >= 1.5 and elem['position'][3] < 2.5:
                    t = 2
                elif elem['position'][3] >= 2.5:
                    t = 3
                else:
                    continue

            assign_upper_value(elem['value'], elem['value'][0], elem['type'], h, t, lat, lon)
        
            if h == self.obs_interp.dim_out - 1:
                assign_surface_value(elem['value'], elem['type'], t, lat, lon)

        if self.da_win > 3:

            d = self.read_json(tstamp + self.cycle_time)

            for message in d:
                elem = d[message]
                if elem['position'][0] == None or elem['position'][1] == None or elem['position'][2] == None or elem['position'][3] == None:
                    continue
                lon = int(np.round(elem['position'][0] / 360 * 1440))
                if lon == 1440:
                    lon = 0
                lat = int(np.round((90 - elem['position'][1]) / 180 * 721))
                if lat == 721:
                    lat = 720
                h = int(np.sum((height - elem['value'][0]) <= 0))

                if elem['position'][3] < -2.5:
                    t = 3
                elif elem['position'][3] >= -2.5 and elem['position'][3] < -1.5:
                    t = 4
                elif elem['position'][3] >= -1.5 and elem['position'][3] < -0.5:
                    t = 5
                else:
                    continue
                    
                assign_upper_value(elem['value'], elem['value'][0], elem['type'], h, t, lat, lon)
                    
                if h == self.obs_interp.dim_out - 1:
                    assign_surface_value(elem['value'], elem['type'], t, lat, lon)

        obs = obs / cnt

        return obs, H

    def get_obs_gt(self, current_time):
        state = [self.get_state(current_time)]
        for i in range(self.da_win - 1):
            current_time += self.step_int_time
            state.append(self.get_state(current_time))
        gt  = torch.stack(state, 0)
        if not self.obs_type[:4] == "real":
            obs = gt # gt = 57315  #+ torch.sqrt(self.obs_var) * torch.randn(self.da_win, 69, 721, 1440)
            return obs, gt
        else:
            return gt
    

class cyclic_4dvar:
    def __init__(self, args):
        self.device     = "cuda"
        self.start_time    = pd.Timestamp(args.start_time)
        self.end_time      = pd.Timestamp(args.end_time)
        self.cycle_time    = pd.Timedelta('6H')
        self.step_int_time = pd.Timedelta('1H')
        self.da_mode       = args.da_mode
        self.q_type        = args.q_type
        self.da_win        = args.da_win
        self.nlon          = 1440
        self.nlat          = 721
        self.hpad          = 112
        self.vname_list    = ['z', 'q', 'u', 'v', 't']
        self.geoheight_list = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
        self.fullname = ['u10', 'v10', 't2m', 'mslp']
        for vname in self.vname_list:
            for geoheight in self.geoheight_list:
                self.fullname.append(vname + str(geoheight))
        self.nlev          = len(self.geoheight_list)
        self.nchannel      = len(self.fullname)
        self.Nit           = args.Nit

        self.scale_factor   = args.scale_factor
        self.obs_coeff      = args.obs_coeff
        self.filter_coeff   = args.filter_coeff
        self.b_matrix       = self.init_b_matrix(args.coeff_dir)
        self.q_matrix       = self.init_q_matrix(args.coeff_dir)  
        self.flow_model     = self.init_model_flow(args.flow_model_dir)
        self.forecast_model = self.init_model_forecast(args.forecast_model_dir)
        self.model_mean, self.model_std, self.model_mean_gpu, self.model_std_gpu = self.get_model_mean_std()

        self.init_tp        = args.init_tp
        self.init_lag = args.init_lag
        self.obs_std  = args.obs_std
        self.obs_type = args.obs_type
        # if self.obs_type[:4] == "real":
        self.interp_dim    = args.interp_dim
        self.obs_interp    = obs_interpolater(self.nlev, self.interp_dim)
        self.mask_eval     = torch.Tensor(np.load("dataset/mask_eval1.npy")).unsqueeze(0).unsqueeze(0) + torch.zeros(self.da_win, 204, 721, 1440)
        self.use_eval      = args.use_eval
        print("use eval:", self.use_eval, args.use_eval)

        self.name  =  "%s_stdmodify%d_%s_std%.3f_win%d_lag%d_filter%.2f_sc%.2f_Nit%d_%s"%(args.prefix, args.modify_tp, args.obs_type, args.obs_std, args.da_win, args.init_lag, args.filter_coeff, args.scale_factor, args.Nit, args.end_time)
        print(self.name)

        self.save_field    = args.save_field
        self.forecast_eval = args.forecast_eval
        self.save_interval = args.save_interval
        self.save_gt       = args.save_gt
        self.save_obs      = args.save_obs
        self.metric        = Metrics()

        self.init_file_dir()

        self.data_reader = data_reader(args.obs_type, args.obs_std, self.model_std, self.da_win, self.cycle_time, self.step_int_time, self.obs_interp, args.obs_from_numpy, args.modify_tp)
        self.metrics_list = {"bg_wrmse": [], "ana_wrmse": [], "bg_mse": [], "ana_mse": [], "bg_bias": [], "ana_bias": [], "error_obs": []}
        self.forecast_wrmse = []
        self.current_time, self.xb = self.get_current_states()
        self.load_eval_ckpts()

        self.static_info   = self.get_static_info() ## for saving redundant calculations
        self.vae_ckpt = args.vae_ckpt
        self.vae = self.init_vae_model(args.param_str)

    def init_b_matrix(self, coeff_dir):
        len_scale = torch.from_numpy(np.load(os.path.join(coeff_dir, "len_scale.npy")) * self.scale_factor).float().to(self.device)
        reg_coeff = torch.from_numpy(np.load(os.path.join(coeff_dir, "reg_coeff.npy"))).float().to(self.device)
        std_sur   = torch.from_numpy(np.load(os.path.join(coeff_dir, "std_sur.npy"))).float().to(self.device)
        vert_eig_value = torch.from_numpy(np.load(os.path.join(coeff_dir, "vert_eig_value.npy"))).float().to(self.device)
        vert_eig_vec   = torch.from_numpy(np.load(os.path.join(coeff_dir, "vert_eig_vec.npy"))).float().to(self.device)
        return {"len_scale": len_scale, "reg_coeff": reg_coeff, "std_sur": std_sur, "vert_eig_value": vert_eig_value, "vert_eig_vec": vert_eig_vec}

    def init_q_matrix(self, coeff_dir):
        if self.da_win == 1:
            return []

        if self.q_type == 0:
            q = []
            for i in range(1, self.da_win):
                q0 = torch.from_numpy(np.load(os.path.join(coeff_dir, "q%d.npy"%i)))
                q.append(torch.broadcast_to(torch.mean(q0, (1, 2), True), (self.nchannel, self.nlat, self.nlon)))
            q = torch.stack(q, 0)
            print("q", q[:, :, 100, 100])

        elif self.q_type == -1:
            q = torch.zeros(self.da_win-1, self.nchannel, self.nlat, self.nlon)

        elif self.q_type == 1:
            q = torch.broadcast_to(torch.from_numpy(np.load(os.path.join(coeff_dir, "new_q.npy"))).unsqueeze(2).unsqueeze(2)[:self.da_win-1], (self.da_win-1, self.nchannel, self.nlat, self.nlon))
            q = F.interpolate(q, size=(721, 1440))
            print("q", q[:, :, 100, 100])
            
        else:
            raise NotImplementedError("not implemented q type")
        return q

    def init_model_forecast(self, path):
        with open("output/model/%s/training_options.yaml"%(path), 'r') as cfg_file:
            cfg_params = yaml.load(cfg_file, Loader = yaml.FullLoader)
        model = LGUnet_all_1(**cfg_params["model"]["params"]["sub_model"]["lgunet_all"])
        checkpoint_dict = torch.load("output/model/%s/checkpoint_latest.pth"%(path))
        checkpoint_model = checkpoint_dict['model']['lgunet_all']
        new_state_dict = OrderedDict()
        for k, v in checkpoint_model.items():
            if "module" == k[:6]:
                name = k[7:]
            else:
                name = k
            if not name == "max_logvar" and not name == "min_logvar":
                new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        model.to(self.device)
        model.eval()
        return model

    def init_model_flow(self, path):
        with open("../fengwu-lite/output/model/%s/training_options.yaml"%(path), 'r') as cfg_file:
            cfg_params = yaml.load(cfg_file, Loader = yaml.FullLoader)
        model = LGUnet_all(**cfg_params["model"]["network_params"])
        checkpoint_dict = torch.load("../fengwu-lite/output/model/%s/checkpoint_best.pth"%(path))
        checkpoint_model = checkpoint_dict['model']
        new_state_dict = OrderedDict()
        for k, v in checkpoint_model.items():
            if "module" == k[:6]:
                name = k[7:]
            else:
                name = k
            if not name == "max_logvar" and not name == "min_logvar":
                new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        model.to(self.device)
        model.eval()
        return model

    def init_vae_model(self, param_str):
        model = VAE_lr(param_str).to(self.device)
        checkpoint_model = torch.load(self.vae_ckpt, map_location='cuda:0')
        new_state_dict = OrderedDict()
        for k, v in checkpoint_model.items():
            if "module" == k[:6]:
                name = k[7:]
            else:
                name = k
            if not name == "max_logvar" and not name == "min_logvar":
                new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        model.eval()
        return model

    def init_file_dir(self):
        os.makedirs("da_cycle_results/%s"%(self.name), exist_ok=True)

    def get_static_info(self):
        ### calculating horizontal factor
        x = np.linspace(-self.hpad, self.hpad, 2*self.hpad+1)
        y = np.linspace(-self.hpad, self.hpad, 2*self.hpad+1)
        xx, yy = np.meshgrid(x, y)

        nlat_s = 128
        nlon_s = 256

        sht = RealSHT(nlat_s, nlon_s, grid="equiangular").to(self.device)
        isht = InverseRealSHT(nlat_s, nlon_s, grid="equiangular").to(self.device)

        kernel = torch.zeros(self.nchannel, nlat_s, nlon_s).to(self.device)
        coeffs_kernel = []
        for layer in range(self.nchannel):
            for i in range(self.hpad):
                kernel[layer, i] = torch.exp(-i**2/(8*self.b_matrix["len_scale"][layer]**2))
            coeffs_kernel.append(sht(kernel[layer]))

        sph_scale = torch.Tensor(np.array(np.broadcast_to(np.arange(0, nlat_s).transpose(), (nlat_s+1, nlat_s)).transpose())).to(self.device)
        sph_scale = 2*np.pi*torch.sqrt(4*np.pi/(2*sph_scale+1))

        ### calculating R
        R = torch.zeros(self.da_win, self.nchannel, self.nlat, self.nlon).to(self.device)
        R[0] = self.data_reader.obs_var
        for i in range(self.da_win - 1):
            R[i+1] = self.data_reader.obs_var + self.q_matrix[i]

        print("R", R[:, :, 100, 100])

        return {"R": R, "sht": sht, "isht": isht, "coeffs_kernel": coeffs_kernel, "sph_scale": sph_scale}

    def get_model_mean_std(self):
        mean_layer = np.array([-0.14186215714480854, 0.22575792335029873, 278.7854495405721, 100980.83590625007, 199832.31609374992, 157706.1917968749, 132973.8087890624, 115011.55044921875, 100822.13164062506, 88999.83613281258, 69620.0044531249, 53826.54542968748, 40425.96180664062, 28769.254521484374, 13687.02337158203, 7002.870792236329, 777.5631800842285, 2.8248029025235157e-06, 2.557213611567022e-06, 4.689598504228342e-06, 1.7365863168379306e-05, 5.37612270545651e-05, 0.00012106754767955863, 0.0003586592462670523, 0.0007819174298492726, 0.0014082587775192225, 0.002245682779466732, 0.004328316930914292, 0.005698622210184111, 0.006659231842495503, 4.44909584343433, 10.046632840633391, 14.321160042285918, 15.298378415107727, 14.48938421010971, 12.895844810009004, 9.628437678813944, 7.07798705458641, 5.110536641478544, 3.4704639464616776, 1.2827875773236155, 0.3961004569224316, -0.18604825597634778, 0.012106836824341376, 0.1010729405652091, 0.2678451650420902, 0.2956721917196408, 0.21001753183547414, 0.03872977272505523, -0.04722135595180817, 0.0007164070030103152, -0.022026948712546065, 0.0075308467486320295, 0.014846984493779027, -0.062323193841984835, -0.15797925526494516, 214.66564151763913, 210.3573041915893, 215.23375904083258, 219.73181056976318, 223.53410289764412, 228.6614455413818, 241.16466262817383, 251.74072200775146, 259.84156120300344, 265.99485839843743, 272.77368919372566, 275.3001181793211, 278.5929747772214])

        std_layer = np.array([5.610453475051704, 4.798220612223473, 21.32010786700973, 1336.2115992274876, 3755.2810557402927, 4357.588191568988, 5253.301115477269, 5540.73074484052, 5405.73040397736, 5020.194961603476, 4104.233456672573, 3299.702929930327, 2629.7201995715513, 2060.9872289877453, 1399.3410970050247, 1187.5419349409494, 1098.9952409939283, 1.1555282996146702e-07, 4.2315237954921815e-07, 3.1627283344500357e-06, 2.093742795871515e-05, 7.02963683704546e-05, 0.00016131853114827985, 0.00048331132466880735, 0.001023028433607086, 0.0016946778969914426, 0.0024928432426471183, 0.004184742037434761, 0.005201345241925773, 0.00611814321149996, 11.557361639969054, 11.884088705628045, 15.407016747306344, 17.286773058038722, 17.720698660431694, 17.078782531259524, 14.509924979003983, 12.215305549952125, 10.503871726997783, 9.286354460633103, 8.179197305830433, 7.93264239491015, 6.126056325796786, 8.417864770061094, 8.178248048405905, 9.998695230009567, 11.896325029659364, 13.360381609448558, 13.474533447403218, 11.44656476066317, 9.321096224035244, 7.835396470389893, 6.858187372121642, 6.186618416862026, 6.345356147017278, 5.23175612906023, 9.495652698988557, 13.738672642636256, 9.090666595626503, 5.933385737657316, 7.389004707914384, 10.212310312072752, 12.773099916244078, 13.459313552230206, 13.858620163486986, 15.021590351519892, 16.00275340237577, 16.88523210573196, 18.59201174892538])

        mean_layer_gpu = torch.from_numpy(mean_layer).float().to(self.device)
        std_layer_gpu  = torch.from_numpy(std_layer).float().to(self.device)
        return mean_layer, std_layer, mean_layer_gpu, std_layer_gpu

    def get_initial_state(self):
        if self.init_tp == 0:
            x0 = self.data_reader.get_state(self.start_time - self.init_lag * pd.Timedelta('6H')).cuda()
            xb = self.integrate(x0, self.forecast_model, self.init_lag).cpu()
        elif self.init_tp == 1:
            xb = self.data_reader.get_state(self.start_time - self.init_lag * pd.Timedelta('6H'))
        elif self.init_tp == 2:
            xb = self.data_reader.get_state(self.start_time - 4 * 183 * pd.Timedelta('6H'))
        # x0 = self.data_reader.get_state(self.start_time - self.init_lag * pd.Timedelta('6H')).cuda()
        # xb = self.integrate(x0, self.forecast_model, self.init_lag).cpu()
        gt = self.data_reader.get_state(self.start_time)
        rmse = torch.sqrt(torch.mean((gt - xb)**2, (1, 2)))
        print("xb rmse per layer", rmse.numpy())
        mse  = torch.mean(((gt - xb) / self.model_std.reshape(-1, 1, 1))**2)
        print("xb mse: %.3g"%(mse))
        return xb

    def integrate(self, xa, model, step, interpolation=False, detach=True):
        za = (xa - self.model_mean_gpu.reshape(-1, 1, 1)) / self.model_std_gpu.reshape(-1, 1, 1)
        z = za.unsqueeze(0)

        if interpolation:
            z = F.interpolate(z, (128, 256))
    
        for i in range(step):
            z = model(z)[:, :self.nchannel]
            if detach:
                z = z.detach()

        if interpolation:
            z = F.interpolate(z, (721, 1440))

        return z.reshape(self.nchannel, self.nlat, self.nlon) * self.model_std_gpu.reshape(-1, 1, 1) + self.model_mean_gpu.reshape(-1, 1, 1)

    def get_current_states(self):
        if os.path.exists("da_cycle_results/%s/current_time.txt"%(self.name)):
            f = open("da_cycle_results/%s/current_time.txt"%self.name, "r")
            self.current_time = pd.Timestamp(f.read())
        else:
            self.current_time = self.start_time

        if os.path.exists("da_cycle_results/%s/xb.npy"%self.name):
            state = np.load("da_cycle_results/%s/xb.npy"%self.name)
            self.xb = torch.from_numpy(state)
        else:
            self.xb = self.get_initial_state()
        
        return self.current_time, self.xb

    def save_ckpt(self, finish=False, gt=None, obs=None):
        if not finish:
            np.save("da_cycle_results/%s/xb"%self.name, self.xb.cpu().numpy())
            with open("da_cycle_results/%s/current_time.txt"%self.name, 'w') as f:
                f.write(str(self.current_time))

    def save_eval_result(self, finish=False, gt=None, obs=None):
        for key in self.metrics_list:
            np.save("da_cycle_results/%s/%s"%(self.name, key), self.metrics_list[key])
        print("finish saving results")
        if self.forecast_eval:
            np.save("da_cycle_results/%s/forecast_wrmse"%(self.name), self.forecast_wrmse)
            print("finish saving forecasting evaluation results")

        if not finish:
            if self.save_field:
                np.save("da_cycle_results/%s/xb_%s"%(self.name, self.current_time), self.xb.detach().cpu().numpy())
                np.save("da_cycle_results/%s/xa_%s"%(self.name, self.current_time), self.xa.detach().cpu().numpy())
                print("finish saving intermediate fields")
            if self.save_gt:
                np.save("intermediate/ground_truth/gt_%s"%(self.current_time), gt.cpu().float().numpy())
                print("finish saving ground truth")
            if self.save_obs:
                np.save("intermediate/ground_truth/obs_%s"%(self.current_time), obs.cpu().float().numpy())
                print("finish saving observations")

    def load_eval_ckpts(self):   
        for key in self.metrics_list:
            if os.path.exists("da_cycle_results/%s/%s.npy"%(self.name, key)):
                self.metrics_list[key] = np.load("da_cycle_results/%s/%s.npy"%(self.name, key)).tolist()

    def get_R_matrix_from_gt(self, yo, gt_tensor):
        # if 1:
        #     gt_aug = []
        #     gt_aug.append(gt_tensor[:, :4])

        #     for i in range(5):
        #         mat = gt_tensor[:, 4+i*self.nlev:4+(i+1)*self.nlev]
        #         mat = F.linear(mat.transpose(1, 3), self.obs_interp.interp).transpose(1, 3)
        #         gt_aug.append(mat)

        #     gt_aug = torch.cat(gt_aug, 1)
        #     R = (gt_aug - yo)**2
        #     # for i in range(self.da_win - 1):
        #     #     R[i+1] = self.data_reader.obs_var + self.q_matrix[i]
        
        if 1:
            R = self.static_info["R"]
            R_aug = []
            R_aug.append(R[:, :4])

            for i in range(5):
                mat = R[:, 4+i*self.nlev:4+(i+1)*self.nlev]
                mat = F.linear(mat.transpose(1, 3), self.obs_interp.interp).transpose(1, 3)
                R_aug.append(mat)

            R = torch.cat(R_aug, 1)

        return R

    def get_obs_info(self):
        start_clock = time.time()
        if not self.obs_type[:4] == "real":
            yo, gt = self.data_reader.get_obs_gt(self.current_time)
            H = self.data_reader.get_obs_mask(self.current_time)
            R = self.static_info["R"]
        else:
            gt = self.data_reader.get_obs_gt(self.current_time)
            yo, H = self.data_reader.get_real_obs(self.current_time)
            for i in range(self.da_win):
                print("before filtering: obs[%d] amount = %.1f" % (i, H[i].sum()) )

            gt_aug = []
            gt_aug.append(gt[:, :4])
            for i in range(5):
                mat = gt[:, 4+i*self.nlev:4+(i+1)*self.nlev]
                mat = F.linear(mat.transpose(1, 3), self.obs_interp.interp.cpu()).transpose(1, 3)
                gt_aug.append(mat)
            gt_aug = torch.cat(gt_aug, 1)

            if self.obs_type[:22] == "real_simu_nofilteringz" or self.obs_type[:10] == "real_simuz":
                mask1 = ((yo - gt_aug) <  self.filter_coeff * torch.Tensor(self.data_reader.std_layer_aug).reshape(1, -1, 1, 1)) * 1
                mask2 = ((yo - gt_aug) > -self.filter_coeff * torch.Tensor(self.data_reader.std_layer_aug).reshape(1, -1, 1, 1)) * 1
                mask = mask1 * mask2
                mask[:, 4:44] = 1
            elif self.obs_type[:21] == "real_simu_nofiltering":
                mask = torch.zeros(yo.shape) + 1
            else:
                mask1 = ((yo - gt_aug) <  self.filter_coeff * torch.Tensor(self.data_reader.std_layer_aug).reshape(1, -1, 1, 1)) * 1
                mask2 = ((yo - gt_aug) > -self.filter_coeff * torch.Tensor(self.data_reader.std_layer_aug).reshape(1, -1, 1, 1)) * 1
                mask = mask1 * mask2

            H = H * mask

            for i in range(self.da_win):
                print("after filtering: obs[%d] amount = %.1f" % (i, H[i].sum()) )

            if self.obs_type[:10] == "real_simuz":
                yo[:, 4:44] = gt_aug[:, 4:44] * H[:, 4:44]
            elif self.obs_type[:9] == "real_simu":
                yo = gt_aug * H

            R = self.get_R_matrix_from_gt(yo, gt)
            # R = self.static_info["R"]
            # print("R min:", R.min())
        end_clock = time.time()
        print("Reading information finished. Time consumed: %d (s)" % (end_clock - start_clock), flush=True)
        return yo, H, R, gt

    # def transform(self, u, xb):

    #     class Horizontal_Corr(nn.Module):
    #         def __init__(self, layer_num, isht, sph_scale, sht, coeffs_kernel, len_scale, nchannel, nlat, nlon):
    #             super(Horizontal_Corr, self).__init__()
    #             self.layer_num = layer_num
    #             self.isht = isht
    #             self.sph_scale = sph_scale
    #             self.sht  = sht
    #             self.coeffs_kernel = coeffs_kernel
    #             self.len_scale = len_scale
    #             self.nchannel = nchannel
    #             self.nlat = nlat
    #             self.nlon = nlon

    #         def forward(self, x):
    #             # apply the first 2 layers
    #             x = cp.checkpoint(self._checkpointed_forward, x)
    #             return x

    #         def _checkpointed_forward(self, x):
    #             # inc_static = torch.zeros(self.nchannel, self.nlat, self.nlon).cuda()
    #             # for i in range(self.nchannel):
    #                 # print(i)
    #             inc_static = self.isht(self.sph_scale*self.sht(x)*self.coeffs_kernel[:, 0].reshape((self.nlat, 1)))

    #             # inc_static = 2000 * inc_static / (self.len_scale.reshape(-1, 1, 1)**2)

    #             return inc_static

    #     inc_static = torch.zeros(self.nchannel, self.nlat, self.nlon).cuda()
    #     for i in range(self.nchannel):
    #         horizon_corr = Horizontal_Corr(i, self.static_info["isht"], self.static_info["sph_scale"], self.static_info["sht"], self.static_info["coeffs_kernel"][i], self.b_matrix["len_scale"], self.nchannel, self.nlat, self.nlon)
    #         inc_static[i] = horizon_corr(u[i])
    #     inc_static = 2000 * inc_static / (self.b_matrix["len_scale"].reshape(-1, 1, 1)**2)
    #     # horizon_corr = Horizontal_Corr(self.static_info["isht"], self.static_info["sph_scale"], self.static_info["sht"], self.static_info["coeffs_kernel"], self.b_matrix["len_scale"], self.nchannel, self.nlat, self.nlon)
    #     # inc_static   = horizon_corr(u)

    #     inc_psi   = inc_static[4+self.nlev*2:4+self.nlev*3]
    #     inc_vmode = torch.clone(inc_static)

    #     for i in range(self.nchannel):
    #         inc_vmode[i] = inc_static[i] + torch.sum(inc_psi * self.b_matrix["reg_coeff"][i].reshape(-1, 1, 1), 0)

    #     inc_vmode[0:4] = inc_vmode[0:4] * self.b_matrix["std_sur"].reshape(-1, 1, 1)

    #     for i in range(5):
    #         sample = inc_vmode[4+self.nlev*i:4+self.nlev*(i+1), :, :].reshape(self.nlev, -1)
    #         sample = torch.matmul(self.b_matrix["vert_eig_vec"][i], torch.matmul(torch.sqrt(torch.diag(self.b_matrix["vert_eig_value"][i])), sample)).reshape(self.nlev, self.nlat, self.nlon)
    #         inc_vmode[4+self.nlev*i:4+self.nlev*(i+1), :, :] = sample

    #     def partial_x(field):
    #         x_scaling = torch.sin(torch.linspace(1/180 * torch.pi, 179/180 * torch.pi, self.nlat)).reshape(1, -1, 1).to(self.device)
    #         field_shift_1 = torch.cat([field[:, :,  1:], field[:, :,  :1]], 2)
    #         field_shift_2 = torch.cat([field[:, :, -1:], field[:, :, :-1]], 2)
    #         return (field_shift_2 - field_shift_1) / (2 * 111195 * 180 / self.nlat * x_scaling)

    #     def partial_y(field):
    #         lat_coord = (torch.arange(self.nlat).to(self.device) * 111195 * 180 / (self.nlat-1), )
    #         return torch.gradient(field, spacing = lat_coord, dim=1)[0]

    #     sfx = partial_x(inc_vmode[4+self.nlev*2:4+self.nlev*3])
    #     sfy = partial_y(inc_vmode[4+self.nlev*2:4+self.nlev*3])
    #     vpx = partial_x(inc_vmode[4+self.nlev*3:4+self.nlev*4])
    #     vpy = partial_y(inc_vmode[4+self.nlev*3:4+self.nlev*4])

    #     inc_vmode[4+self.nlev*2:4+self.nlev*3] =  sfy - vpx
    #     inc_vmode[4+self.nlev*3:4+self.nlev*4] = -sfx - vpy

    #     return inc_vmode + xb

    def transform(self, u, xb):
        inc_static = []
        nlat_s = 128
        nlon_s = 256

        for i in range(self.nchannel):
            coeffs_field  = self.static_info["sht"](u[i])
            inc_static.append(self.static_info["isht"](self.static_info["sph_scale"]*coeffs_field*self.static_info["coeffs_kernel"][i][:, 0].reshape((nlat_s, 1))).unsqueeze(0))

        inc_static = torch.cat(inc_static, 0)
        inc_static = 11 * inc_static / (self.b_matrix["len_scale"].reshape(-1, 1, 1)**2)

        if self.b_matrix["reg_coeff"][0].shape[0] == self.nlev:
            inc_psi   = inc_static[4+self.nlev*2:4+self.nlev*3]
        else:
            inc_psi   = torch.cat([inc_static[4+self.nlev*0:4+self.nlev*1], inc_static[4+self.nlev*2:4+self.nlev*3]], 0)
        inc_vmode = torch.clone(inc_static)

        for i in range(self.nchannel):
            inc_vmode[i] = inc_static[i] + torch.sum(inc_psi * self.b_matrix["reg_coeff"][i].reshape(-1, 1, 1), 0)

        inc_sfvp  = torch.clone(inc_vmode)

        inc_sfvp[0:4] = inc_vmode[0:4] * self.b_matrix["std_sur"].reshape(-1, 1, 1)

        for i in range(5):
            sample = inc_vmode[4+self.nlev*i:4+self.nlev*(i+1), :, :].reshape(self.nlev, -1)
            sample = torch.matmul(self.b_matrix["vert_eig_vec"][i], torch.matmul(torch.sqrt(torch.diag(self.b_matrix["vert_eig_value"][i])), sample)).reshape(self.nlev, nlat_s, nlon_s)
            inc_sfvp[4+self.nlev*i:4+self.nlev*(i+1), :, :] = sample

        def partial_x(field):
            x_scaling = torch.sin(torch.linspace(1/180 * torch.pi, 179/180 * torch.pi, nlat_s)).reshape(1, -1, 1).to(self.device)
            field_shift_1 = torch.cat([field[:, :,  1:], field[:, :,  :1]], 2)
            field_shift_2 = torch.cat([field[:, :, -1:], field[:, :, :-1]], 2)
            return (field_shift_2 - field_shift_1) / (2 * 111195 * 180 / nlat_s * x_scaling)

        def partial_y(field):
            lat_coord = (torch.arange(nlat_s).to(self.device) * 111195 * 180 / (nlat_s-1), )
            return torch.gradient(field, spacing = lat_coord, dim=1)[0]

        inc_recon  = torch.clone(inc_sfvp)

        sfx = partial_x(inc_sfvp[4+self.nlev*2:4+self.nlev*3])
        sfy = partial_y(inc_sfvp[4+self.nlev*2:4+self.nlev*3])
        vpx = partial_x(inc_sfvp[4+self.nlev*3:4+self.nlev*4])
        vpy = partial_y(inc_sfvp[4+self.nlev*3:4+self.nlev*4])

        inc_recon[4+self.nlev*2:4+self.nlev*3] =  sfy - vpx
        inc_recon[4+self.nlev*3:4+self.nlev*4] = -sfx - vpy

        output = F.interpolate(inc_recon.unsqueeze(0), (721, 1440)).squeeze(0) + xb
        # output = F.interpolate(output.unsqueeze(0), (721, 1440)).squeeze(0)

        return output

    def one_step_DA(self, gt, xb, yo, H, R, mode):
        if self.use_eval:
            print("use eval, before obs num:", H.sum())
            H_old = H.clone()
            H = H.clone() * (1 - self.mask_eval)
            print("after obs num:", H.sum())
        else:
            print("not use eval")
            H_old = H
        if mode == "free_run":
            xb = xb.cpu()
            gt_norm  = (gt[0] - self.model_mean.reshape(-1, 1, 1)) / self.model_std.reshape(-1, 1, 1)  # C x H x W
            xb_norm  = (xb - self.model_mean.reshape(-1, 1, 1)) / self.model_std.reshape(-1, 1, 1)  # C x H x W
            WRMSE_bg = self.metric.WRMSE(xb_norm.unsqueeze(0).clone().cpu(), gt_norm.unsqueeze(0).clone().cpu(), None, None, self.model_std).detach()
            bias_bg  = self.metric.Bias(xb_norm.unsqueeze(0).clone().cpu(), gt_norm.unsqueeze(0).clone().cpu(), None, None, self.model_std).detach()
            MSE_bg   = torch.mean((xb_norm - gt_norm)**2).item()

            self.metrics_list["bg_wrmse"].append(WRMSE_bg)
            self.metrics_list["bg_bias"].append(bias_bg)
            self.metrics_list["bg_mse"].append(MSE_bg)

            start_clock = time.time()
            xa = xb
            layer = 11
            print("MSE (total): %.4g RMSE (z500): %.4g Bias (z500): %.4g" % (MSE_bg, WRMSE_bg[layer], bias_bg[layer]), flush=True)
            end_clock   = time.time()

            self.metrics_list["ana_wrmse"].append(WRMSE_bg)
            self.metrics_list["ana_bias"].append(bias_bg)
            self.metrics_list["ana_mse"].append(MSE_bg)

            print("%s DA finished. Time consumed: %d (s)" % (self.current_time, end_clock - start_clock), flush=True)

            return xa.cuda()

        elif mode == "interpolation":
            xb = xb.cpu()
            gt_norm  = (gt[0] - self.model_mean.reshape(-1, 1, 1)) / self.model_std.reshape(-1, 1, 1)  # C x H x W
            xb_norm  = (xb - self.model_mean.reshape(-1, 1, 1)) / self.model_std.reshape(-1, 1, 1)  # C x H x W
            WRMSE_bg = self.metric.WRMSE(xb_norm.unsqueeze(0).clone().cpu(), gt_norm.unsqueeze(0).clone().cpu(), None, None, self.model_std).detach()
            bias_bg  = self.metric.Bias(xb_norm.unsqueeze(0).clone().cpu(), gt_norm.unsqueeze(0).clone().cpu(), None, None, self.model_std).detach()
            MSE_bg   = torch.mean((xb_norm - gt_norm)**2).item()

            print("RMSE (z500): %.4g Bias (z500): %.4g q500: %.4g, t2m: %.4g t850: %.4g u500: %.4g, v500: %.4g" % (WRMSE_bg[11].item(), bias_bg[11].item(), WRMSE_bg[24].item(), WRMSE_bg[2].item(), WRMSE_bg[66].item(), WRMSE_bg[37].item(), WRMSE_bg[50].item()), flush=True)

            self.metrics_list["bg_wrmse"].append(WRMSE_bg)
            self.metrics_list["bg_bias"].append(bias_bg)
            self.metrics_list["bg_mse"].append(MSE_bg)

            start_clock = time.time()
            y0 = yo[0].numpy()
            H0 = H[0].numpy()
            # # 将 A 中的已知值提取出来（对应 B 中为 1 的位置）
            # known_values = y0[H0 == 1]
            # known_coords = np.argwhere(H0 == 1)  # 已知值的坐标

            # # 构造插值目标点（未知值的坐标）
            # unknown_coords = np.argwhere(H0 == 0)

            if self.obs_type[:4] == "real":
                xb0 = xb.unsqueeze(0)
                xb_aug = []
                xb_aug.append(xb0[:, :4])

                for i in range(5):
                    mat = xb0[:, 4+i*self.nlev:4+(i+1)*self.nlev]
                    mat = F.linear(mat.transpose(1, 3), self.obs_interp.interp.cpu()).transpose(1, 3)
                    xb_aug.append(mat)

                xb_aug = torch.cat(xb_aug, 1)
                xb0 = xb_aug.squeeze(0).numpy()

            else:
                xb0 = xb.numpy()

            # # 使用 griddata 进行三维插值
            # xa = xb0.numpy().copy()  # 创建 A 的副本用于填补
            # xa[H0 == 0] = griddata(known_coords, known_values, unknown_coords, method='linear')
            # mask = np.isnan(xa)
            # xa[mask] = xb0[mask]
            # xa = torch.from_numpy(xa)
            # print(xa)

            xa = xb0.copy()  # 创建 A 的副本用于填补
            for i in range(204):
                # print("layer", i)
                a = y0[i]
                b = H0[i]
                known_values = a[b == 1]
                known_coords = np.argwhere(b == 1)  # 已知值的坐标
                unknown_coords = np.argwhere(b == 0)
                # print("layer", i, len(known_values))
                if len(known_values) > 10:
                
                    xa[i][b == 0] = griddata(known_coords, known_values, unknown_coords, method='linear')
                
            mask = np.isnan(xa)
            xa[mask] = xb0[mask]
            xa = torch.from_numpy(xa)

            if self.obs_type[:4] == "real":
                xa0 = xa.unsqueeze(0)
                xa_aug = []
                xa_aug.append(xa0[:, :4])

                for i in range(5):
                    mat = xa0[:, 4+i*40:4+(i+1)*40]
                    mat = F.linear(mat.transpose(1, 3), self.obs_interp.interp_inv.cpu()).transpose(1, 3)
                    xa_aug.append(mat)

                xa_aug = torch.cat(xa_aug, 1)
                xa = xa_aug.squeeze(0)
            # print(xa)

            end_clock   = time.time()

            xa_norm  = (xa - self.model_mean.reshape(-1, 1, 1)) / self.model_std.reshape(-1, 1, 1)  # C x H x W
            WRMSE_ana = self.metric.WRMSE(xa_norm.unsqueeze(0).clone().cpu(), gt_norm.unsqueeze(0).clone().cpu(), None, None, self.model_std).detach()
            bias_ana  = self.metric.Bias(xa_norm.unsqueeze(0).clone().cpu(), gt_norm.unsqueeze(0).clone().cpu(), None, None, self.model_std).detach()
            MSE_ana   = torch.mean((xa_norm - gt_norm)**2).item()

            print("RMSE (z500): %.4g Bias (z500): %.4g q500: %.4g, t2m: %.4g t850: %.4g u500: %.4g, v500: %.4g" % (WRMSE_ana[11].item(), bias_ana[11].item(), WRMSE_ana[24].item(), WRMSE_ana[2].item(), WRMSE_ana[66].item(), WRMSE_ana[37].item(), WRMSE_ana[50].item()), flush=True)

            self.metrics_list["ana_wrmse"].append(WRMSE_ana)
            self.metrics_list["ana_bias"].append(bias_ana)
            self.metrics_list["ana_mse"].append(MSE_ana)
            print("%s DA finished. Time consumed: %d (s)" % (self.current_time, end_clock - start_clock), flush=True)

            return xa.cuda()


        elif mode == "sc4dvar":
            def cal_loss_bg(x0):
                """
                x0:     C x H x W
                """
                return torch.sum(x0**2) / 2

            def cal_loss_obs(x):
                """
                x0:       C x H x W
                obs:      T x C x H x W
                H:        T x C x H x W
                obs_var:  T x C x H x W
                """
                x_list = [x, ]
                for i in range(self.da_win-1):
                    x = self.integrate(x, self.flow_model, 1, True)[:69]
                    x_list.append(x)

                x_pred = torch.stack(x_list, 0)   # T x C x H x W

                if self.obs_type[:4] == "real":
                    x_aug = []
                    x_aug.append(x_pred[:, :4])

                    for i in range(5):
                        mat = x_pred[:, 4+i*self.nlev:4+(i+1)*self.nlev]
                        mat = F.linear(mat.transpose(1, 3), self.obs_interp.interp).transpose(1, 3)
                        x_aug.append(mat)

                    x_aug = torch.cat(x_aug, 1)
                    x_pred = x_aug

                return torch.sum( H * (x_pred - yo) ** 2 / R ) / 2

            def loss(w):
                xhat = self.transform(w, xb)
                return cal_loss_bg(w) + self.obs_coeff * cal_loss_obs(xhat)

            def closure():
                lbfgs.zero_grad()
                objective = loss(w)
                objective.backward()
                return objective 

            xb = xb.cuda()
            yo = yo.cuda()
            H  = H.cuda()
            R  = R.cuda()

            nlat_s = 128
            nlon_s = 256
            w = torch.autograd.Variable(torch.zeros(self.nchannel, nlat_s, nlon_s).to(self.device), requires_grad=True)
            gt_norm = (gt[0] - self.model_mean.reshape(-1, 1, 1)) / self.model_std.reshape(-1, 1, 1)
            gt_norm = gt_norm.cuda()
            lbfgs = optim.LBFGS([w], history_size=10, max_iter=5, line_search_fn="strong_wolfe")
            idx = 11
            start_clock = time.time()           

            kk = 0
            while kk <= self.Nit:
                xhat = self.transform(w, xb).detach()
                xhat_norm  = (xhat - self.model_mean_gpu.reshape(-1, 1, 1)) / self.model_std_gpu.reshape(-1, 1, 1)
                WRMSE_GT = self.metric.WRMSE(xhat_norm.unsqueeze(0).clone().detach().cpu(), gt_norm.unsqueeze(0).clone().detach().cpu(), None, None, self.model_std).detach()
                bias_GT = self.metric.Bias(xhat_norm.unsqueeze(0).clone().detach().cpu(), gt_norm.unsqueeze(0).clone().detach().cpu(), None, None, self.model_std).detach()
                RMSE_z500_GT = WRMSE_GT[idx].item()
                bias_z500_GT = bias_GT[idx].item()
                MSE_GT = torch.mean((xhat_norm - gt_norm)**2).detach().item()
                loss_total = loss(w)
                loss_bg  = cal_loss_bg(w).detach().item()
                loss_obs = cal_loss_obs(xhat).detach().item()\

                print("iter: %d, RMSE (z500): %.4g Bias (z500): %.4g q500: %.4g, t2m: %.4g t850: %.4g u500: %.4g, v500: %.4g, loss reg: %.4g loss obs: %.4g loss: %.4g" % (kk, RMSE_z500_GT, bias_z500_GT, WRMSE_GT[24].item(), WRMSE_GT[2].item(), WRMSE_GT[66].item(), WRMSE_GT[37].item(), WRMSE_GT[50].item(), loss_bg, loss_obs, loss_bg + self.obs_coeff * loss_obs), flush=True)

                # print("iter: %d, MSE (total): %.4g RMSE (z500): %.4g Bias (z500): %.4g" % (kk, MSE_GT, RMSE_z500_GT, bias_z500_GT), flush=True)
                if kk == self.Nit:
                    if self.obs_type[:4] == "real":
                        xhat0 = xhat.unsqueeze(0)
                        xhat_aug = []
                        xhat_aug.append(xhat0[:, :4])

                        for i in range(5):
                            mat = xhat0[:, 4+i*self.nlev:4+(i+1)*self.nlev]
                            mat = F.linear(mat.transpose(1, 3), self.obs_interp.interp).transpose(1, 3)
                            xhat_aug.append(mat)

                        xhat_aug = torch.cat(xhat_aug, 1)
                        xhat_aug = xhat_aug.squeeze(0)
                    else:
                        xhat_aug = xhat
                    if self.use_eval:
                        error_obs = torch.sqrt(torch.sum((xhat_aug.clone().detach().cpu() - yo[0].clone().detach().cpu())**2 * self.mask_eval[0] * H_old[0].clone().detach().cpu(), (1, 2)) / torch.sum(self.mask_eval[0].clone().detach().cpu() * H_old[0].clone().detach().cpu(), (1, 2))).numpy()
                        print(error_obs[4:34])
                
                if kk == 0:
                    self.metrics_list["bg_wrmse"].append(WRMSE_GT)
                    self.metrics_list["bg_bias"].append(bias_GT)
                elif kk == self.Nit:
                    self.metrics_list["ana_wrmse"].append(WRMSE_GT)
                    self.metrics_list["ana_bias"].append(bias_GT)
                    if self.use_eval:
                        self.metrics_list["error_obs"].append(error_obs)

                if kk < self.Nit:
                    lbfgs.step(closure)

                kk = kk + 1
            
            w.detach()
            xhat = self.transform(w, xb)
            end_clock = time.time()
            print("%s DA finished. Time consumed: %d (s)" % (self.current_time, end_clock - start_clock), flush=True)

            return xhat

        elif mode == "vae4dvar":

            stdTr = torch.Tensor([0.18955279, 0.22173745, 0.03315084, 0.08258388, 0.03021586, 0.0194484 , 0.01700376, 0.01931592, 0.02327741, 0.02647366, 0.02925515, 0.0304862 , 0.03300306, 0.03865351, 0.05609745, 0.0682424 , 0.07762259, 0.50658824, 0.29907974, 0.22097995, 0.22990653, 0.26931248, 0.27226337, 0.26211415, 0.24042704, 0.20803592, 0.18460007, 0.12343913, 0.06593712, 0.04856134, 0.11308974, 0.11406155, 0.10717956, 0.12138538, 0.14543332, 0.16263002, 0.17114112, 0.16359221, 0.1600293 , 0.16136173, 0.17905815, 0.19142863, 0.18638292, 0.13128242, 0.1593278, 0.16516368, 0.17795471, 0.19510655, 0.20854117, 0.21904777, 0.21593404, 0.21397153, 0.21613599, 0.23249907, 0.23790329, 0.21999044, 0.06977215, 0.03924686, 0.06015565, 0.11465897, 0.09490499, 0.06113996, 0.05008726, 0.04878271, 0.04601997, 0.04151259, 0.04477754, 0.04275933, 0.03838996]).to(self.device).reshape(1, 69, 1, 1)

            def loss(z):
                loss_reg = torch.sum(z**2) / 2
                x = self.vae.decoder_hr(z)
                # x = x - torch.mean(x, (2, 3), keepdim=True)
                x = (x * stdTr) * self.model_std_gpu.reshape(1, -1, 1, 1) + xb
                x = x[0]
                # print(x.shape, yo.shape, H.shape, R.shape)
                x_list = [x, ]
                for i in range(self.da_win-1):
                    x = self.integrate(x, self.flow_model, 1, True, False)[:69]
                    x_list.append(x)

                x_pred = torch.stack(x_list, 0)   # T x C x H x W
                if self.obs_type[:4] == "real":
                    x_aug = []
                    x_aug.append(x_pred[:, :4])

                    for i in range(5):
                        mat = x_pred[:, 4+i*self.nlev:4+(i+1)*self.nlev]
                        mat = F.linear(mat.transpose(1, 3), self.obs_interp.interp).transpose(1, 3)
                        x_aug.append(mat)

                    x_aug = torch.cat(x_aug, 1)
                    x_pred = x_aug
                loss_obs = torch.sum( H * (x_pred - yo) ** 2 / R ) / 2
                return loss_reg + self.obs_coeff * loss_obs #+ loss_det

            def cal_loss(z):
                loss_reg = torch.sum(z**2) / 2
                x = self.vae.decoder_hr(z)
                # x = x - torch.mean(x, (2, 3), keepdim=True)
                x = (x * stdTr) * self.model_std_gpu.reshape(1, -1, 1, 1) + xb
                x = x[0]
                # print(x.shape, yo.shape, H.shape, R.shape)
                x_list = [x, ]
                for i in range(self.da_win-1):
                    x = self.integrate(x, self.flow_model, 1, True)[:69]
                    x_list.append(x)

                x_pred = torch.stack(x_list, 0)   # T x C x H x W
                if self.obs_type[:4] == "real":
                    x_aug = []
                    x_aug.append(x_pred[:, :4])

                    for i in range(5):
                        mat = x_pred[:, 4+i*self.nlev:4+(i+1)*self.nlev]
                        mat = F.linear(mat.transpose(1, 3), self.obs_interp.interp).transpose(1, 3)
                        x_aug.append(mat)

                    x_aug = torch.cat(x_aug, 1)
                    x_pred = x_aug
                # print(x_pred.shape, yo.shape, H.shape)
                loss_obs = torch.sum( H * (x_pred - yo) ** 2 / R ) / 2
                return loss_reg.detach().cpu(), loss_obs.detach().cpu()
            
            z = torch.zeros(1, 32, 128, 256, requires_grad=True, device="cuda")
            gt_norm  = (gt[0] - self.model_mean.reshape(-1, 1, 1)) / self.model_std.reshape(-1, 1, 1)
            lbfgs = optim.LBFGS([z], history_size=10, max_iter=10, line_search_fn="strong_wolfe")

            def closure():
                lbfgs.zero_grad()
                objective = loss(z)
                objective.backward()
                return objective 

            xb = xb.cuda()
            yo = yo.cuda()
            H  = H.cuda()
            R  = R.cuda()

            idx = 11
            start_clock = time.time()   
            for kk in range(self.Nit+1):
                output = self.vae.decoder_hr(z).detach()
                # output = output - torch.mean(output, (2, 3), keepdim=True)
                # output = z
                xhat = output[0] * stdTr[0] * self.model_std_gpu.reshape(-1, 1, 1) + xb
                xhat_norm  = (xhat - self.model_mean_gpu.reshape(-1, 1, 1)) / self.model_std_gpu.reshape(-1, 1, 1)
                WRMSE_GT = self.metric.WRMSE(xhat_norm.unsqueeze(0).clone().detach().cpu(), gt_norm.unsqueeze(0).clone().detach().cpu(), None, None, self.model_std).detach()
                bias_GT = self.metric.Bias(xhat_norm.unsqueeze(0).clone().detach().cpu(), gt_norm.unsqueeze(0).clone().detach().cpu(), None, None, self.model_std).detach()
                RMSE_z500_GT = WRMSE_GT[idx].item()
                bias_z500_GT = bias_GT[idx].item()
                loss_reg, loss_obs = cal_loss(z)

                # print(xhat.shape, yo.shape, H_old.shape, self.mask_eval.shape)
                
                print("iter: %d, RMSE (z500): %.4g Bias (z500): %.4g q500: %.4g, t2m: %.4g t850: %.4g u500: %.4g, v500: %.4g, loss reg: %.4g loss obs: %.4g loss: %.4g" % (kk, RMSE_z500_GT, bias_z500_GT, WRMSE_GT[24].item(), WRMSE_GT[2].item(), WRMSE_GT[66].item(), WRMSE_GT[37].item(), WRMSE_GT[50].item(), loss_reg, loss_obs, loss_reg + self.obs_coeff * loss_obs), flush=True)
                if kk == self.Nit:
                    if self.obs_type[:4] == "real":
                        xhat0 = xhat.unsqueeze(0)
                        xhat_aug = []
                        xhat_aug.append(xhat0[:, :4])

                        for i in range(5):
                            mat = xhat0[:, 4+i*self.nlev:4+(i+1)*self.nlev]
                            mat = F.linear(mat.transpose(1, 3), self.obs_interp.interp).transpose(1, 3)
                            xhat_aug.append(mat)

                        xhat_aug = torch.cat(xhat_aug, 1)
                        xhat_aug = xhat_aug.squeeze(0)
                    else:
                        xhat_aug = xhat
                    if self.use_eval:
                        error_obs = torch.sqrt(torch.sum((xhat_aug.clone().detach().cpu() - yo[0].clone().detach().cpu())**2 * self.mask_eval[0] * H_old[0].clone().detach().cpu(), (1, 2)) / torch.sum(self.mask_eval[0].clone().detach().cpu() * H_old[0].clone().detach().cpu(), (1, 2))).numpy()
                        print(error_obs[4:34])
                
                if kk == 0:
                    self.metrics_list["bg_wrmse"].append(WRMSE_GT)
                    self.metrics_list["bg_bias"].append(bias_GT)
                elif kk == self.Nit:
                    self.metrics_list["ana_wrmse"].append(WRMSE_GT)
                    self.metrics_list["ana_bias"].append(bias_GT)
                    if self.use_eval:
                        self.metrics_list["error_obs"].append(error_obs)

                if kk < self.Nit:
                    lbfgs.step(closure)

            output = self.vae.decoder_hr(z)
            # output = output - torch.mean(output, (2, 3), keepdim=True)
            end_clock = time.time()
            print("%s DA finished. Time consumed: %d (s)" % (self.current_time, end_clock - start_clock), flush=True)

            return output[0].detach() * stdTr[0] * self.model_std_gpu.reshape(-1, 1, 1) + xb

        else:
            raise NotImplementedError("not implemented da mode")

    def evaluate(self, ):
        return

    def run_assimilation(self):
        epoch = 0

        while(self.current_time + self.cycle_time <= self.end_time):
            print("current time:", self.current_time)

            print("obtaining observations...")
            yo, H, R, gt = self.get_obs_info()
            
            print("assimilating...")
            self.xa = self.one_step_DA(gt, self.xb, yo, H, R, self.da_mode)  # [69, 721, 1440]

            self.save_eval_result(finish=False, gt=gt, obs=yo)

            print("integrating...")
            self.xb = self.integrate(self.xa, self.forecast_model, 1)

            if self.forecast_eval:
                self.evaluate()

            self.current_time = self.current_time + self.cycle_time

            if epoch % self.save_interval == 0:
                self.save_ckpt(finish=False, gt=gt, obs=yo)
                
            epoch += 1

        print("DA complete")
        self.save_eval_result(finish=True, gt=None)

if __name__ == "__main__":
    args = arg_parser()
    da_agent = cyclic_4dvar(args)
    da_agent.run_assimilation()

import argparse
import os
import torch
from utils.builder import ConfigBuilder
import utils.misc as utils
import yaml
from utils.logger import get_logger
import copy
from utils.metrics import MetricsRecorder
import numpy as np
from model.model import vae_nmc_model

mean_layer = np.array([-0.14186215714480854, 0.22575792335029873, 278.7854495405721, 100980.83590625007, 199832.31609374992, 157706.1917968749, 132973.8087890624, 115011.55044921875, 100822.13164062506, 88999.83613281258, 69620.0044531249, 53826.54542968748, 40425.96180664062, 28769.254521484374, 13687.02337158203, 7002.870792236329, 777.5631800842285, 2.8248029025235157e-06, 2.557213611567022e-06, 4.689598504228342e-06, 1.7365863168379306e-05, 5.37612270545651e-05, 0.00012106754767955863, 0.0003586592462670523, 0.0007819174298492726, 0.0014082587775192225, 0.002245682779466732, 0.004328316930914292, 0.005698622210184111, 0.006659231842495503, 4.44909584343433, 10.046632840633391, 14.321160042285918, 15.298378415107727, 14.48938421010971, 12.895844810009004, 9.628437678813944, 7.07798705458641, 5.110536641478544, 3.4704639464616776, 1.2827875773236155, 0.3961004569224316, -0.18604825597634778, 0.012106836824341376, 0.1010729405652091, 0.2678451650420902, 0.2956721917196408, 0.21001753183547414, 0.03872977272505523, -0.04722135595180817, 0.0007164070030103152, -0.022026948712546065, 0.0075308467486320295, 0.014846984493779027, -0.062323193841984835, -0.15797925526494516, 214.66564151763913, 210.3573041915893, 215.23375904083258, 219.73181056976318, 223.53410289764412, 228.6614455413818, 241.16466262817383, 251.74072200775146, 259.84156120300344, 265.99485839843743, 272.77368919372566, 275.3001181793211, 278.5929747772214])

std_layer = np.array([5.610453475051704, 4.798220612223473, 21.32010786700973, 1336.2115992274876, 3755.2810557402927, 4357.588191568988, 5253.301115477269, 5540.73074484052, 5405.73040397736, 5020.194961603476, 4104.233456672573, 3299.702929930327, 2629.7201995715513, 2060.9872289877453, 1399.3410970050247, 1187.5419349409494, 1098.9952409939283, 1.1555282996146702e-07, 4.2315237954921815e-07, 3.1627283344500357e-06, 2.093742795871515e-05, 7.02963683704546e-05, 0.00016131853114827985, 0.00048331132466880735, 0.001023028433607086, 0.0016946778969914426, 0.0024928432426471183, 0.004184742037434761, 0.005201345241925773, 0.00611814321149996, 11.557361639969054, 11.884088705628045, 15.407016747306344, 17.286773058038722, 17.720698660431694, 17.078782531259524, 14.509924979003983, 12.215305549952125, 10.503871726997783, 9.286354460633103, 8.179197305830433, 7.93264239491015, 6.126056325796786, 8.417864770061094, 8.178248048405905, 9.998695230009567, 11.896325029659364, 13.360381609448558, 13.474533447403218, 11.44656476066317, 9.321096224035244, 7.835396470389893, 6.858187372121642, 6.186618416862026, 6.345356147017278, 5.23175612906023, 9.495652698988557, 13.738672642636256, 9.090666595626503, 5.933385737657316, 7.389004707914384, 10.212310312072752, 12.773099916244078, 13.459313552230206, 13.858620163486986, 15.021590351519892, 16.00275340237577, 16.88523210573196, 18.59201174892538])

device = "cuda"

mean_layer_gpu = torch.from_numpy(mean_layer).float().to(device)
std_layer_gpu  = torch.from_numpy(std_layer).float().to(device)

#----------------------------------------------------------------------------

def subprocess_fn(args):
    utils.setup_seed(args.seed * args.world_size + args.rank)

    logger = get_logger("training VAE", args.run_dir, utils.get_rank(), filename='get_error%d.log'%args.predict_len)

    # args.logger = logger
    args.cfg_params["logger"] = logger

    # build config
    logger.info('Building config ...')
    builder = ConfigBuilder(**args.cfg_params)

    # build model
    logger.info('Building models ...')
    model = vae_nmc_model(logger, args.param_str, args.sigma, "output/model/model_0.25degree/checkpoint_latest.pth", "nf_model/ckpts/vae_parameters0_old_sigma2.00_epoch4_19790101_20151231_finetuned.pt", device=device, **args.cfg_params["model"])

    model_without_ddp = utils.DistributedParallel_VAEModel(model, args.local_rank)

    if args.world_size > 1:
        utils.check_ddp_consistency(model_without_ddp.vae)
        utils.check_ddp_consistency(model_without_ddp.fengwu)

    # build dataset
    logger.info('Building dataloaders ...')
    
    dataset_params = args.cfg_params['dataset']
    dataset_params['train']['type'] = "weather_dataset"
    dataset_params['test']['type'] = "weather_dataset"
    dataset_params['train']['years']['train'][0] = args.start_year
    print(dataset_params['train']['years'])

    train_dataloader = builder.get_dataloader(dataset_params=dataset_params, split ='train', batch_size=args.batch_size)

    logger.info('dataloaders build complete')

    params = [p for p in model_without_ddp.vae.parameters() if p.requires_grad]
    cnt_params = sum([p.numel() for p in params])
    # print("params {key}:".format(key=key), cnt_params)
    logger.info("params: {cnt_params}".format(cnt_params=cnt_params))

    logger.info('begin training vae ...')

    model.train(train_dataloader, args.epoch)

    
def main(args):
    if args.world_size > 1:
        utils.init_distributed_mode(args)
    else:
        args.rank = 0
        args.distributed = False
        args.local_rank = 0
        torch.cuda.set_device(args.local_rank)

    run_dir = args.cfgdir
    print(run_dir)
    
    args.cfg = os.path.join(args.cfgdir, 'training_options.yaml')
    with open(args.cfg, 'r') as cfg_file:
        cfg_params = yaml.load(cfg_file, Loader = yaml.FullLoader)
    # 根据申请的cpu数来设置dataloader的线程数
    cfg_params['dataloader']['num_workers'] = args.per_cpus
    cfg_params['dataset']['train']['length'] = args.length
    # cfg_params['dataset']['train']['file_stride'] = args.predict_len

    #判断是否使用常量数据
    dataset_vnames = cfg_params['dataset']['train'].get("vnames", None)

    args.cfg_params = cfg_params
    args.run_dir = run_dir
    if "relative_checkpoint_dir" in cfg_params:
        args.relative_checkpoint_dir = cfg_params['relative_checkpoint_dir']

    print('Launching processes...')
    subprocess_fn(args)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',           type = int,     default = 0,                                            help = 'seed')
    parser.add_argument('--cuda',           type = int,     default = 0,                                            help = 'cuda id')
    parser.add_argument('--world_size',     type = int,     default = 1,                                            help = 'Number of progress')
    parser.add_argument('--per_cpus',       type = int,     default = 1,                                            help = 'Number of perCPUs to use')
    parser.add_argument('--length',         type = int,     default = 2,                                           help = "predict len")
    parser.add_argument('--metric_list',    nargs = '+',                                                            help = 'metric list')
    # parser.add_argument('--world_size',     type = int,     default = -1,                                           help = 'number of distributed processes')
    parser.add_argument('--init_method',    type = str,     default = 'tcp://127.0.0.1:23456',                      help = 'multi process init method')
    parser.add_argument('--cfgdir',         type = str,     default = '/mnt/petrelfs/chenkang/code/game/output',  help = 'Where to save the results')
    parser.add_argument('--batch_size',     type = int,     default = 32,                                           help = "batch size")
    parser.add_argument('--predict_len',    type = int,     default = 15,                                           help = "predict len")
    parser.add_argument('--sigma',          type = float,   default = 0.3,                                           help = "loss sigma")
    parser.add_argument('--epoch',          type = int,     default = 20,                                           help = "epoch number")
    parser.add_argument('--param_str',      type = str,     default = None,                                         help = 'vae parameter setting')
    parser.add_argument('--start_year',     type = str,     default = None,                                         help = 'vae parameter setting')

    args = parser.parse_args()

    main(args)

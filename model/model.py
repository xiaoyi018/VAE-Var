import torch
import torch.nn as nn
import torch.nn.functional as F

# from networks.PVFlash import BidirectionalTransformer
from utils.builder import get_optimizer, get_lr_scheduler
from utils.metrics import MetricsRecorder
import utils.misc as utils
import time
import datetime
from pathlib import Path
import torch.cuda.amp as amp
import os
from collections import OrderedDict
from torch.functional import F
import torch.optim as optim
from networks.LGUnet_all import LGUnet_all_1
from nf_model.vae import VAE_lr, loss_function
import numpy as np
import json
from torch.utils.tensorboard import SummaryWriter
import PIL.Image
from torchvision.transforms import ToTensor


class basemodel(nn.Module):
    def __init__(self, logger, writer, **params) -> None:
        super().__init__()
        self.model = {}
        self.sub_model_name = []
        self.params = params
        self.logger = logger
        self.writer = writer
        self.save_best_param = self.params.get("save_best", "MSE")
        self.metric_best = None



        self.begin_epoch = 0
        self.metric_best = 1000
        self.std_data = self._get_std()

        self.gscaler = amp.GradScaler(init_scale=1024, growth_interval=2000)

        
        # self.whether_final_test = self.params.get("final_test", False)
        # self.predict_length = self.params.get("predict_length", 20)


        network_type = params.get('type', "basenetwork")
        network_params = params.get("network_params", None)
        optimizer_params = params.get("optimizer_params", None)
        lr_params = params.get("lr_params", None)
        if network_type == "LGUnet_all":
            self.model = LGUnet_all(**network_params)
        else:
            raise NotImplementedError('Invalid network type')


        # print(lr_params)

        self.optimizer = get_optimizer(self.model, optimizer_params)
        self.lr_scheduler = get_lr_scheduler(self.optimizer, lr_params)


        # load metrics
        eval_metrics_list = params.get('metrics_list', [])
        print(eval_metrics_list)
        if len(eval_metrics_list) > 0:
            self.eval_metrics = MetricsRecorder(eval_metrics_list)
        else:
            self.eval_metrics = None

        self.model.eval()


        self.extra_params = params.get("extra_params", {})
        self.two_step_training = self.extra_params.get("two_step_training", False)
        if self.two_step_training:
            checkpoint_path = self.extra_params.get('checkpoint_path', None)
            if checkpoint_path is None:
                self.logger.info("finetune checkpoint path not exist")
            else:
                self.load_checkpoint(checkpoint_path, load_model=True, load_optimizer=False, load_scheduler=False, load_epoch=False, load_metric_best=False)
        
        self.loss_type = self.extra_params.get("loss_type", "LpLoss")

        if self.loss_type == "LpLoss":
            self.loss = self.LpLoss
        elif self.loss_type == "Possloss":
            self.loss = self.Possloss



        if self.loss_type == "Possloss":
            output_dim = self.params['network_params']["out_chans"]
            img_size = self.params['network_params'].get("img_size", [32, 64])
            self.max_logvar = self.model.max_logvar = torch.nn.Parameter((torch.ones((1, output_dim*img_size[-2]*img_size[-1]//2)).float() / 2))
            self.min_logvar = self.model.min_logvar = torch.nn.Parameter((-torch.ones((1, output_dim*img_size[-2]*img_size[-1]//2)).float() * 10))


    def to(self, device):
        self.device = device
        self.model.to(device)
        
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    def _get_std(self):
        # with open('./dataset/weatherbench_69.json',mode='r') as f:
        #     datainfo = json.load(f)

        # std_layer = np.zeros(69)
        # for key in datainfo:
        #     std_layer[datainfo[key]["channel"]] = datainfo[key]["std"]
        std_layer = np.array([5.610453475051704, 4.798220612223473, 21.32010786700973, 1336.2115992274876, 3755.2810557402927, 4357.588191568988, 5253.301115477269, 5540.73074484052, 5405.73040397736, 5020.194961603476, 4104.233456672573, 3299.702929930327, 2629.7201995715513, 2060.9872289877453, 1399.3410970050247, 1187.5419349409494, 1098.9952409939283, 1.1555282996146702e-07, 4.2315237954921815e-07, 3.1627283344500357e-06, 2.093742795871515e-05, 7.02963683704546e-05, 0.00016131853114827985, 0.00048331132466880735, 0.001023028433607086, 0.0016946778969914426, 0.0024928432426471183, 0.004184742037434761, 0.005201345241925773, 0.00611814321149996, 11.557361639969054, 11.884088705628045, 15.407016747306344, 17.286773058038722, 17.720698660431694, 17.078782531259524, 14.509924979003983, 12.215305549952125, 10.503871726997783, 9.286354460633103, 8.179197305830433, 7.93264239491015, 6.126056325796786, 8.417864770061094, 8.178248048405905, 9.998695230009567, 11.896325029659364, 13.360381609448558, 13.474533447403218, 11.44656476066317, 9.321096224035244, 7.835396470389893, 6.858187372121642, 6.186618416862026, 6.345356147017278, 5.23175612906023, 9.495652698988557, 13.738672642636256, 9.090666595626503, 5.933385737657316, 7.389004707914384, 10.212310312072752, 12.773099916244078, 13.459313552230206, 13.858620163486986, 15.021590351519892, 16.00275340237577, 16.88523210573196, 18.59201174892538])
        return std_layer

    def data_preprocess(self, data):
        inp = [data[0]]
        if self.two_step_training == False:
            for i in range(1, len(data)-1):
                inp.append(data[i])
            inp = torch.cat(inp, dim=1).float().to(self.device, non_blocking=True)

            tar_step1 = data[-1].float().to(self.device, non_blocking=True)
            # inp, tar_step1 = [x.float().to(self.device, non_blocking=True) for x in data]
            # tar_step1 = tar_step1[:,self.constants_len:]
            tar_step2 = None

        else:
            for i in range(1, len(data) - self.pred_len):
                inp.append(data[i])
            inp = torch.cat(inp, dim=1).float().to(self.device, non_blocking=True)
            tar_step1 = data[0-self.pred_len].float().to(self.device, non_blocking=True)
            tar_step2 = []
            for i in range(0, self.pred_len-1):
                tar_step2.append(data[i+1-self.pred_len].float().to(self.device, non_blocking=True))
    
        # print(time.time()- begin_time)
        # print(input_data.shape)
        return inp, tar_step1, tar_step2
    

    def loss(self, predict, target):
        return torch.abs(predict, target).mean()
    

    def LpLoss(self, pred, target):
        num_examples = pred.size()[0]

        diff_norms = torch.norm(pred.reshape(num_examples,-1) - target.reshape(num_examples,-1), 2, 1)
        y_norms = torch.norm(target.reshape(num_examples,-1), 2, 1)

        return torch.mean(diff_norms/y_norms)


    def Possloss(self, pred, target, **kwargs):
        # print(pred.shape, target.shape, self.max_logvar.shape, self.min_logvar.shape)
        inc_var_loss = kwargs.get("inc_var_loss", True)
        loss_weight = kwargs.get("weight", None)
        
        
        num_examples = pred.size()[0]

        mean, log_var = pred.chunk(2, dim = 1)
        # log_var = torch.tanh(log_var)

        # mean = mean.reshape(num_examples, -1)
        log_var = log_var.reshape(num_examples, -1)
        # target = target.reshape(num_examples, -1)


        # if not hasattr(self, 'max_logvar'):
        #     self.max_logvar = torch.nn.Parameter((torch.ones((1, target.shape[-1])).float() / 2), requires_grad=True).to(self.device)
        #     self.min_logvar = torch.nn.Parameter((-torch.ones((1, target.shape[-1])).float() * 10), requires_grad=True).to(self.device)


        log_var = self.max_logvar - F.softplus(self.max_logvar - log_var)
        log_var = self.min_logvar + F.softplus(log_var - self.min_logvar)

        log_var = log_var.reshape(*(target.shape))

        inv_var = torch.exp(-log_var)
        if inc_var_loss:
            # Average over batch and dim, sum over ensembles.
            mse_loss = torch.mean(torch.pow(mean - target, 2) * inv_var, dim=(-1, -2, -3))
            var_loss = torch.mean(log_var, dim=(-1, -2, -3))
            # mse_loss = torch.mean(torch.pow(mean - target, 2) * inv_var * weight)
            # var_loss = torch.mean(log_var * weight)

            # mse_loss = torch.mean(torch.mean(torch.pow(mean - target, 2) * inv_var, dim=-1), dim=-1)
            # var_loss = torch.mean(torch.mean(log_var, dim=-1), dim=-1)
            total_loss = mse_loss + var_loss
        else:
            mse_loss = torch.mean(torch.pow(mean - target, 2), dim=(-1,-2,-3))
            # mse_loss = torch.mean(torch.pow(mean - target, 2), dim=(1, 2))
            total_loss = mse_loss
            
        total_loss += 0.01 * torch.mean(self.max_logvar) - 0.01 * torch.mean(self.min_logvar)

        if loss_weight is not None:
            total_loss = total_loss * (2 ** (torch.tensor(loss_weight.astype(np.float32)).to(self.device) / 8))

        return torch.mean(total_loss)


    def train_one_step(self, batch_data, step):
        inp, tar_step1, tar_step2 = self.data_preprocess(batch_data)
        # with amp.autocast():
        predict = self.model(inp)

        step_one_loss = self.loss(predict, tar_step1)
        # step_one_loss = torch.mean((predict - tar_step1) ** 2)
        step_two_loss = None

        if self.two_step_training:
            step1_inp = predict
            predict_step_2 = self.model(step1_inp)
            step_two_loss = self.loss(predict_step_2, tar_step2)
            loss = step_one_loss + step_two_loss
        else:
            loss = step_one_loss

        self.optimizer.zero_grad()

        loss.backward()
        self.optimizer.step()

        return {'Possloss': loss.item(), "step_one_loss": step_one_loss.item(), "step_two_loss": step_two_loss.item() if self.two_step_training else 0}


    def test_one_step(self, batch_data):
        inp, tar_step1, tar_step2 = self.data_preprocess(batch_data)
        predict = self.model(inp)

        step_one_loss = self.loss(predict, tar_step1)
        step_two_loss = None

        if self.two_step_training:
            step1_inp = predict
            predict_step_2 = self.model(step1_inp)
            step_two_loss = self.loss(predict_step_2, tar_step2)
            loss = step_one_loss + step_two_loss
        else:
            loss = step_one_loss

        data_dict = {}
        data_dict['gt'] = tar_step1
        data_dict['pred'] = predict[:,:tar_step1.shape[1]]
        metrics_loss = self.eval_metrics.evaluate_batch(data_dict)

        metrics_loss.update({'Possloss': loss.item(), "step_one_loss": step_one_loss.item(), "step_two_loss": step_two_loss.item() if step_two_loss !=None else 0})
        
        return metrics_loss


    def train_one_epoch(self, train_data_loader, epoch, max_epoches):

        self.lr_scheduler.step(epoch)

        # test_logger = {}

        end_time = time.time()        
        self.model.train()

        metric_logger = utils.MetricLogger(delimiter="  ")
        iter_time = utils.SmoothedValue(fmt='{avg:.3f}')
        data_time = utils.SmoothedValue(fmt='{avg:.3f}')
        max_step = len(train_data_loader)

        header = 'Epoch [{epoch}/{max_epoches}][{step}/{max_step}]'
        for step, batch in enumerate(train_data_loader):
            self.lr_scheduler.step(epoch*max_step+step)
            # record data read time
            data_time.update(time.time() - end_time)
            
            loss = self.train_one_step(batch, step)

            if utils.get_world_size() > 1:
                utils.check_ddp_consistency(self.model)
            # record loss and time
            metric_logger.update(**loss)
            iter_time.update(time.time() - end_time)
            end_time = time.time()

            # output to logger
            if (step+1) % 100 == 0 or step+1 == max_step:
                eta_seconds = iter_time.global_avg * (max_step - step - 1 + max_step * (max_epoches-epoch-1))
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                self.logger.info(
                    metric_logger.delimiter.join(
                        [header,
                        "lr: {lr}",
                        "eta: {eta}",
                        "time: {time}",
                        "data: {data}",
                        "memory: {memory:.0f}",
                        "{meters}"
                        ]
                    ).format(
                        epoch=epoch+1, max_epoches=max_epoches, step=step+1, max_step=max_step,
                        lr=self.optimizer.param_groups[0]["lr"],
                        eta=eta_string,
                        time=str(iter_time),
                        data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / (1024. * 1024),
                        meters=str(metric_logger)
                    ))
                
    def load_checkpoint(self, checkpoint_path, load_model=True, load_optimizer=True, load_scheduler=True, load_epoch=True, load_metric_best=True):
        if os.path.exists(checkpoint_path):
            checkpoint_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        else:
            self.logger.info("checkpoint is not exist")
            return
        checkpoint_model = checkpoint_dict['model']
        checkpoint_optimizer = checkpoint_dict['optimizer']
        checkpoint_lr_scheduler = checkpoint_dict['lr_scheduler']
        if load_model:
            new_state_dict = OrderedDict()
            for k, v in checkpoint_model.items():
                if "module" == k[:6]:
                    name = k[7:]
                else:
                    name = k
                new_state_dict[name] = v
            self.model.load_state_dict(new_state_dict)
        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint_optimizer)
        if load_scheduler:
            self.lr_scheduler.load_state_dict(checkpoint_lr_scheduler)
        if load_epoch:
            self.begin_epoch = checkpoint_dict['epoch']
        if load_metric_best and 'metric_best' in checkpoint_dict:
            self.metric_best = checkpoint_dict['metric_best']
        if 'amp_scaler' in checkpoint_dict:
            self.gscaler.load_state_dict(checkpoint_dict['amp_scaler'])
        self.logger.info("last epoch:{epoch}, metric best:{metric_best}".format(epoch=checkpoint_dict['epoch'], metric_best=checkpoint_dict['metric_best']))


    def save_checkpoint(self, epoch, checkpoint_savedir, save_type='save_best'): 
        checkpoint_savedir = Path(checkpoint_savedir)
        # checkpoint_path = checkpoint_savedir / '{}'.format('checkpoint_best.pth' \
        #                     if save_type == 'save_best' else 'checkpoint_latest.pth')

        if save_type == "save_best":
            checkpoint_path = checkpoint_savedir / '{}'.format('checkpoint_best.pth')
        # elif epoch==4:
        #     checkpoint_path = checkpoint_savedir / '{}'.format('checkpoint_latest_5.pth')
        # elif epoch==9:
        #     checkpoint_path = checkpoint_savedir / '{}'.format('checkpoint_latest_10.pth')
        else:
            checkpoint_path = checkpoint_savedir / '{}'.format('checkpoint_latest.pth')

    
        # print(save_type, checkpoint_path)

        if utils.get_world_size() > 1 and utils.get_rank() == 0:
            torch.save(
                {
                'epoch':            epoch+1,
                'model':            self.model.module.state_dict(),
                'optimizer':        self.optimizer.state_dict(),
                'lr_scheduler':     self.lr_scheduler.state_dict(),
                'metric_best':      self.metric_best,
                'amp_scaler':       self.gscaler.state_dict(),
                }, checkpoint_path
            )
        elif utils.get_world_size() == 1:
            torch.save(
                {
                'epoch':            epoch+1,
                'model':            self.model.state_dict(),
                'optimizer':        self.optimizer.state_dict(),
                'lr_scheduler':     self.lr_scheduler.state_dict(),
                'metric_best':      self.metric_best,
                'amp_scaler':       self.gscaler.state_dict(),
                }, checkpoint_path
            )


    def whether_save_best(self, metric_logger):
        metric_now = metric_logger.meters[self.save_best_param].global_avg
        if self.metric_best is None:
            self.metric_best = metric_now
            return True
        if metric_now < self.metric_best:
            self.metric_best = metric_now
            return True
        return False



    def trainer(self, train_data_loader, test_data_loader, max_epoches, checkpoint_savedir=None, save_ceph=False, resume=False):
        for epoch in range(self.begin_epoch, max_epoches):

            train_data_loader.sampler.set_epoch(epoch)
            self.train_one_epoch(train_data_loader, epoch, max_epoches)
   
            metric_logger = self.test(test_data_loader, epoch)

            # save model
            if checkpoint_savedir is not None:
                if self.whether_save_best(metric_logger):
                    self.save_checkpoint(epoch, checkpoint_savedir, save_type='save_best')
                if (epoch + 1) % 1 == 0:
                    self.save_checkpoint(epoch, checkpoint_savedir, save_type='save_latest')
            # end_time = time.time()
            # print("save model time", end_time - begin_time2)
        

    @torch.no_grad()
    def test(self, test_data_loader, epoch):
        metric_logger = utils.MetricLogger(delimiter="  ")
        # set model to eval
        self.model.eval()


        for step, batch in enumerate(test_data_loader):
            loss = self.test_one_step(batch)
            metric_logger.update(**loss)
        
        self.logger.info('  '.join(
                [f'Epoch [{epoch + 1}](val stats)',
                 "{meters}"]).format(
                    meters=str(metric_logger)
                 ))

        return metric_logger


    def test_final(self, valid_data_loader, predict_length):
        metric_logger = []
        for i in range(predict_length):
            metric_logger.append(utils.MetricLogger(delimiter="  "))
        # set model to eval
        self.model.eval()

        index = 0
        total_step = len(valid_data_loader)

        for step, batch in enumerate(valid_data_loader):
            batch = torch.stack(batch, 1)
            batch_len = batch.shape[0]
            losses = self.multi_step_predict(batch, index, batch_len)
            for i in range(len(losses)):
                metric_logger[i].update(**losses[i])
            index += batch_len

            self.logger.info("#"*80)
            self.logger.info(step)
            self.writer.add_scalar('Test/RMSE(one step / total)', metric_logger[0].meters["RMSE"].global_avg, step)
            plot_buf = self.eval_metrics.plot_all_var(metric_logger[0].meters, "WRMSE")
            self.writer.add_figure('Test/RMSE(one step / varwise)', plot_buf, step)

            if step % 10 == 0 or step == total_step-1:
                for i in range(predict_length):
                    self.logger.info('  '.join(
                            [f'final valid {i}th step predict (val stats)',
                            "{meters}"]).format(
                                meters=str(metric_logger[i])
                            ))

        return None

    def calculate_q(self, data_loader):
        # set model to eval
        self.model.eval()

        total_step = len(data_loader)
        error_var = np.zeros((69, 128, 256))

        self.logger.info(total_step)

        for step, batch in enumerate(data_loader):
            batch = torch.stack(batch, 1)[0].to(self.device) #[step, 69, 128, 256]
            # self.logger.info("#"*80)
            self.logger.info(step)
            output = self.model(batch[0:1])[:,:69]
            # output = self.model(output)[:,:69]
            # output = (output - batch[2:3])**2
            output = (output - batch[1:2])**2
            error_var += output.squeeze(0).cpu().detach().numpy()

        self.logger.info(total_step)

        return error_var / total_step


    def multi_step_predict(self, batch_data, index, batch_len):
        # last_inp = tensor_data[:inp_length-1].float().to(self.device, non_blocking=True).transpose(0,1).transpose(1,2)
        # pred = tensor_data[inp_length-1:inp_length].float().to(self.device, non_blocking=True).transpose(0,1).transpose(1,2)

        inp = batch_data[:, :1].float().to(self.device, non_blocking=True)
        metrics_losses = []

        print(batch_data.shape) #[1, 21, 69, 128, 256]
        for i in range(1, batch_data.shape[1]):
            tar = batch_data[:, i].float().to(self.device, non_blocking=True)

            pred = self.model(inp.flatten(1,2)) #[1, 69, 128, 256]
            data_dict = {}
            data_dict['gt'] = tar
            data_dict['pred'] = pred[:,:tar.shape[1]]
            data_dict['clim_mean'] = None
            data_dict['std'] = torch.from_numpy(self.std_data).to(self.device)
            metrics_losses.append(self.eval_metrics.evaluate_batch(data_dict))
            inp = pred[:,:tar.shape[1]].unsqueeze(1)
            
        
        return metrics_losses


class vae_nmc_model(nn.Module):
    def __init__(self, logger, param_str, sigma, path_fengwu, path_vae, device, **params) -> None:
        super().__init__()

        self.params = params
        self.device = device
        self.logger = logger

        self.ckpt_fengwu = path_fengwu
        self.ckpt_vae   = path_vae
        self.params = params["params"]["sub_model"]["lgunet_all"]

        self.fengwu = LGUnet_all_1(**self.params).to(self.device)
        self.fengwu.eval()

        self.sigma = sigma

        self.param_str = param_str
        self.vae = VAE_lr(param_str).to(self.device)
        self.vae.train()

        self.err_std = torch.Tensor([0.18955279, 0.22173745, 0.03315084, 0.08258388, 
        0.03021586, 0.0194484 , 0.01700376, 0.01931592, 0.02327741, 0.02647366, 0.02925515, 0.0304862 , 0.03300306, 0.03865351, 0.05609745, 0.0682424 , 0.07762259, 
        0.50658824, 0.29907974, 0.22097995, 0.22990653, 0.26931248, 0.27226337, 0.26211415, 0.24042704, 0.20803592, 0.18460007, 0.12343913, 0.06593712, 0.04856134, 
        0.11308974, 0.11406155, 0.10717956, 0.12138538, 0.14543332, 0.16263002, 0.17114112, 0.16359221, 0.1600293 , 0.16136173, 0.17905815, 0.19142863, 0.18638292, 
        0.13128242, 0.1593278, 0.16516368, 0.17795471, 0.19510655, 0.20854117, 0.21904777, 0.21593404, 0.21397153, 0.21613599, 0.23249907, 0.23790329, 0.21999044, 
        0.06977215, 0.03924686, 0.06015565, 0.11465897, 0.09490499, 0.06113996, 0.05008726, 0.04878271, 0.04601997, 0.04151259, 0.04477754, 0.04275933, 0.03838996]).to(self.device).reshape(1, 69, 1, 1)

        self.load_checkpoint()

    def load_checkpoint(self):
        checkpoint_dict = torch.load(self.ckpt_fengwu)
        checkpoint_model = checkpoint_dict['model']['lgunet_all']
        new_state_dict = OrderedDict()
        for k, v in checkpoint_model.items():
            if "module" == k[:6]:
                name = k[7:]
            else:
                name = k
            if not name == "max_logvar" and not name == "min_logvar":
                new_state_dict[name] = v
        self.fengwu.load_state_dict(new_state_dict)
        if self.ckpt_vae:
            checkpoint_dict = torch.load(self.ckpt_vae, map_location="cuda:0")
            new_state_dict = OrderedDict()
            for k, v in checkpoint_dict.items():
                if "module" == k[:6]:
                    name = k[7:]
                else:
                    name = k
                new_state_dict[name] = v
            # print(new_state_dict)
            self.vae.load_state_dict(new_state_dict)

    def train(self, data_loader, epoch_num=20, lr=1e-4):
        self.fengwu.eval()
        optimizer = optim.Adam(self.vae.parameters(), lr=lr)

        self.logger.info(str(len(data_loader)))
        for i in range(epoch_num):
            self.vae.train()
            for j, batch in enumerate(data_loader):

                batch = torch.stack(batch, 1).to(self.device)  # B x 5 x 69 x 128 x 256
                pred1 = batch[:, 0]
                for ii in range(4):
                    # print(ii)
                    pred1 = self.fengwu(pred1)[:,:69].detach()
                pred1 = pred1.cpu()
                pred2 = batch[:, 4]
                batch = batch.cpu()
                for ii in range(0):
                    pred2 = self.fengwu(pred2)[:,:69].detach()
                pred2 = pred2.cpu()
                # batch = batch.cpu()

                err = (pred2 - pred1) / self.err_std.cpu()

                err = err.to(self.device)
                err = F.interpolate(err, (128, 256))
                
                # np.save("sample", err.cpu().numpy())

                # print(err.shape)
                # print(err)

                # if i == 0 and j == 0:
                #     with torch.no_grad():
                #         log_p, logdet, _ = self.glow.module(err)
                #         continue

                # else:
                recon_batch, mu, log_var = self.vae(err)
                intermediate = self.vae.enc(err).detach().cpu().numpy()
                print(intermediate.shape)
                np.save("intermediate", intermediate)
                print("finishi saving")
                xxx = y

                loss, rec_loss, kld_loss = loss_function(recon_batch, err, mu, log_var, self.sigma)
        

                # print(_)
                # for elem in _:
                #     print(elem.max(), elem.min())

                # loss, log_p, log_det = self.calc_loss(log_p, logdet)
                self.vae.zero_grad()
                loss.backward()
                # warmup_lr = args.lr * min(1, i * batch_size / (50000 * 10))
                warmup_lr = lr
                optimizer.param_groups[0]["lr"] = warmup_lr
                optimizer.step()

                if (j+1) % 10 == 0 or (j+1) == len(data_loader):
                    self.logger.info("epoch: %d iter number: %d loss: %.3f rec loss: %.3f kld loss: %.3f"%(i, j, loss, rec_loss, kld_loss))

                # err = err.cpu()
                # recon_batch = recon_batch.cpu()
                # loss = loss.cpu()
                # rec_loss = rec_loss.cpu()
                # kld_loss = kld_loss.cpu()
                # mu = mu.cpu()
                # log_var = log_var.cpu()
                # torch.cuda.empty_cache()

            self.logger.info("saving model")
            # torch.save(
            #     self.vae.state_dict(), "nf_model/ckpts/vae_nogap_%s_sigma%.2f_epoch%d_19790101_20151231_finetuned.pt" % (self.param_str, self.sigma, i+1)
            # )

            self.logger.info("evaluation")
            self.vae.eval()

            z = torch.randn(8, 32, 128, 256).to(self.device) #* 0.7
            y = self.vae.module.decoder(z) * self.err_std.to(self.device)
            y = y.detach().cpu().numpy()
            # np.save(
            #     "nf_model/samples/vae_nogap_%s_sigma%.2f_epoch%d_19790101_20151231_finetuned" % (self.param_str, self.sigma, i+1), y
            # )
            z = z.cpu()
            torch.cuda.empty_cache()


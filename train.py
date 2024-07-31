# 导入必要的库和模块
import argparse
import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist

# 导入自定义模型和数据集相关的函数和类
from models import model_factory
from data.dataset import available_datasets, get_train_dataloader, get_val_dataloader, dataset
from utils import save_options, get_optim_and_scheduler, set_mode, set_requires_grad, set_lambda

# 导入其他工具和库
import gc
import os
import random
import numpy as np
from logitadjust import LogitAdjust
from proco import ProCoLoss
from collections import Counter
import matplotlib.pyplot as plt

# 导入用于绘制混淆矩阵和其他可视化的库
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm

# 设置默认字体
plt.rcParams['font.sans-serif'] = ['Times New Roman']
# 如果需要显示中文字体，可以取消下面的注释并设置为中文字体
# plt.rcParams['axes.unicode_minus'] = False  # 显示负号
# 设置字体属性
font_prop = fm.FontProperties(family='DejaVu Sans')

# 设置随机种子以保证实验的可重复性
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 解析命令行参数的函数
def get_args():
    parser = argparse.ArgumentParser(description="Script to train DG_VIA_ER", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", choices=["PACS"], help="多域数据集", default="PACS")
    parser.add_argument("--source", choices=available_datasets, help="源域", nargs='+')
    parser.add_argument("--target", choices=available_datasets, help="目标域", default="art_painting")
    parser.add_argument("--batch_size", "-b", type=int, default=64, help="批处理大小")
    parser.add_argument("--image_size", type=int, default=224, help="图像大小")
    parser.add_argument("--data_dir", default="./dataset", help="数据目录")
    parser.add_argument("--datalist_dir", default="./datalist", help="数据列表目录")
    parser.add_argument("--min_scale", default=0.8, type=float, help="最小缩放比例")
    parser.add_argument("--max_scale", default=1.0, type=float, help="最大缩放比例")
    parser.add_argument("--flip", default=0.5, type=float, help="随机水平翻转的概率")
    parser.add_argument("--jitter", default=0.4, type=float, help="颜色抖动量")
    parser.add_argument("--lr", type=float, default=.001, help="主模型的学习率")
    parser.add_argument("--tem", type=float, default=0.1, help="温度参数")
    parser.add_argument("--tau", type=float, default=1, help="tau参数")
    parser.add_argument("--lr_c", type=float, default=.0001, help="分类器T_i的学习率")
    parser.add_argument("--lr_cp", type=float, default=.0001, help="分类器T_i^'的学习率")
    parser.add_argument("--lr_d", type=float, default=.001, help="判别器的学习率")
    parser.add_argument("--lbd_c", type=float, default=0.05, help="分类器T_i的权重")
    parser.add_argument("--lbd_cp", type=float, default=0.001, help="分类器T_i^' (GRL)的权重")
    parser.add_argument("--lbd_d", type=float, default=0.1, help="判别器(GRL)的权重")
    parser.add_argument("--contrast_dim", type=int, default=512, help="对比学习的维度")
    parser.add_argument("--epochs", type=int, default=100, help="训练的轮数")
    parser.add_argument("--lr_steps", type=int, default=60, nargs='+', help='学习率衰减的步长')
    parser.add_argument("--lr_gamma", type=float, default=0.1, help='学习率衰减的乘数因子')
    parser.add_argument("--warmup_step", type=int, default=10, help='预热步数')
    parser.add_argument("--warmup_weight", type=float, default=0.01, help='预热权重')
    parser.add_argument("--num_classes", "-c", type=int, default=7, help="类别数")
    parser.add_argument("--network", choices=model_factory.nets_map.keys(), default="resnet18", help="选择使用的网络")
    parser.add_argument("--exp_folder", default="experiments", help="保存日志和模型的目录")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()

# 训练类
class Trainer:
    def __init__(self, args):
        # 设置随机种子
        set_random_seed(args.seed)
        # 根据数据集和目标域确定源域
        args.source = [d for d in dataset[args.dataset] if d != args.target]
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 初始化主模型、判别器和分类器
        main_model, dis_model, c_model, cp_model = model_factory.get_network(args.network)(
            num_classes=args.num_classes, num_domains=len(args.source)
        )
        self.main_model = self._model2device(main_model)
        self.dis_model = self._model2device(dis_model)
        self.c_model = self._model2device(c_model)
        self.cp_model = self._model2device(cp_model)
        self.batch_size = self.args.batch_size

        # 获取训练和验证数据加载器
        self.source_loader_list, self.val_loader, self.img_num_per_domain = get_train_dataloader(args)
        self.target_loader = get_val_dataloader(args)
        self.test_loaders = {"val": self.val_loader, "test": self.target_loader}

        # 输出数据集大小信息
        print("Dataset size: train %d, val %d, test %d" % (sum(self.img_num_per_domain), len(self.val_loader.dataset), len(self.target_loader.dataset)))

        # 获取优化器和学习率调度器
        self.optimizer, self.scheduler, self.classifier_optimizer = get_optim_and_scheduler(
            self.main_model, self.dis_model, args.lr, args.lr_d, epochs=args.epochs, lr_steps=args.lr_steps, gamma=args.lr_gamma
        )
        self.num_classes = args.num_classes
        self.num_domains = len(args.source)

        # 设置保存路径并创建目录
        self.base_dir = os.path.join(args.exp_folder, args.network, args.dataset)
        self.save_dir = os.path.join(self.base_dir, args.target)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        save_options(args, self.save_dir)
        
        # 初始化日志文件路径
        self.log_file = os.path.join(self.save_dir, "loss_log.txt")
        self.log_file1 = os.path.join(self.save_dir, "val_log.txt")
        self.log_file2 = os.path.join(self.save_dir, "test_log.txt")
        self.log_file3 = os.path.join(self.save_dir, "loss.txt")
        
        # 计算类别数量列表
        self.cls_num_list = self._compute_cls_num_list()

        # 初始化LogitAdjust和ProCoLoss损失函数
        self.criterion_ce = LogitAdjust(cls_num_list=self.cls_num_list, tau=args.tau).cuda()
        self.criterion_scl = ProCoLoss(contrast_dim=args.contrast_dim, temperature=args.tem, num_classes=self.num_classes).cuda()

    # 计算每个类别的样本数量
    def _compute_cls_num_list(self):
        cls_num_list = [0] * self.num_classes
        for loader in self.source_loader_list:
            for _, labels in loader:
                labels = labels.numpy()
                cls_counts = Counter(labels)
                for cls, count in cls_counts.items():
                    cls_num_list[cls] += count
        return cls_num_list
    
    # 将模型放置到设备上（CPU或GPU）
    def _model2device(self, model):
        if model is None:
            return None
        model.to(self.device)
        return model

    # 计算判别器的损失
    def _compute_dis_loss(self, feature, domains):
        if self.dis_model is not None:
            domain_logit = self.dis_model(feature)
            weight = [1.0 / img_num for img_num in self.img_num_per_domain]
            weight = torch.FloatTensor(weight).to(self.device)
            weight = weight / weight.sum() * self.num_domains
            domain_loss = F.cross_entropy(domain_logit, domains, weight=weight)
        else:
            domain_loss = torch.zeros(1, requires_grad=True).to(self.device)
        return domain_loss

    # 计算分类器的损失
    def _compute_cls_loss(self, model, feature, label, domain, mode="self"):
        if model is not None:
            feature_list = []
            label_list = []
            weight_list = []
            for i in range(self.num_domains):
                if mode == "self":
                    feature_list.append(feature[domain == i])
                    label_list.append(label[domain == i])
                else:
                    feature_list.append(feature[domain != i])
                    label_list.append(label[domain != i])
                weight = torch.zeros(self.num_classes).to(self.device)
                for j in range(self.num_classes):
                    weight[j] = 0 if (label_list[-1] == j).sum() == 0 else 1.0 / (label_list[-1] == j).sum().float()
                weight = weight / weight.sum()
                weight_list.append(weight)
            class_logit = model(feature_list)
            loss = 0
            for p, l, w in zip(class_logit, label_list, weight_list):
                if p is None:
                    continue
                loss += F.cross_entropy(p, l, weight=w) / self.num_domains
        else:
            loss = torch.zeros(1, requires_grad=True).to(self.device)
        return loss

    # 计算包含平方损失项的分类器损失
    def _compute_CLS_SLD_loss(self, model, feature, label, domain, batch_size=64, mode="self"):
        if model is not None:
            feature_list = []
            label_list = []
            weight_list = []
            for i in range(self.num_domains):
                if mode == "self":
                    feature_list.append(feature[domain == i])
                    label_list.append(label[domain == i])
                else:
                    feature_list.append(feature[domain != i])
                    label_list.append(label[domain != i])
                weight = torch.zeros(self.num_classes).to(self.device)
                for j in range(self.num_classes):
                    weight[j] = 0 if (label_list[-1] == j).sum() == 0 else 1.0 / (label_list[-1] == j).sum().float()
                weight = weight / weight.sum()
                weight_list.append(weight)
            class_logit = model(feature_list)
            loss = 0
            for p, l, w in zip(class_logit, label_list, weight_list):
                if p is None:
                    continue
                loss += (F.cross_entropy(p, l, weight=w) + torch.sum((torch.pow(p, 2))) / batch_size) / self.num_domains
        else:
            loss = torch.zeros(1, requires_grad=True).to(self.device)
        return loss

    # 执行一个训练轮次的操作
    def _do_epoch(self):
        # 设置模型为训练模式
        set_mode(self.main_model, "train")
        set_mode(self.dis_model, "train")
        set_mode(self.c_model, "train")
        set_mode(self.cp_model, "train")

        # 设置判别器和分类器的梯度更新权重
        set_lambda([self.dis_model], [self.args.lbd_d])
        set_lambda([self.c_model, self.cp_model], [self.args.lbd_c, self.args.lbd_cp])
        loader_iter_list = []
        loader_size_list = []

        # 根据当前轮次确定预热权重
        if self.current_epoch < self.args.warmup_step:
            aux_weight = self.args.warmup_weight
            main_weight = self.args.warmup_weight
        else:
            aux_weight = 1
            main_weight = 1

        # 初始化数据加载器的迭代器
        for loader in self.source_loader_list:
            loader_iter_list.append(enumerate(loader))
            loader_size_list.append(len(loader))

        # 迭代数据进行训练
        for it in range(max(loader_size_list)):
            data = []
            labels = []
            domains = []
            for idx, iter_ in zip(range(self.num_domains), loader_iter_list):
                try:
                    item = iter_.__next__()
                except StopIteration:
                    loader_iter_list[idx] = enumerate(self.source_loader_list[idx])
                    item = loader_iter_list[idx].__next__()
                data.append(item[1][0])
                labels.append(item[1][1])
                domains.append(torch.ones(labels[-1].size(0)).long() * idx)

            data = torch.cat(data, dim=0).to(self.device)
            labels = torch.cat(labels, dim=0).to(self.device)
            domains = torch.cat(domains, dim=0).to(self.device)

            # 冻结主模型，更新分类器
            set_requires_grad(self.main_model, False)
            set_requires_grad(self.c_model, True)
            _, feature = self.main_model(data)
            c_loss_self = self._compute_CLS_SLD_loss(self.c_model, feature.detach(), labels, domains, self.batch_size, mode="self") * aux_weight
            self.optimizer.zero_grad()
            c_loss_self.backward()
            self.optimizer.step()

            # 恢复所有模型的梯度更新，继续训练
            set_requires_grad([self.main_model, self.dis_model, self.c_model, self.cp_model], True)
            self.classifier_optimizer.zero_grad()
            self.optimizer.zero_grad()
            class_logit, feature = self.main_model(data)
            main_loss = F.cross_entropy(class_logit, labels) * main_weight

            main_loss.backward()
            self.optimizer.zero_grad()
            self.optimizer.step()
            self.classifier_optimizer.first_step(zero_grad=True)

            # 第二步优化
            class_logit, feature = self.main_model(data)
            main_loss = F.cross_entropy(class_logit, labels) * main_weight
            dis_loss = self._compute_dis_loss(feature, domains) * aux_weight

            # 计算对比学习损失
            current_batch_size = feature.size(0)
            if current_batch_size % self.batch_size == 0 and current_batch_size % 3 == 0:
                split_sizes = [self.batch_size, self.batch_size, self.batch_size]
            else:
                remain = current_batch_size - self.batch_size * 2
                split_sizes = [self.batch_size, self.batch_size, remain]

            f1, f2, f3 = torch.split(feature, split_sizes, dim=0)
            lf1, lf2, lf3 = torch.split(labels, split_sizes, dim=0)

            contrast_logits1 = self.criterion_scl(f1, lf1)
            contrast_logits2 = self.criterion_scl(f2, lf2)
            contrast_logits = (contrast_logits1 + contrast_logits2) / 2

            scl_loss = (self.criterion_ce(contrast_logits1, lf1) + self.criterion_ce(contrast_logits2, lf2)) / 2

            set_requires_grad(self.c_model, False)
            c_loss_others = self._compute_cls_loss(self.c_model, feature, labels, domains, mode="others") * aux_weight

            cp_loss = self._compute_cls_loss(self.cp_model, feature, labels, domains, mode="self") * aux_weight

            # 计算总损失并更新模型
            loss = dis_loss + main_loss + scl_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.classifier_optimizer.second_step(zero_grad=True)

            loss += c_loss_self

            # 日志记录损失值
            message = "epoch %d iter %d: all %.6f main %.6f dis %.6f c_self %.6f c_others %.6f cp %.6f\n" % (
                self.current_epoch, it, loss.data, main_loss.data, dis_loss.data, c_loss_self.data, c_loss_others.data, cp_loss.data
            )

            with open(self.log_file, "a") as fid:
                fid.write(message)
            with open(self.log_file3, "a") as fid3:
                fid3.write("%.6f\n" % (main_loss.data))
            print(message)

            # 释放不需要的变量以节省内存
            del loss, main_loss, dis_loss, c_loss_self, c_loss_others, cp_loss

        # 验证模型
        self.main_model.eval()
        with torch.no_grad():
            with open(self.log_file, "a") as fid:
                for phase, loader in self.test_loaders.items():
                    class_correct, all_domains, conf_matrix = self.do_test(loader)
                    class_correct = class_correct.float()
                    class_acc = class_correct.mean() * 100.0
                    self.results[phase][self.current_epoch] = class_acc
                    if phase == "val":
                        message = "epoch %d: val_all_acc %.5f" % (self.current_epoch, class_acc)
                        with open(self.log_file1, "a") as fid1:
                            for i in range(self.num_domains):
                                cc_i = class_correct[all_domains == i]
                                ca_i = cc_i.mean() * 100.0
                                message += " val_%s_acc %.5f" % (self.args.source[i], ca_i)
                                fid1.write("%.6f," % (ca_i))
                            fid1.write('\n')
                        message += "\n"
                        fid.write(message)
                        print(message)
                    elif phase == "test":
                        message = "epoch %d: test_acc %.5f\n" % (self.current_epoch, class_acc)
                        fid.write(message)
                        print(message)
                        with open(self.log_file2, "a") as fid2:
                            fid2.write("%.6f\n" % (class_acc))

    # 执行模型测试
    def do_test(self, loader):
        class_correct = []
        all_domains = []
        all_labels = []
        all_preds = []

        for _, ((data, labels), domains) in enumerate(loader):
            data, labels, domains = data.to(self.device), labels.to(self.device), domains.to(self.device)
            class_logit, _ = self.main_model(data)

            _, cls_pred = class_logit.max(dim=1)
            class_correct.append(cls_pred == labels.data)
            all_domains.append(domains)
            all_labels.append(labels.data)
            all_preds.append(cls_pred)

        all_labels = torch.cat(all_labels, 0).cpu().numpy()
        all_preds = torch.cat(all_preds, 0).cpu().numpy()

        # 计算混淆矩阵
        conf_matrix = confusion_matrix(all_labels, all_preds)
        return torch.cat(class_correct, 0), torch.cat(all_domains, 0), conf_matrix

    # 执行整个训练过程
    def do_training(self):
        self.results = {"val": torch.zeros(self.args.epochs), "test": torch.zeros(self.args.epochs)}

        for self.current_epoch in range(self.args.epochs):
            self._do_epoch()
            self.scheduler.step()
            _, _, conf_matrix = self.do_test(self.test_loaders["test"])

        val_res = self.results["val"]
        test_res = self.results["test"]
        idx_best = val_res.argmax()
        message = "Best val %.5f (epoch: %d), corresponding test %.5f\n" % (val_res.max(), idx_best, test_res[idx_best])
        print(message)
        with open(self.log_file, "a") as fid:
            fid.write(message)

# 程序的入口，解析命令行参数并开始训练
if __name__ == "__main__":
    args = get_args()
    trainer = Trainer(args)
    trainer.do_training()

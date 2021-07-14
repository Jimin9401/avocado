from tqdm import tqdm
from torch.utils.data import IterableDataset, DataLoader
import torch
import apex
from model.losses import NTXentLoss,AlignLoss
from model.classification_model import PretrainedTransformer
from overrides import overrides
from sklearn.metrics import classification_report,f1_score,accuracy_score
from itertools import chain

class Trainer:
    def __init__(self, args, model: PretrainedTransformer, train_batchfier, test_batchfier, optimizers,
                 update_step, criteria, clip_norm, mixed_precision):
        self.args = args
        self.model = model
        self.train_batchfier = train_batchfier
        self.test_batchfier = test_batchfier
        self.optimizers = optimizers
        self.criteria = criteria
        self.step = 0
        self.update_step = update_step
        self.mixed_precision = mixed_precision
        self.clip_norm = clip_norm

    def reformat_inp(self, inp):
        raise NotImplementedError

    def train_epoch(self):
        return NotImplementedError

    def test_epoch(self):
        return NotImplementedError


class CFTrainer(Trainer):
    def __init__(self, args, model, train_batchfier, test_batchfier, optimizers,
                 update_step, criteria, clip_norm, mixed_precision, n_label):
        super(CFTrainer, self).__init__(args, model, train_batchfier, test_batchfier, optimizers,
                                        update_step, criteria, clip_norm, mixed_precision)
        self.n_label = n_label

    @overrides
    def reformat_inp(self, inp):

        inp_tensor = tuple(i.to("cuda") for i in inp)

        return inp_tensor

    def train_epoch(self):

        model = self.model
        batchfier = self.train_batchfier
        criteria = self.criteria
        optimizer = self.optimizers

        if isinstance(batchfier, IterableDataset):
            batchfier = DataLoader(dataset=batchfier,
                                   batch_size=batchfier.size,
                                   shuffle=False,
                                   num_workers=1,
                                   collate_fn=batchfier.collate, pin_memory=True)

        model.train()
        tot_loss, step_loss, tot_cnt, n_bar, acc = 0, 0, 0, 0, 0
        pbar_cnt = 0
        model.zero_grad()
        pbar = tqdm(batchfier, total=batchfier.dataset.num_buckets)

        for inp in pbar:
            inp,attn_mask,gt = self.reformat_inp(inp)

            logits, _ = model(inp,attn_mask)

            loss = criteria(logits.view(-1,logits.size(-1)), gt.view(-1))

            step_loss += loss.item()
            tot_loss += loss.item()
            tot_cnt += 1

            if self.mixed_precision:
                with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if not tot_cnt % self.update_step:
                self.step += 1
                pbar_cnt += 1
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_norm)
                optimizer.step()
                # scheduler.step(self.step)
                model.zero_grad()
                pbar.set_description(
                    "training loss : %f  , iter : %d" % (
                        step_loss / (self.update_step * pbar_cnt),
                         n_bar), )
                pbar.update()
                # if pbar_cnt == 100:
                #     pbar, n_bar, pbar_cnt, step_loss, acc = reset_pbar(pbar, n_bar)

        pbar.close()

    def test_epoch(self):
        model = self.model
        batchfier = self.test_batchfier

        if self.args.dataset=="bio_ner":
            batchfier.collate=batchfier.collate_ner


        if isinstance(self.criteria, tuple):
            _, criteria = self.criteria
        else:
            criteria = self.criteria

        if isinstance(batchfier, IterableDataset):
            batchfier = DataLoader(dataset=batchfier,
                                   batch_size=batchfier.size,
                                   shuffle=False,
                                   collate_fn=batchfier.collate, pin_memory=True)

        model.eval()
        model.zero_grad()
        pbar = tqdm(batchfier)
        pbar_cnt = 0
        step_loss = 0
        tot_score = 0.0

        true_buff = []
        eval_buff = []

        for inp in pbar:
            with torch.no_grad():
                inp = self.reformat_inp(inp)
                gt = inp[-1].view(-1)
                logits, _ = model(inp[0])
                preds = torch.argmax(logits, -1)
                preds = preds.view(-1)
                loss = criteria(logits.view(-1, logits.size(-1)), inp[-1].view(-1))
                step_loss += loss.item()
                pbar_cnt += 1

            true_buff.append(gt.tolist())
            eval_buff.append(preds.tolist())
            score = torch.mean((preds == gt).to(torch.float))
            tot_score += score

            pbar.set_description(
                "test loss : %f  test accuracy : %f" % (
                    step_loss / pbar_cnt, tot_score / pbar_cnt), )
            pbar.update()
        pbar.close()
        true_buff= list(chain(*true_buff))
        eval_buff = list(chain(*eval_buff))
        accuracy = accuracy_score(true_buff,eval_buff)
        print()
        if self.args.dataset =="chemprot":
            f1 = f1_score(true_buff,eval_buff,labels=list(range(0,self.n_label)),average="micro")
        else :
            f1 = f1_score(true_buff,eval_buff,labels=list(range(0,self.n_label)),average="macro")

        print("test accuracy: {0:.4f}  test f1: {1:.4f}".format(accuracy, f1))

        return accuracy, f1



class ContrastiveTrainer(Trainer):
    def __init__(self, args, model, train_batchfier, test_batchfier, optimizers,
                 update_step, criteria, clip_norm, mixed_precision, n_label):
        super(ContrastiveTrainer, self).__init__(args, model, train_batchfier, test_batchfier, optimizers,
                                                 update_step, criteria, clip_norm, mixed_precision)
        self.n_label = n_label
        if args.align_type=="cosine":
            self.align_loss = AlignLoss(args, args.per_gpu_train_batch_size, temperature=1.0)
        elif args.align_type=="simclr":
            self.align_loss = NTXentLoss(args, args.per_gpu_train_batch_size,temperature=args.temperature)
        else:
            print(args.align_type)
            raise NotImplementedError
    @overrides
    def reformat_inp(self, inp):
        inp_tensor = tuple(i.to("cuda") for i in inp)
        return inp_tensor

    def train_epoch(self):

        model = self.model
        batchfier = self.train_batchfier
        criteria = self.criteria
        optimizer = self.optimizers

        if self.args.dataset=="bio_ner":
            batchfier.collate=batchfier.collate_ner


        if isinstance(batchfier, IterableDataset):
            batchfier = DataLoader(dataset=batchfier,
                                   batch_size=batchfier.size,
                                   shuffle=False,
                                   num_workers=1,
                                   collate_fn=batchfier.collate, pin_memory=True,drop_last=True)

        # cached_data_loader=get_cached_data_loader(batchfier,batchfier.size,custom_collate=batchfier.collate,shuffle=False)
        model.train()
        tot_loss, step_loss, tot_cnt, n_bar, acc = 0, 0, 0, 0, 0

        pbar_cnt = 0
        model.zero_grad()
        pbar = tqdm(batchfier, total=batchfier.dataset.num_buckets)

        for inp in pbar:
            inp = self.reformat_inp(inp)

            _, hi = model(inp[0])  # Bert Tokens
            # with torch.no_grad():
            logits, hj = model(inp[1]) # Domain Tokens

            loss_align = self.align_loss(hi, hj)
            loss_ce = self.criteria(logits.view(-1,logits.size(-1)), inp[-1].view(-1))

            loss = loss_align + loss_ce
            step_loss += loss.item()
            tot_loss += loss.item()
            tot_cnt += 1

            if self.mixed_precision:
                with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if not tot_cnt % self.update_step:
                self.step += 1
                pbar_cnt += 1
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_norm)
                optimizer.step()
                # scheduler.step(self.step)
                model.zero_grad()
                pbar.set_description(
                    "training loss : %f , iter : %d" % (
                        step_loss / (self.update_step * pbar_cnt),
                         n_bar), )
                pbar.update()
                # if pbar_cnt == 100:
                #     pbar, n_bar, pbar_cnt, step_loss, acc = reset_pbar(pbar, n_bar)

        pbar.close()

    def test_epoch(self):
        model = self.model
        batchfier = self.test_batchfier

        if self.args.dataset=="bio_ner":
            batchfier.collate=batchfier.collate_ner

        if isinstance(self.criteria, tuple):
            _, criteria = self.criteria
        else:
            criteria = self.criteria

        if isinstance(batchfier, IterableDataset):
            batchfier = DataLoader(dataset=batchfier,
                                   batch_size=batchfier.size,
                                   shuffle=False,
                                   collate_fn=batchfier.collate, pin_memory=True)

        model.eval()
        # cached_data_loader=get_cached_data_loader(batchfier,batchfier.size,custom_collate=batchfier.collate,shuffle=False)
        model.zero_grad()
        pbar = tqdm(batchfier)
        pbar_cnt = 0
        step_loss = 0
        tot_score = 0.0

        true_buff = []
        eval_buff = []

        for inp in pbar:
            with torch.no_grad():
                inp = self.reformat_inp(inp)
                gt = inp[-1].view(-1)
                logits, _ = model(inp[1])
                preds = torch.argmax(logits, -1)
                preds = preds.view(-1)
                loss = criteria(logits.view(-1,logits.size(-1)), gt)
                step_loss += loss.item()
                pbar_cnt += 1

            true_buff.append(gt.tolist())
            eval_buff.append(preds.tolist())

            score = torch.mean((preds == gt).to(torch.float))
            tot_score += score

            pbar.set_description(
                "test loss : %f  test accuracy : %f" % (
                    step_loss / pbar_cnt, tot_score / pbar_cnt), )
            pbar.update()
        pbar.close()

        true_buff= list(chain(*true_buff))
        eval_buff = list(chain(*eval_buff))
        accuracy = accuracy_score(true_buff,eval_buff)
        if self.args.dataset =="chemprot":
            f1 = f1_score(true_buff,eval_buff,labels=list(range(0,self.n_label)),average="micro")
        else :
            f1 = f1_score(true_buff,eval_buff,labels=list(range(0,self.n_label)),average="macro")

        print()
        print("test accuracy: {0:.4f}  test f1: {1:.4f}".format(accuracy, f1))

        return accuracy, f1



class CFTrainer(Trainer):
    def __init__(self, args, model, train_batchfier, test_batchfier, optimizers,
                 update_step, criteria, clip_norm, mixed_precision, n_label):
        super(CFTrainer, self).__init__(args, model, train_batchfier, test_batchfier, optimizers,
                                        update_step, criteria, clip_norm, mixed_precision)
        self.n_label = n_label

    @overrides
    def reformat_inp(self, inp):

        inp_tensor = tuple(i.to("cuda") for i in inp)

        return inp_tensor

    def train_epoch(self):

        model = self.model
        batchfier = self.train_batchfier
        criteria = self.criteria
        optimizer = self.optimizers

        if isinstance(batchfier, IterableDataset):
            batchfier = DataLoader(dataset=batchfier,
                                   batch_size=batchfier.size,
                                   shuffle=False,
                                   num_workers=1,
                                   collate_fn=batchfier.collate, pin_memory=True)

        model.train()
        tot_loss, step_loss, tot_cnt, n_bar, acc = 0, 0, 0, 0, 0
        pbar_cnt = 0
        model.zero_grad()
        pbar = tqdm(batchfier, total=batchfier.dataset.num_buckets)

        for inp in pbar:
            inp,attn_mask,gt = self.reformat_inp(inp)

            logits, _ = model(inp,attn_mask)

            loss = criteria(logits.view(-1,logits.size(-1)), gt.view(-1))

            step_loss += loss.item()
            tot_loss += loss.item()
            tot_cnt += 1

            if self.mixed_precision:
                with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if not tot_cnt % self.update_step:
                self.step += 1
                pbar_cnt += 1
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_norm)
                optimizer.step()
                # scheduler.step(self.step)
                model.zero_grad()
                pbar.set_description(
                    "training loss : %f  , iter : %d" % (
                        step_loss / (self.update_step * pbar_cnt),
                         n_bar), )
                pbar.update()
                # if pbar_cnt == 100:
                #     pbar, n_bar, pbar_cnt, step_loss, acc = reset_pbar(pbar, n_bar)

        pbar.close()

    def test_epoch(self):
        model = self.model
        batchfier = self.test_batchfier

        if self.args.dataset=="bio_ner":
            batchfier.collate=batchfier.collate_ner


        if isinstance(self.criteria, tuple):
            _, criteria = self.criteria
        else:
            criteria = self.criteria

        if isinstance(batchfier, IterableDataset):
            batchfier = DataLoader(dataset=batchfier,
                                   batch_size=batchfier.size,
                                   shuffle=False,
                                   collate_fn=batchfier.collate, pin_memory=True)

        model.eval()
        model.zero_grad()
        pbar = tqdm(batchfier)
        pbar_cnt = 0
        step_loss = 0
        tot_score = 0.0

        true_buff = []
        eval_buff = []

        for inp in pbar:
            with torch.no_grad():
                inp = self.reformat_inp(inp)
                gt = inp[-1].view(-1)
                logits, _ = model(inp[0])
                preds = torch.argmax(logits, -1)
                preds = preds.view(-1)
                loss = criteria(logits.view(-1, logits.size(-1)), inp[-1].view(-1))
                step_loss += loss.item()
                pbar_cnt += 1

            true_buff.append(gt.tolist())
            eval_buff.append(preds.tolist())
            score = torch.mean((preds == gt).to(torch.float))
            tot_score += score

            pbar.set_description(
                "test loss : %f  test accuracy : %f" % (
                    step_loss / pbar_cnt, tot_score / pbar_cnt), )
            pbar.update()
        pbar.close()
        true_buff= list(chain(*true_buff))
        eval_buff = list(chain(*eval_buff))
        accuracy = accuracy_score(true_buff,eval_buff)
        print()
        if self.args.dataset =="chemprot":
            f1 = f1_score(true_buff,eval_buff,labels=list(range(0,self.n_label)),average="micro")
        else :
            f1 = f1_score(true_buff,eval_buff,labels=list(range(0,self.n_label)),average="macro")

        print("test accuracy: {0:.4f}  test f1: {1:.4f}".format(accuracy, f1))

        return accuracy, f1



class InexpensiveTrainer(CFTrainer):
    def __init__(self, args, model, train_batchfier, test_batchfier, optimizers,
                 update_step, criteria, clip_norm, mixed_precision, n_label):
        super(InexpensiveTrainer,self).__init__(args, model, train_batchfier, test_batchfier, optimizers,
                     update_step, criteria, clip_norm, mixed_precision, n_label)

    @overrides
    def reformat_inp(self, inp):
        inp_tensor = tuple(i.to("cuda") for i in inp)
        return inp_tensor

    def train_epoch(self):

        model = self.model
        batchfier = self.train_batchfier
        criteria = self.criteria
        optimizer = self.optimizers

        if isinstance(batchfier, IterableDataset):
            batchfier = DataLoader(dataset=batchfier,
                                   batch_size=batchfier.size,
                                   shuffle=False,
                                   num_workers=1,
                                   collate_fn=batchfier.collate, pin_memory=True)

        model.train()
        tot_loss, step_loss, tot_cnt, n_bar, acc = 0, 0, 0, 0, 0
        pbar_cnt = 0
        model.zero_grad()
        pbar = tqdm(batchfier, total=batchfier.dataset.num_buckets)

        for inp in pbar:
            inp,attn_mask,gt = self.reformat_inp(inp)

            logits, _ = model(inp[0],attn_mask)
            bert_logits = model(inp[0], attn_mask)
            domain_logits = model(inp[1], attn_mask)
            logits = (bert_logits+domain_logits)/2

            loss = criteria(logits.view(-1,logits.size(-1)), gt.view(-1))
            step_loss += loss.item()
            tot_loss += loss.item()
            tot_cnt += 1

            if self.mixed_precision:
                with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if not tot_cnt % self.update_step:
                self.step += 1
                pbar_cnt += 1
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_norm)
                optimizer.step()
                # scheduler.step(self.step)
                model.zero_grad()
                pbar.set_description(
                    "training loss : %f  , iter : %d" % (
                        step_loss / (self.update_step * pbar_cnt),
                         n_bar), )
                pbar.update()
                # if pbar_cnt == 100:
                #     pbar, n_bar, pbar_cnt, step_loss, acc = reset_pbar(pbar, n_bar)

        pbar.close()

    def test_epoch(self):
        model = self.model
        batchfier = self.test_batchfier

        if self.args.dataset=="bio_ner":
            batchfier.collate=batchfier.collate_ner


        if isinstance(self.criteria, tuple):
            _, criteria = self.criteria
        else:
            criteria = self.criteria

        if isinstance(batchfier, IterableDataset):
            batchfier = DataLoader(dataset=batchfier,
                                   batch_size=batchfier.size,
                                   shuffle=False,
                                   collate_fn=batchfier.collate, pin_memory=True)

        model.eval()
        model.zero_grad()
        pbar = tqdm(batchfier)
        pbar_cnt = 0
        step_loss = 0
        tot_score = 0.0

        true_buff = []
        eval_buff = []

        for inp in pbar:
            with torch.no_grad():
                inp = self.reformat_inp(inp)
                gt = inp[-1].view(-1)
                logits, _ = model(inp[0])
                preds = torch.argmax(logits, -1)
                preds = preds.view(-1)
                loss = criteria(logits.view(-1, logits.size(-1)), inp[-1].view(-1))
                step_loss += loss.item()
                pbar_cnt += 1

            true_buff.append(gt.tolist())
            eval_buff.append(preds.tolist())
            score = torch.mean((preds == gt).to(torch.float))
            tot_score += score

            pbar.set_description(
                "test loss : %f  test accuracy : %f" % (
                    step_loss / pbar_cnt, tot_score / pbar_cnt), )
            pbar.update()
        pbar.close()
        true_buff= list(chain(*true_buff))
        eval_buff = list(chain(*eval_buff))
        accuracy = accuracy_score(true_buff,eval_buff)
        print()
        if self.args.dataset =="chemprot":
            f1 = f1_score(true_buff,eval_buff,labels=list(range(0,self.n_label)),average="micro")
        else :
            f1 = f1_score(true_buff,eval_buff,labels=list(range(0,self.n_label)),average="macro")

        print("test accuracy: {0:.4f}  test f1: {1:.4f}".format(accuracy, f1))

        return accuracy, f1

import numpy as np
import pandas as pd



def dataset_to_feature(dataset:list,origin_vocab_size:int):

    domain_size=[]

    for row in dataset:
        domain_size.append([r for r in row if row > origin_vocab_size ])

    df=pd.DataFrame({"text": dataset, "domain_text": domain_size})

    df["origin_length"] = [len(t) for t in df["text"].to_list()]
    df["domain_length"] = [len(t) for t in df["domain_text"].to_list()]
    df["ratio"] = df["origin_length"] / df["domain_length"]

    return df



def domain_statistics(batchfier,padding_index,origin_vocab_size:int):

    dataset = []
    for inps in batchfier:
        dataset.extend(inps[0].tolist())
    dataset = [d[d.find(padding_index)] for d in dataset]
    df = dataset_to_feature(dataset,origin_vocab_size)


class Trainer:
    def __init__(self, model, loss_func, optimizer,metrics,word2id,device):
        self.model=model
        self.loss_func=loss_func
        self.optimizer = optimizer
        self.metrics=metrics
        self.device=device
        self.word2id = word2id
        self.id2word = {v:k for k,v in word2id.items()}


    def reformat_inp(self,inps):
        return [inp.to(self.device) for inp in inps]

    def train(self,data_loader):
        self.model.train()

        pbar = tqdm(data_loader["train"], desc="training...")

        step=0
        step_loss = 0.0

        for inps in pbar:
            inps = self.reformat_inp(inps)
            input_ids, label = inps[0], inps[-1]
            start_logits, end_logits = self.model(input_ids)
            loss = self.loss_func(start_logits, label[:, 0]) + self.loss_func(end_logits, label[:, 1])
            step_loss+=loss
            step+=1
            loss.backward()  # compute gradients
            self.optimizer.step()  # update parameters
            self.optimizer.zero_grad()  # reset process
            pbar.set_description(
                "training loss : %f " % (step_loss/step))
            pbar.update()


    @torch.no_grad()
    def evaluate(self,data_loader, ):
        self.model.eval()

        contexts = torch.LongTensor()
        results = torch.LongTensor()
        gt = torch.LongTensor()

        pbar = tqdm(data_loader["train"], desc="training...")

        for inps in pbar:
            inps = self.reformat_inp(inps)
            input_ids, label = inps[0], inps[1]
            start_logits, end_logits = self.model(input_ids)
            loss = self.loss_func(start_logits, label[:, 0]) + self.loss_func(end_logits, label[:, 1])


            contexts = torch.cat([contexts,input_ids],dim=0)
            preds = torch.argmax(logits, dim=1)
            results = torch.cat([results, preds.cpu()],dim=1)
            gt = torch.cat([gt, label.cpu()],dim=1)

        examples = {"contexts": contexts, "results": results, "gt":gt }
        outs = self.reconstruct(examples)

        return

    def measure(self,examples):

        return [self.metrics.compute(example) for example in examples]



    def reconstruct(self,examples,references):

        contexts = examples["contexts"].tolist()
        preds = examples["results"].tolist()
        outs = []

        for context,pred,reference in zip(contexts,preds,references):
            hypothesis = context[pred[0]:pred[1]+1]
            hypothesis = " ".join([self.id2word[token] for token in hypothesis])

            outs.append({"hypthesis":hypothesis,"reference":reference})

        return outs
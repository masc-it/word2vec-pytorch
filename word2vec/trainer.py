from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from word2vec.data_reader import DataReader, Word2vecDataset
from word2vec.model import SkipGramModel
import sys


class Word2VecTrainer:
    def __init__(self, input_file, output_file, emb_dimension=100, batch_size=32, window_size=5, iterations=10,
                 initial_lr=0.001, min_count=12):

        self.data = DataReader(input_file, min_count)
        dataset = Word2vecDataset(self.data, window_size)
        self.dataloader = DataLoader(dataset, batch_size=batch_size,
                                     shuffle=False, num_workers=0, collate_fn=dataset.collate)

        self.output_file_name = output_file
        self.emb_size = len(self.data.word2id)
        self.emb_dimension = emb_dimension
        self.batch_size = batch_size
        self.iterations = iterations
        self.initial_lr = initial_lr
        self.skip_gram_model = SkipGramModel(self.emb_size, self.emb_dimension)

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        if self.use_cuda:
            self.skip_gram_model.cuda()

    def train(self):

        for iteration in range(self.iterations):

            optimizer = optim.SparseAdam(
                self.skip_gram_model.parameters(), lr=self.initial_lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, len(self.dataloader))

            running_loss = 0.0
            with tqdm(enumerate(self.dataloader), unit="batch") as pbar:
                for i, sample_batched in pbar:

                    if len(sample_batched[0]) > 1:
                        pos_u = sample_batched[0].to(self.device)
                        pos_v = sample_batched[1].to(self.device)
                        neg_v = sample_batched[2].to(self.device)

                        optimizer.zero_grad()
                        loss = self.skip_gram_model.forward(
                            pos_u, pos_v, neg_v)
                        loss.backward()
                        optimizer.step()
                        scheduler.step()
                        running_loss = running_loss * 0.9 + loss.item() * 0.1
                        pbar.set_postfix({"loss": running_loss})

            self.skip_gram_model.save_embedding(
                self.data.id2word, self.output_file_name)


if __name__ == '__main__':
    w2v = Word2VecTrainer(input_file=sys.argv[1], output_file="out.vec")
    w2v.train()

import matplotlib as mpl#NOQA
mpl.use('Agg')
import matplotlib.pyplot as plt
from functools import partial
import numpy as np
from clize import run

from skimage.io import imsave

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils

from machinedesign.viz import grid_of_images_default

from model import Gen
from model import Discr
from model import Clf

from data import load_dataset


from scipy.spatial.distance import cdist
from lapjv import lapjv

def compute_objectness(v):
    marginal = v.mean(dim=0)
    marginal = marginal.repeat(v.size(0), 1)
    score = v * torch.log(v / (marginal))
    score = score.sum(dim=1).mean()
    #score = math.exp(score)
    return score


def grid_embedding(h):
    assert int(np.sqrt(h.shape[0])) ** 2 == h.shape[0], 'Nb of examples must be a square number'
    size = np.sqrt(h.shape[0])
    grid = np.dstack(np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))).reshape(-1, 2)
    cost_matrix = cdist(grid, h, "sqeuclidean").astype('float32')
    cost_matrix = cost_matrix * (100000 / cost_matrix.max())
    _, rows, cols = lapjv(cost_matrix)
    return rows



def save_weights(m, folder='out', prefix=''):
    if isinstance(m, nn.Linear):
        w = m.weight.data
        if np.sqrt(w.size(1)) == int(w.size(1)):
            s = int(np.sqrst(w.size(1)))
            w = w.view(w.size(0), 1, s, s)
            gr = grid_of_images_default(np.array(w.tolist()), normalize=True)
            imsave('{}/{}_feat_{}.png'.format(folder, prefix, w.size(0)), gr)
    elif isinstance(m, nn.ConvTranspose2d):
        w = m.weight.data
        if w.size(1) == 1:
            w = w.view(w.size(0) * w.size(1), w.size(2), w.size(3))
            gr = grid_of_images_default(np.array(w.tolist()), normalize=True)
            imsave('{}/{}_feat_{}.png'.format(folder, prefix, w.size(0)), gr)
        elif w.size(1) == 3:
            gr = grid_of_images_default(np.array(w.tolist()), normalize=True)
            imsave('{}/{}_feat_{}.png'.format(folder, prefix, w.size(0)), gr)


def clf(*, folder='out', dataset='celeba', no=10):
    lr = 1e-4
    batch_size = 64
    train = load_dataset(dataset, split='train')
    trainl = torch.utils.data.DataLoader(
        train, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=1
    )
    valid = load_dataset(dataset, split='valid')
    validl = torch.utils.data.DataLoader(
        valid, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=1
    )

    x0, _ = train[0]
    nc = x0.size(0)
    discr = Clf(nc=nc, no=no)
    discr = discr.cuda()
    opt = optim.Adam(discr.parameters(), lr=lr, betas=(0.5, 0.999))
    nb_epochs = 40
    avg_acc = 0.
    crit = nn.CrossEntropyLoss()
    max_valid_acc = 0
    for epoch in range(nb_epochs):
        discr.train()
        for X, y in trainl:
            X = Variable(X).cuda()
            y = Variable(y).cuda()
            discr.zero_grad()
            ypred = discr(X)
            loss = crit(ypred, y)
            _, m = ypred.max(1)
            acc = (m == y).float().mean().cpu().data[0]
            avg_acc = avg_acc * 0.9 + acc * 0.1
            loss.backward()
            opt.step()
        accs = []
        discr.eval()
        for X, y in validl:
            X = Variable(X).cuda()
            y = Variable(y).cuda()
            ypred = discr(X)
            _, m = ypred.max(1)
            accs.extend((m==y).float().data.cpu().numpy())
        valid_acc = np.mean(accs)
        if valid_acc > max_valid_acc:
            max_valid_acc = valid_acc
            torch.save(discr, '{}/clf.th'.format(folder))
        print('Epoch {:03d}/{:03d}, Avg acc train : {:.3f}, Acc valid : {:.3f}'.format(epoch + 1, nb_epochs, avg_acc, valid_acc))


def train(*, 
          folder='out', 
          dataset='mnist', 
          classifier='out/clf.th', 
          resume=False, 
          wasserstein=False, 
          batch_size=64, 
          nz=100):

    lr = 0.0002
    nb_epochs = 3000
    dataset = load_dataset(dataset, split='train')
    x0, _ = dataset[0]
    nc = x0.size(0)
    w = x0.size(1)
    h = x0.size(2)
    _save_weights = partial(save_weights, folder=folder, prefix='gan')
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=1
    )
    act = 'tanh'
    if resume:
        gen = torch.load('{}/gen.th'.format(folder))
        discr = torch.load('{}/discr.th'.format(folder))
    else:
        gen = Gen(nz=nz, nc=nc, act=act, w=w)
        discr = Discr(nc=nc, act='', w=w)
    clf = torch.load(classifier)
    clf.eval()
    if wasserstein:
        gen_opt = optim.RMSprop(gen.parameters(), lr=lr)
        discr_opt = optim.RMSprop(discr.parameters(), lr=lr)
    else:
        gen_opt = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
        discr_opt = optim.Adam(discr.parameters(), lr=lr, betas=(0.5, 0.999))

    input = torch.FloatTensor(batch_size, nc, w, h)
    noise = torch.FloatTensor(batch_size, nz, 1, 1)
    label = torch.FloatTensor(batch_size)

    if wasserstein:
        real_label = 1
        fake_label = -1
        criterion = lambda output, label:(output*label).mean()
    else:
        real_label = 1
        fake_label = 0
        criterion = nn.BCELoss()
    
    gen = gen.cuda()
    discr =  discr.cuda()
    input, label = input.cuda(), label.cuda()
    noise = noise.cuda()
    clf = clf.cuda()

    giter = 0
    diter = 0

    dreal_list = []
    dfake_list = []

    for epoch in range(nb_epochs):
        for i, (X, _), in enumerate(dataloader):
            if wasserstein:
                # clamp parameters to a cube
                for p in discr.parameters():
                    p.data.clamp_(-0.01, 0.01)
            # Update discriminator
            discr.zero_grad()
            batch_size = X.size(0)
            X = X.cuda()
            input.resize_as_(X).copy_(X)
            label.resize_(batch_size).fill_(real_label)
            inputv = Variable(input)
            labelv = Variable(label)
            output = discr(inputv)
            labelpred = output[:, 0:1] if wasserstein else nn.Sigmoid()(output[:, 0:1])
            errD_real = criterion(labelpred, labelv)
            errD_real.backward()
            D_x = labelpred.data.mean()
            dreal_list.append(D_x)
            noise.resize_(batch_size, nz, 1, 1).uniform_(-1, 1)
            noisev = Variable(noise)
            fake = gen(noisev)
            labelv = Variable(label.fill_(fake_label))
            output = discr(fake.detach())

            labelpred = output[:, 0:1] if wasserstein else nn.Sigmoid()(output[:, 0:1])
            errD_fake = criterion(labelpred, labelv)
            errD_fake.backward()
            D_G_z1 = labelpred.data.mean()
            dfake_list.append(D_G_z1)
            discr_opt.step()
            diter += 1
            
            # Update generator
            gen.zero_grad()
            fake = gen(noisev)
            labelv = Variable(label.fill_(real_label))
            output = discr(fake)
            labelpred = output[:, 0:1] if wasserstein else nn.Sigmoid()(output[:, 0:1])

            n = gen.transform(noisev)
            fake = gen(n)
            obj = clf(fake)[:, 0].mean()
            output = discr(fake)
            labelpred2 = output[:, 0:1] if wasserstein else nn.Sigmoid()(output[:, 0:1])

            errG = criterion(labelpred, labelv)  + criterion(labelpred2, labelv) + 0.05 * obj
            errG.backward()
            gen_opt.step()
            print('{}/{} dreal : {:.6f} dfake : {:.6f} obj : {:.6f}'.format(epoch, nb_epochs, D_x, D_G_z1, obj.data[0]))
            if giter % 100 == 0:
                x = 0.5 * (X + 1) if act == 'tanh' else X
                f = 0.5 * (fake.data + 1) if act == 'tanh' else fake.data
                vutils.save_image(x, '{}/real_samples.png'.format(folder), normalize=True)
                vutils.save_image(f, '{}/fake_samples_epoch_{:03d}.png'.format(folder, epoch), normalize=True)
                torch.save(gen, '{}/gen.th'.format(folder))
                torch.save(discr, '{}/discr.th'.format(folder))
                gen.apply(_save_weights)
                fig = plt.figure()
                plt.plot(dreal_list, label='real')
                plt.plot(dfake_list, label='fake')
                plt.legend()
                plt.savefig('{}/discr.png'.format(folder))
                plt.close(fig)
            giter += 1

if __name__ == '__main__':
    run([train, clf])

import torch
import numpy as np

from scipy import linalg

encoder_out = torch.load("./encoder_out.pt", map_location=torch.device('cpu'))
encoder_mask = torch.load("./hs_mask.pt", map_location=torch.device('cpu'))

tao=0.05
lengths = encoder_mask.squeeze(-2).sum(-1)
dmodel = encoder_out.size(0)
lens_ls = []
reduced_hs = []
for i in range(len(lengths)):
    batch_i = encoder_out[i][:lengths[i]]
    q, r, p = linalg.qr(batch_i[:90].detach().cpu().numpy().transpose(0, 1), pivoting=True)
    diag = np.abs(np.diag(r))
    print(lengths[i])
    print(np.all(diag[1:] <= diag[:-1]))
    if np.all(diag[1:] <= diag[:-1]):
        diag_norm = diag / np.max(diag)
        print(diag_norm)
        place_threshold = (diag_norm <= tao).sum()
        print(place_threshold)
        len_i = lengths[i] - place_threshold
        print(len_i)
        lens_ls.append(len_i)
        reduced_hs.append(batch_i[p[:len_i]])
    else:
        lens_ls.append(lengths[i])
        reduced_hs.append(batch_i)

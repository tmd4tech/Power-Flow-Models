import time, torch
import gnn as G
tr = G.idx("n1_fold_0/train"); sc = G.Scalers(tr)
X, M, Y = sc.x(tr), G.edge_mask(tr), sc.y(tr)
for bs in [128, 512, 1024]:
    try:
        torch.manual_seed(0)
        m = G.GNN(8).to(G.DEVICE); opt = torch.optim.Adam(m.parameters(), 1e-3)
        for rep in range(2):                     # first pass warms up the GPU
            if G.DEVICE.type == "cuda": torch.cuda.synchronize()
            t = time.time(); perm = torch.randperm(len(tr), device=G.DEVICE)
            for k in range(0, len(tr), bs):
                b = perm[k:k + bs]; opt.zero_grad()
                loss = ((m(X[b], M[b]) - Y[b]) ** 2).mean(); loss.backward(); opt.step()
            if G.DEVICE.type == "cuda": torch.cuda.synchronize()
        print(f"batch {bs:5d}: {time.time() - t:6.1f} s per epoch, depth 8, {len(tr)} samples")
    except torch.cuda.OutOfMemoryError:
        print(f"batch {bs:5d}: out of GPU memory"); torch.cuda.empty_cache()

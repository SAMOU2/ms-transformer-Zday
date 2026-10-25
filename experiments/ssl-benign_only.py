import numpy as np, torch, torch.nn.functional as F, json, time
from sklearn.metrics import roc_auc_score
from model_ms import TokenizedMSEncoder, supcon_loss, anomaly_score
torch.manual_seed(0); np.random.seed(0)
meta=json.load(open("data/processed/meta.json")); n_net,n_sys=meta["n_net"],meta["n_sys"]
Xtr=np.load("data/processed/X_train.npy"); ytr=np.load("data/processed/y_train.npy",allow_pickle=True)
Xte=np.load("data/processed/X_test.npy"); yte=np.load("data/processed/y_test.npy",allow_pickle=True)
bm=np.array(["BENIGN" in str(l).upper() for l in ytr]); bt=np.array(["BENIGN" in str(l).upper() for l in yte])
Xb=torch.FloatTensor(Xtr[bm]); tb=(~bt).astype(int)
print(f"SSL sur {len(Xb):,} bénins uniquement")
m=TokenizedMSEncoder(n_net=n_net,n_sys=n_sys); opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-4)
t0=time.time()
for ep in range(10):
    m.train(); perm=torch.randperm(len(Xb)); tot=0; nb=0
    for i in range(0,len(Xb),1024):
        xb=Xb[perm[i:i+1024]]
        if len(xb)<8: continue
        z1=m(xb)["z"]; z2=m(xb+torch.randn_like(xb)*0.05)["z"]
        z=torch.cat([z1,z2])
        # SimCLR : la seule paire positive est (i, i+B) — les autres sont des négatifs
        yl=torch.cat([torch.arange(len(xb)), torch.arange(len(xb))])
        loss=supcon_loss(z,yl)
        opt.zero_grad(); loss.backward(); opt.step(); tot+=loss.item(); nb+=1
    print(f"  epoch {ep+1}/10 loss={tot/nb:.4f}")
m.eval()
def emb(X):
    o=[]
    with torch.no_grad():
        for i in range(0,len(X),4096): o.append(m(torch.FloatTensor(X[i:i+4096]))["z"])
    return torch.cat(o)
Zb=emb(Xtr[bm]); cent=F.normalize(Zb.mean(0),dim=0)
Zt=emb(Xte)
sc_all=anomaly_score(Zt,cent).numpy()
print(f"\nAUROC zero-day (SSL bénin) = {roc_auc_score(tb,sc_all):.4f}  [{time.time()-t0:.0f}s]")
for cls in sorted(set(yte)):
    if "BENIGN" in cls.upper(): continue
    mk=np.array([str(l)==cls for l in yte])
    y2=np.r_[np.zeros(bt.sum()),np.ones(mk.sum())]; s2=np.r_[sc_all[bt],sc_all[mk]]
    print(f"  {cls:<28}{roc_auc_score(y2,s2):>8.4f}")
torch.save({"state":m.state_dict(),"centroid":cent}, "/tmp/ssl_model.pt")

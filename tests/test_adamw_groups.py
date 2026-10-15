"""AdamW grouping excludes scalar/vector parameters and preserves legacy defaults."""
import copy,sys,unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from train_joint import make_optimizer

class AdamWGroupTests(unittest.TestCase):
    def test_groups_partition_trainable_parameters(self):
        model=torch.nn.Sequential(torch.nn.Linear(4,4),torch.nn.LayerNorm(4))
        cfg=dict(learning_rate=3e-4,weight_decay=.01,adamw_decay_policy='matrix_only',adamw_betas=[.9,.999],adamw_eps=1e-8)
        opt=make_optimizer(model,cfg)
        ids=[id(p) for g in opt.param_groups for p in g['params']]
        self.assertEqual(len(ids),len(set(ids)))
        self.assertEqual(set(ids),{id(p) for p in model.parameters()})
        for g in opt.param_groups:
            for p in g['params']:self.assertEqual(g['weight_decay'],.01 if p.ndim>=2 else 0.)
        initial={n:p.detach().clone() for n,p in model.named_parameters()}
        for p in model.parameters():p.grad=torch.zeros_like(p)
        opt.step()
        for n,p in model.named_parameters():
            expected=initial[n]*(1-3e-4*.01) if p.ndim>=2 else initial[n]
            torch.testing.assert_close(p,expected,rtol=0,atol=0)

    def test_legacy_optimizer_matches_default_adamw(self):
        a=torch.nn.Linear(2,2);b=copy.deepcopy(a);cfg=dict(learning_rate=.001,weight_decay=.01)
        first=make_optimizer(a,cfg);second=torch.optim.AdamW(b.parameters(),lr=.001,weight_decay=.01)
        for p,q in zip(a.parameters(),b.parameters()):p.grad=torch.ones_like(p);q.grad=torch.ones_like(q)
        first.step();second.step()
        for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p,q,rtol=0,atol=0)

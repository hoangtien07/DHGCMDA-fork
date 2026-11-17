"""Paired gradient probe on deep-copied models with restored Torch RNG."""
import copy
import hashlib
import torch

def _sha(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def run_pair(model,old_graph,new_graph,objective):
    state=torch.get_rng_state().clone()
    original={k:v.detach().clone() for k,v in model.state_dict().items()}
    def once(graph):
        clone=copy.deepcopy(model);clone.train(model.training);clone.zero_grad(set_to_none=True)
        torch.set_rng_state(state.clone())
        loss,details=objective(clone,graph)
        assert loss.ndim==0 and bool(torch.isfinite(loss))
        loss.backward()
        grads=torch.cat([p.grad.detach().reshape(-1) if p.grad is not None else torch.zeros_like(p).reshape(-1)
                         for p in clone.parameters()])
        assert bool(torch.isfinite(grads).all())
        return {'loss':float(loss.detach()),'grad_norm':float(grads.double().norm()),
                'grad_sha256':_sha(grads),'end_rng_sha256':_sha(torch.get_rng_state()),'details':details},grads
    try:
        old,go=once(old_graph);new,gn=once(new_graph);repeat,gr=once(new_graph)
        same=(new==repeat and torch.equal(gn,gr))
        loss_rel=abs(new['loss']-repeat['loss'])/max(1.,abs(new['loss']))
        grad_rel=float((gn.double()-gr.double()).norm())/max(1.,float(gn.double().norm()))
        rng_same=new['end_rng_sha256']==repeat['end_rng_sha256']
        within=loss_rel<=1e-5 and grad_rel<=1e-4 and rng_same
        dot=float(torch.dot(go.double(),gn.double()));den=float(go.double().norm()*gn.double().norm())
        result={'old':old,'new':new,'repeat':repeat,'repeat_identical':bool(same),
                'repeat_within_tolerance':bool(within),'repeat_loss_scaled_abs_difference':loss_rel,
                'repeat_gradient_relative_difference':grad_rel,'repeat_final_rng_equal':bool(rng_same),
                'grad_difference_norm':float((gn.double()-go.double()).norm()),
                'grad_cosine':dot/den if den else None,
                'final_rng_equal':old['end_rng_sha256']==new['end_rng_sha256']}
        assert within,('repeat gate failed',loss_rel,grad_rel,rng_same)
        assert all(torch.equal(original[k],v) for k,v in model.state_dict().items())
        return result
    finally:
        torch.set_rng_state(state)

"""Declared reduction-only control; no tensors detached or RNG drawn here."""
import math

def combine_objectives(l_type,it,et,rt,mass,cfg):
    mass=float(mass)
    if not math.isfinite(mass) or mass<=0:
        raise ValueError('common weighted-positive mass must be finite and >0')
    mode=cfg['unit_mode']
    loss=l_type
    if mode=='c5_means':
        if it is not None:loss=loss+it['lam']*it['L_intra']
        if et is not None:loss=loss+et['lam']*et['L_inter']
        if rt is not None:loss=loss+rt['lam']*rt['L_recon']
    elif mode=='common_mass_sums':
        if it is not None:loss=loss+it['lam']*((it['n_mi']+it['n_dis'])*it['L_intra'])/mass
        if et is not None:loss=loss+et['lam']*(et['info_sum_live']+float(cfg['inter_margin_coef'])*et['margin_live'])/mass
        if rt is not None:loss=loss+rt['lam']*((rt['recon_valid_mi']+rt['recon_valid_dis'])*rt['L_recon'])/mass
    else:raise ValueError('unknown unit_mode '+str(mode))
    return loss

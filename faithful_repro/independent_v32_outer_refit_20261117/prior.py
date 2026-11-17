import numpy as np

def fit_counts(pairs, y, n_mi, n_dis):
    pairs=np.asarray(pairs);y=np.asarray(y)
    if pairs.ndim!=2 or pairs.shape[1]!=2 or y.ndim!=2 or len(y)!=len(pairs) or not np.isin(y,[0,1]).all() or not (y.sum(1)>0).all():raise ValueError('invalid pairs or labels')
    if pairs.min()<0 or pairs[:,0].max()>=n_mi or pairs[:,1].max()>=n_dis:raise ValueError('pair out of bounds')
    c_m=np.zeros((n_mi,y.shape[1]),dtype=np.float64);c_d=np.zeros((n_dis,y.shape[1]),dtype=np.float64)
    np.add.at(c_m,pairs[:,0],y);np.add.at(c_d,pairs[:,1],y);glob=y.sum(0).astype(float)
    if not (glob>0).all():raise ValueError('empty class')
    return {'mi':c_m,'dis':c_d,'global':glob}

def predict_counts(counts,pairs):
    pairs=np.asarray(pairs);g=counts['global'];m=counts['mi'][pairs[:,0]].copy();d=counts['dis'][pairs[:,1]].copy()
    m[m.sum(1)==0]=g;d[d.sum(1)==0]=g;m+=(g/g.sum())*1e-12;d+=(g/g.sum())*1e-12
    return {'global':np.broadcast_to(g,(len(pairs),len(g))).copy(),'disease':d,'mirna':m,'product_laplace1':np.log1p(counts['mi'][pairs[:,0]])+np.log1p(counts['dis'][pairs[:,1]])-np.log1p(g)}

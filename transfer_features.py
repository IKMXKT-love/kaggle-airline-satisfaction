"""Train-local, source-informed features for the bounded transfer experiment."""
import numpy as np
import pandas as pd

FD = 'Flight Distance'
FREQUENCY_GROUPS = [[FD], ['Age'], ['Departure Delay in Minutes'],
                    ['Arrival Delay in Minutes'], [FD, 'Class'], [FD, 'Type of Travel']]
RATINGS = ['Seat comfort', 'Inflight entertainment']
CONDITION = ['Customer Type', 'Type of Travel', 'Class']


def key(frame, columns):
    result = pd.Series('', index=frame.index, dtype='string')
    for c in columns:
        values = (frame[c].astype('float64').astype('string') if
                  pd.api.types.is_numeric_dtype(frame[c]) else frame[c].astype('string'))
        values = values.fillna('__NA__')
        result = result + values.str.len().astype(str) + ':' + values
    return result


class TransferFeatures:
    def fit(self, train, source):
        assert not {'satisfaction', 'id'} & set(train)
        assert not {'satisfaction', 'id'} & set(source)
        self.frequency = []
        for columns in FREQUENCY_GROUPS:
            a, b = key(train, columns).value_counts(), key(source, columns).value_counts()
            self.frequency.append((columns, a, b, len(train), len(source), len(a.index.union(b.index))+1))
        self.profiles = []
        for name, frame, columns in [('source_fd', source, [FD]),
                                     ('source_context', source, CONDITION),
                                     ('fit_fd', train, [FD])]:
            k = key(frame, columns)
            tables = {}
            for c in RATINGS:
                s = frame[c].astype(float)
                z = pd.DataFrame(dict(k=k, value=s, square=s*s,
                                      high=(s>=4).astype(float), zero=(s==0).astype(float)))
                tables[c] = (z.groupby('k').agg(n=('value','count'), total=('value','sum'),
                             square=('square','sum'), high=('high','sum'), zero=('zero','sum')),
                             dict(mean=float(s.mean()), square=float((s*s).mean()),
                                  high=float((s>=4).mean()), zero=float((s==0).mean())))
            self.profiles.append((name, columns, tables))
        k = key(train, [FD])
        self.route_n = k.value_counts()
        self.class_props = {}
        for value in ['Business', 'Eco', 'Eco Plus']:
            v = train['Class'].astype(str).eq(value).astype(float)
            self.class_props[value] = (pd.DataFrame(dict(k=k,v=v)).groupby('k').v.sum(), float(v.mean()))
        return self

    def transform(self, frame, mode):
        if mode not in ['frequency', 'profiles', 'combined']:
            raise ValueError(mode)
        result = {}
        if mode in ['frequency', 'combined']:
            for i, (columns, a, b, na, nb, cardinality) in enumerate(self.frequency):
                k = key(frame, columns)
                ca, cb = k.map(a).fillna(0).to_numpy(float), k.map(b).fillna(0).to_numpy(float)
                pa, pb = (ca+1)/(na+cardinality), (cb+1)/(nb+cardinality)
                ratio = np.log(pa/pb)
                ratio[(ca==0)&(cb==0)] = 0
                result[f'transfer_freq_{i}_logratio'] = np.clip(ratio,-6,6)
                result[f'transfer_freq_{i}_source_support'] = np.log1p(cb)
                result[f'transfer_freq_{i}_source_unseen'] = (cb==0).astype(float)
                result[f'transfer_freq_{i}_fit_rare'] = (ca<10).astype(float)
        if mode in ['profiles', 'combined']:
            for name, columns, tables in self.profiles:
                k = key(frame, columns)
                for c, (table, global_) in tables.items():
                    n = k.map(table.n).fillna(0).to_numpy(float)
                    def shrink(column, prior):
                        return (k.map(table[column]).fillna(0).to_numpy(float)+20*prior)/(n+20)
                    mean = shrink('total',global_['mean'])
                    square = shrink('square',global_['square'])
                    sd = np.sqrt(np.maximum(square-mean*mean,0))
                    prefix = name+'__'+c
                    result[prefix+'__sd'] = sd
                    result[prefix+'__high'] = shrink('high',global_['high'])
                    result[prefix+'__zero'] = shrink('zero',global_['zero'])
                    result[prefix+'__deviation'] = np.clip((frame[c].to_numpy(float)-mean)/np.maximum(sd,.5),-6,6)
                result[name+'__support'] = np.log1p(n)
            k = key(frame,[FD])
            n = k.map(self.route_n).fillna(0).to_numpy(float)
            for value,(counts,prior) in self.class_props.items():
                result['route_class_share__'+value] = (k.map(counts).fillna(0).to_numpy(float)+20*prior)/(n+20)
        out = pd.DataFrame(result,index=frame.index).astype('float32')
        assert np.isfinite(out.to_numpy()).all()
        return out

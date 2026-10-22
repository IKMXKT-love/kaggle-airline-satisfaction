"""Resource-only adapters around pinned TabICL 2.2.0; no library edits."""
import gc
import sys
import time
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / '.artifact_runtime/tabicl_finetune_deps'),
               str(ROOT / '.artifact_runtime/tabicl_runtime')]

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, log_loss
from tabicl import TabICLClassifier, FinetunedTabICLClassifier
from tabicl._model.kv_cache import TabICLCache
from tabicl._finetune.base import ValidationMetrics


def release(model):
    if model is not None:
        model.model_kv_cache_ = None
        if hasattr(model, 'model_'):
            model.model_.clear_cache()
    gc.collect()
    torch.cuda.empty_cache()


class CPUCacheClassifier(TabICLClassifier):
    """Keep identical KV tensors on CPU; transfer one ensemble view at a time."""

    def _build_kv_cache(self):
        train_data = self.ensemble_generator_.transform(X=None, mode='train')
        self.model_kv_cache_ = OrderedDict()
        for norm_method, (xs, ys) in train_data.items():
            caches = []
            for start in range(len(xs)):
                xb = torch.from_numpy(xs[start:start+1]).float().to(self.device_)
                yb = torch.from_numpy(ys[start:start+1]).float().to(self.device_)
                with torch.no_grad():
                    self.model_.forward_with_cache(
                        X_train=xb, y_train=yb, use_cache=False, store_cache=True,
                        cache_mode=self.cache_mode_, inference_config=self.inference_config_)
                caches.append(self.model_._cache.to('cpu'))
                self.model_.clear_cache()
                del xb, yb
            self.model_kv_cache_[norm_method] = TabICLCache.concat(caches)
            del caches

    def _batch_forward_with_cache(self, xs, kv_cache):
        outputs = []
        for start in range(len(xs)):
            gpu_cache = kv_cache.slice_batch(start, start+1).to(self.device_)
            outputs.append(super()._batch_forward_with_cache(xs[start:start+1], gpu_cache))
            del gpu_cache
        return np.concatenate(outputs, axis=0)


class SequentialViewsClassifier(TabICLClassifier):
    """Same four views/logits, processing a complete query set one view at a time."""

    def _build_kv_cache(self):
        # Context caches are constructed lazily per view in predict_all.
        self.model_kv_cache_ = None

    def predict_all(self, x, query_batch=1024, maximum_seconds=1200, progress=False):
        encoded = self.X_encoder_.transform(x)
        training = self.ensemble_generator_.transform(X=None, mode='train')
        testing = self.ensemble_generator_.transform(encoded, mode='test')
        average = None
        members = 0
        start = time.time()
        for norm, (xs, ys) in training.items():
            queries = testing[norm][0]
            shuffles = self.ensemble_generator_.class_shuffles_[norm]
            for view in range(len(xs)):
                xb = torch.from_numpy(xs[view:view+1]).float().to(self.device_)
                yb = torch.from_numpy(ys[view:view+1]).float().to(self.device_)
                with torch.no_grad():
                    self.model_.forward_with_cache(X_train=xb, y_train=yb, use_cache=False,
                        store_cache=True, cache_mode='kv', inference_config=self.inference_config_)
                cache = self.model_._cache
                self.model_.clear_cache()
                del xb, yb
                chunks = []
                for begin in range(0, len(encoded), query_batch):
                    q = torch.from_numpy(queries[view:view+1,begin:begin+query_batch]).float().to(self.device_)
                    with torch.no_grad():
                        logits = self.model_.forward_with_cache(X_test=q, cache=cache,
                            return_logits=True, softmax_temperature=self.softmax_temperature,
                            inference_config=self.inference_config_)
                    chunks.append(logits.float().cpu().numpy()[0])
                    del q, logits
                    elapsed = time.time()-start
                    if elapsed > maximum_seconds:
                        raise TimeoutError('Sequential inference time limit exceeded')
                    if progress and begin > 0 and begin % 32768 == 0:
                        print('VIEW_QUERY_PROGRESS', norm, view, begin, len(encoded), flush=True)
                output = np.concatenate(chunks, axis=0)
                if average is None:
                    average = np.zeros_like(output)
                average += output[...,shuffles[view]]
                members += 1
                del cache, output, chunks
                if progress:
                    print('VIEW_COMPLETE', norm, view, 'seconds', round(time.time()-start,1), flush=True)
        average /= members
        average = self.softmax(average, axis=-1, temperature=self.softmax_temperature)
        return average / average.sum(axis=1, keepdims=True)


def inference_parameters(checkpoint, estimators=4):
    return dict(n_estimators=estimators, batch_size=1, kv_cache=True,
                norm_methods=['none', 'power'], feat_shuffle_method='latin',
                class_shuffle_method='shift', softmax_temperature=.9, average_logits=True,
                model_path=str(checkpoint), allow_auto_download=False, device='cuda',
                use_amp=True, use_fa3=False, offload_mode='auto', random_state=42,
                n_jobs=4, verbose=False)


class BoundedFinetuner(FinetunedTabICLClassifier):
    """Keep official training loop; bound validation/final contexts to fit rows."""

    def _create_inner_estimator(self, *, n_estimators, device):
        kwargs = dict(self.extra_classifier_kwargs or {})
        kwargs.update(inference_parameters(self.model_path, n_estimators))
        kwargs['device'] = device
        inner = SequentialViewsClassifier(**kwargs)
        original_fit = inner.fit

        def fit_capped(x, y):
            positions = self.context_positions_
            assert len(x) == self.training_rows_, 'Unexpected full-fit row order/size'
            assert len(np.unique(positions)) == len(positions)
            assert positions.min() >= 0 and positions.max() < len(x)
            return original_fit(np.asarray(x)[positions], np.asarray(y)[positions])

        inner.fit = fit_capped
        return inner

    def _run_validation(self, inner, x_train, y_train, x_val, y_val):
        import time
        start = time.time()
        try:
            inner.fit(x_train, y_train)
            prediction = inner.predict_all(x_val, maximum_seconds=300)[:, 1]
            assert np.isfinite(prediction).all()
            metric = float(roc_auc_score(y_val, prediction))
            row = dict(validation_call=len(self.validation_trace_), auc=metric,
                       seconds=time.time()-start, rows=len(y_val),
                       context_rows=len(self.context_positions_))
            self.validation_trace_.append(row)
            print('INNER_VALIDATION', row, flush=True)
            return ValidationMetrics(primary=metric,
                                     secondary=dict(log_loss=float(log_loss(y_val, prediction))))
        finally:
            release(inner)

    def _make_experiment_logger(self):
        return self.run_logger_

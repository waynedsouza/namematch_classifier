from transformers import Trainer, TrainingArguments , is_torch_tpu_available
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import torch
from datetime import datetime
import os
if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl
class WTrainer(Trainer):
    best_scores=[0.7 , 0.7 , 0.7]
    best_loss=999
    def __init__(self,*args, **kwargs):
        #print("args" , args)
        #print("kwargs" , kwargs['args'])
        self.prefix = kwargs['args'].output_dir
        super(WTrainer,self).__init__(*args, **kwargs)

    def logscores(self, _metrics , path):
        #_acc,_f1 , _precision_score , _recall_score
        with open(path , 'a') as f:
            _score = ''
            for k,v in _metrics.items():
                _score+=str(k)
                _score+=':'
                _score+=str(v )
                _score+='   '
            f.write(_score)
    def updateBestScores(self,_metric):
        _acc=_metric['eval_accuracy']
        _f1=_metric['eval_f1']
        _precision_score=_metric['eval_precision']
        _recall_score=_metric['eval_recall']
        _loss = _metric['eval_loss']
        if _acc> self.best_scores[0]:
            self.best_scores[0]=_acc
        if _precision_score> self.best_scores[1]:
            self.best_scores[1]=_precision_score
        if _recall_score> self.best_scores[2]:
            self.best_scores[2]=_recall_score
        if _loss< self.best_loss:
            self.best_loss=_loss

    def doStuff(self ,_metric , model):
        prefix=self.prefix
        _acc=_metric['eval_accuracy']
        _f1=_metric['eval_f1']
        _precision_score=_metric['eval_precision']
        _recall_score=_metric['eval_recall']
        _loss = _metric['eval_loss']
        if not os.path.isdir(self.prefix+'/output/'):
            try:
                os.makedirs(self.prefix+'/output/')
            except Exception as e:
                print("Could not create output directory " , e)
            
        try:

            if _acc> self.best_scores[0] and _precision_score >  self.best_scores[1] and _recall_score >  self.best_scores[2]:
                model_save_path = self.prefix+'/output/%s-best_scores_-'%prefix+'-'+datetime.now().strftime("%Y-%m-%d")
                torch.save(model , model_save_path)
                self.logscores(_metric , self.prefix+"/output/%s-bestscores.log"%prefix)
            elif _acc> self.best_scores[0] and _acc> 0.9:
                model_save_path = self.prefix+'/output/%s-best_acc_-'%prefix+'-'+datetime.now().strftime("%Y-%m-%d")
                torch.save(model , model_save_path)
                self.logscores(_metric , self.prefix+"/output/%s-bestacc.log"%prefix)
            elif _acc> 0.85:
                self.logscores( _metric , self.prefix+"/output/%s-opploss.log"%prefix)
                if _loss<self.best_loss:
                    model_save_path = self.prefix+'/output/%s-best_loss_-'%prefix+'-'+datetime.now().strftime("%Y-%m-%d")
                    torch.save(model , model_save_path)
                    self.logscores(_metric , self.prefix+"/output/%s-bestloss.log"%prefix)
        except Exception:
            pass
        self.updateBestScores(_metric)


    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log:
            if is_torch_tpu_available():
                xm.mark_step()

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            if isinstance(self.eval_dataset, dict):
                for eval_dataset_name, eval_dataset in self.eval_dataset.items():
                    metrics = self.evaluate(
                        eval_dataset=eval_dataset,
                        ignore_keys=ignore_keys_for_eval,
                        metric_key_prefix=f"eval_{eval_dataset_name}",
                    )
            else:
                metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, self.state.global_step, metrics)
            self.doStuff(metrics , model)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)
